#!/usr/bin/env python3
"""Install chain v2: (1) manager.touch forwards req_context (second fix for
information starvation) (2) chain.py v2 = identity-preserving touch + denoised
reuse counting + highest-priority hint guard.
Discipline: manager already has .bak-chain (pristine) as the undo anchor; this
patch adds a uniqueness check + ast.
Usage: python setup_chain_v2.py / python setup_chain_v2.py undo (full restore to pristine + re-register)
"""
import ast
import shutil
import sys
from pathlib import Path

import vllm

VROOT = Path(vllm.__file__).parent
MANAGER = VROOT / "v1/kv_offload/cpu/manager.py"
CHAIN_FILE = VROOT / "v1/kv_offload/cpu/policies/chain.py"

TOUCH_OLD = '''    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self._policy.touch(keys)'''
TOUCH_NEW = '''    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        if hasattr(self._policy, "touch_ctx"):
            self._policy.touch_ctx(keys, req_context)
        else:
            self._policy.touch(keys)'''

CHAIN_V2 = '''# SPDX-License-Identifier: Apache-2.0
"""ChainCachePolicy v2: revisit-aware guarding.
v1 -> v2:
  - identity-preserving touch (no longer rebuilds cid), fixing a hidden v1 defect
  - denoised reuse counting: count a reuse on touch only if the chain has had no
    insert since its last touch
    (a t2 lookup-touch precedes its insert -> counted; a post-store touch follows
     the insert -> not counted)
  - hint: touch_ctx reads kv_transfer_params["chain_hint"]=="sticky" -> highest guard priority
  - guard ranking: (sticky desc, reuse_count desc, last_reuse desc), GUARD=2
  - eviction and write-rejection same as v1: whole-chain MRU, reject if it would break a guarded chain
"""
import itertools
import logging
from collections.abc import Iterable

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy

logger = logging.getLogger("vllm")

GUARD_CHAINS = 2


class _Chain:
    __slots__ = ("blocks", "last_touch", "last_reuse", "reuse_count",
                 "inserted_since_touch", "sticky")

    def __init__(self, clock: int):
        self.blocks: dict[int, OffloadKey] = {}
        self.last_touch = clock
        self.last_reuse = 0
        self.reuse_count = 0
        self.inserted_since_touch = False
        self.sticky = False


class ChainCachePolicy(CachePolicy):

    def __init__(self, cache_capacity: int):
        self.blocks: dict[OffloadKey, BlockStatus] = {}
        self.meta: dict[OffloadKey, tuple[int, int]] = {}
        self.chains: dict[int, _Chain] = {}
        self._clock = itertools.count(1)
        self._chain_ids = itertools.count()
        self._pending_chain: int | None = None

    # ---- internal ----
    def _assign(self, key: OffloadKey, cid: int, pos: int) -> None:
        old = self.meta.get(key)
        if old is not None:
            ocid, opos = old
            ch = self.chains.get(ocid)
            if ch is not None:
                ch.blocks.pop(opos, None)
                if not ch.blocks:
                    self.chains.pop(ocid, None)
        self.meta[key] = (cid, pos)
        self.chains[cid].blocks[pos] = key

    def _drop_meta(self, key: OffloadKey) -> None:
        m = self.meta.pop(key, None)
        if m is None:
            return
        cid, pos = m
        ch = self.chains.get(cid)
        if ch is not None:
            ch.blocks.pop(pos, None)
            if not ch.blocks:
                self.chains.pop(cid, None)

    def _touch_impl(self, keys: Iterable[OffloadKey], sticky: bool) -> None:
        self._pending_chain = None
        ks = [k for k in keys if k in self.blocks]
        if not ks:
            return
        clk = next(self._clock)
        # majority vote to determine membership in an existing chain (identity-preserving)
        votes: dict[int, int] = {}
        for k in ks:
            m = self.meta.get(k)
            if m is not None:
                votes[m[0]] = votes.get(m[0], 0) + 1
        if votes:
            cid = max(votes, key=lambda c: votes[c])
            ch = self.chains[cid]
            if not ch.inserted_since_touch:
                ch.reuse_count += 1
                ch.last_reuse = clk
            ch.inserted_since_touch = False
            ch.last_touch = clk
            if sticky:
                ch.sticky = True
            # absorb keys not yet assigned / assigned to a different chain
            base = (max(ch.blocks) + 1) if ch.blocks else 0
            for k in ks:
                m = self.meta.get(k)
                if m is None or m[0] != cid:
                    self._assign(k, cid, base)
                    base += 1
        else:
            cid = next(self._chain_ids)
            self.chains[cid] = _Chain(clk)
            self.chains[cid].sticky = sticky
            for pos, k in enumerate(ks):
                self._assign(k, cid, pos)

    # ---- CachePolicy interface + extensions ----
    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.blocks.get(key)

    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.blocks[key] = block
        if key not in self.meta:
            if self._pending_chain is None or self._pending_chain not in self.chains:
                self._pending_chain = next(self._chain_ids)
                self.chains[self._pending_chain] = _Chain(next(self._clock))
            ch = self.chains[self._pending_chain]
            pos = (max(ch.blocks) + 1) if ch.blocks else 0
            self._assign(key, self._pending_chain, pos)
            ch.inserted_since_touch = True

    def remove(self, key: OffloadKey) -> None:
        del self.blocks[key]
        self._drop_meta(key)

    def touch(self, keys: Iterable[OffloadKey]) -> None:
        self._touch_impl(keys, sticky=False)

    def touch_ctx(self, keys: Iterable[OffloadKey], req_context) -> None:
        params = getattr(req_context, "kv_transfer_params", None) or {}
        self._touch_impl(keys, sticky=(params.get("chain_hint") == "sticky"))

    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []
        self._pending_chain = None
        ranked = sorted(
            self.chains.keys(),
            key=lambda c: (self.chains[c].sticky,
                           self.chains[c].reuse_count,
                           self.chains[c].last_reuse),
            reverse=True,
        )
        guarded = set(ranked[:GUARD_CHAINS])
        # non-guarded chains die newest-touched-first (MRU)
        order = sorted((c for c in self.chains if c not in guarded),
                       key=lambda c: self.chains[c].last_touch, reverse=True)
        candidates: list[tuple[OffloadKey, BlockStatus]] = []
        for cid in order:
            for pos in sorted(self.chains[cid].blocks, reverse=True):
                key = self.chains[cid].blocks[pos]
                block = self.blocks.get(key)
                if block is None or block.ref_cnt != 0 or key in protected:
                    continue
                candidates.append((key, block))
                if len(candidates) == n:
                    break
            if len(candidates) == n:
                break
        if len(candidates) < n:
            logger.info("[CHAIN-REJECT] need=%d got=%d chains=%d", n,
                        len(candidates), len(self.chains))
            return None
        logger.info("[CHAIN-EVICT] n=%d chains=%d guarded=%s", n,
                    len(self.chains),
                    [(self.chains[c].reuse_count, self.chains[c].sticky)
                     for c in ranked[:GUARD_CHAINS]])
        for key, _ in candidates:
            del self.blocks[key]
            self._drop_meta(key)
        return candidates
'''


def undo() -> None:
    b = MANAGER.with_suffix(".py.bak-chain")
    if b.exists():
        shutil.copy2(b, MANAGER)
        print("[OK] manager.py restored to pristine (registration also reverted; to reinstall, rerun setup_chain_policy.py + this script)")
    if CHAIN_FILE.exists():
        CHAIN_FILE.unlink()
        print("[OK] deleted chain.py")


def install() -> None:
    ast.parse(CHAIN_V2)
    src = MANAGER.read_text()
    ast.parse(src)
    if "touch_ctx" in src:
        print("[SKIP] manager.touch already has the forwarding patch")
    else:
        if src.count(TOUCH_OLD) != 1:
            sys.exit("[FATAL] manager.touch target fragment not unique, stopping")
        src = src.replace(TOUCH_OLD, TOUCH_NEW, 1)
        ast.parse(src)
        MANAGER.write_text(src)
        ast.parse(MANAGER.read_text())
        print("[OK] manager.touch forwarding patch written")
    if not CHAIN_FILE.exists():
        sys.exit("[FATAL] chain.py not found, run setup_chain_policy.py first to complete registration")
    CHAIN_FILE.write_text(CHAIN_V2)
    ast.parse(CHAIN_FILE.read_text())
    from importlib import import_module, invalidate_caches
    invalidate_caches()
    import_module("vllm.v1.kv_offload.cpu.policies.chain").ChainCachePolicy(64)
    print("[OK] chain.py upgraded to v2, import + instantiation passed")


if __name__ == "__main__":
    undo() if (len(sys.argv) > 1 and sys.argv[1] == "undo") else install()
