#!/usr/bin/env python3
"""Install chain v3 (overwrites chain.py; the manager touch_ctx forwarding patch
from v2 is reused).
v2 -> v3 (each item comes from a W-series failure):
  (1) guard by block budget (<=50% capacity) rather than chain count -- fixes
      "everyone guarded -> starvation" (zero EVICT / 1364 REJECT / pool frozen at 59 blocks)
  (2) ghost chains: metadata survives after a chain dies; get() on an absent key
      records probed, re-insertion claims the reuse credit -- fixes survivorship bias
  (3) [CHAIN-HINT] arrival-side logging -- to settle the W2 broken-pipe suspicion
Usage: python chain_v3_install.py
"""
import ast
from pathlib import Path

import vllm

CHAIN_FILE = Path(vllm.__file__).parent / "v1/kv_offload/cpu/policies/chain.py"

CHAIN_V3 = '''# SPDX-License-Identifier: Apache-2.0
"""ChainCachePolicy v3: block-budget guarding + ghost-chain revisit credit + hint.
Guarding: rank chains by (sticky, reuse_count, last_reuse) desc, accumulate guard
      up to the block budget (GUARD_BUDGET_FRAC * capacity); the non-guarded region
      is evicted whole-chain by MRU.
Ghost: after a chain's blocks are cleared, retain its key set and credit; get() on
      an absent key sets probed; when a key is re-inserted, the new chain claims the
      (reuse_count + probed) credit.
"""
import itertools
import logging
from collections.abc import Iterable

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy

logger = logging.getLogger("vllm")

GUARD_BUDGET_FRAC = 0.5
GHOST_KEY_CAP_FACTOR = 8  # total ghost keys <= capacity * this factor


class _Chain:
    __slots__ = ("blocks", "all_keys", "last_touch", "last_reuse",
                 "reuse_count", "inserted_since_touch", "sticky")

    def __init__(self, clock: int):
        self.blocks: dict[int, OffloadKey] = {}
        self.all_keys: set = set()
        self.last_touch = clock
        self.last_reuse = 0
        self.reuse_count = 0
        self.inserted_since_touch = False
        self.sticky = False


class _Ghost:
    __slots__ = ("keys", "reuse_count", "sticky", "probed", "born")

    def __init__(self, keys: set, reuse_count: int, sticky: bool, born: int):
        self.keys = keys
        self.reuse_count = reuse_count
        self.sticky = sticky
        self.probed = False
        self.born = born


class ChainCachePolicy(CachePolicy):

    def __init__(self, cache_capacity: int):
        self.capacity = cache_capacity
        self.guard_budget = max(1, int(cache_capacity * GUARD_BUDGET_FRAC))
        self.blocks: dict[OffloadKey, BlockStatus] = {}
        self.meta: dict[OffloadKey, tuple[int, int]] = {}
        self.chains: dict[int, _Chain] = {}
        self.ghosts: dict[int, _Ghost] = {}
        self.ghost_key2cid: dict[OffloadKey, int] = {}
        self._clock = itertools.count(1)
        self._chain_ids = itertools.count()
        self._pending_chain: int | None = None
        self._hint_logged = 0

    # ---- ghost ----
    def _make_ghost(self, cid: int, ch: _Chain) -> None:
        gid = cid
        self.ghosts[gid] = _Ghost(set(ch.all_keys), ch.reuse_count,
                                  ch.sticky, next(self._clock))
        for k in ch.all_keys:
            self.ghost_key2cid[k] = gid
        # prune
        total = len(self.ghost_key2cid)
        cap = self.capacity * GHOST_KEY_CAP_FACTOR
        if total > cap:
            for old_gid in sorted(self.ghosts, key=lambda g: self.ghosts[g].born):
                for k in self.ghosts[old_gid].keys:
                    self.ghost_key2cid.pop(k, None)
                del self.ghosts[old_gid]
                if len(self.ghost_key2cid) <= cap:
                    break

    def _adopt_ghost(self, key: OffloadKey, ch: _Chain) -> None:
        gid = self.ghost_key2cid.get(key)
        if gid is None:
            return
        g = self.ghosts.pop(gid, None)
        if g is None:
            self.ghost_key2cid.pop(key, None)
            return
        for k in g.keys:
            self.ghost_key2cid.pop(k, None)
        credit = g.reuse_count + (1 if g.probed else 0)
        if credit > ch.reuse_count:
            ch.reuse_count = credit
            ch.last_reuse = next(self._clock)
        ch.sticky = ch.sticky or g.sticky
        logger.info("[CHAIN-ADOPT] ghost=%d credit=%d sticky=%s", gid, credit,
                    g.sticky)

    # ---- internal ----
    def _assign(self, key: OffloadKey, cid: int, pos: int) -> None:
        old = self.meta.get(key)
        if old is not None:
            ocid, opos = old
            ch = self.chains.get(ocid)
            if ch is not None:
                ch.blocks.pop(opos, None)
                if not ch.blocks:
                    self._make_ghost(ocid, ch)
                    self.chains.pop(ocid, None)
        self.meta[key] = (cid, pos)
        self.chains[cid].blocks[pos] = key
        self.chains[cid].all_keys.add(key)

    def _drop_meta(self, key: OffloadKey) -> None:
        m = self.meta.pop(key, None)
        if m is None:
            return
        cid, pos = m
        ch = self.chains.get(cid)
        if ch is not None:
            ch.blocks.pop(pos, None)
            if not ch.blocks:
                self._make_ghost(cid, ch)
                self.chains.pop(cid, None)

    def _touch_impl(self, keys: Iterable[OffloadKey], sticky: bool) -> None:
        self._pending_chain = None
        ks = [k for k in keys if k in self.blocks]
        if not ks:
            return
        clk = next(self._clock)
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
        b = self.blocks.get(key)
        if b is None:
            gid = self.ghost_key2cid.get(key)
            if gid is not None:
                g = self.ghosts.get(gid)
                if g is not None and not g.probed:
                    g.probed = True
                    logger.info("[CHAIN-GHOST-PROBE] ghost=%d", gid)
        return b

    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.blocks[key] = block
        if key not in self.meta:
            if self._pending_chain is None or self._pending_chain not in self.chains:
                self._pending_chain = next(self._chain_ids)
                self.chains[self._pending_chain] = _Chain(next(self._clock))
            ch = self.chains[self._pending_chain]
            self._adopt_ghost(key, ch)
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
        if params and self._hint_logged < 3:
            self._hint_logged += 1
            logger.info("[CHAIN-HINT] params=%s", params)
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
        guarded: set[int] = set()
        used = 0
        for cid in ranked:
            sz = len(self.chains[cid].blocks)
            if used + sz > self.guard_budget:
                break
            guarded.add(cid)
            used += sz
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
            logger.info("[CHAIN-REJECT] need=%d got=%d chains=%d guard_used=%d/%d",
                        n, len(candidates), len(self.chains), used,
                        self.guard_budget)
            return None
        logger.info("[CHAIN-EVICT] n=%d chains=%d guard_used=%d/%d guarded=%s",
                    n, len(self.chains), used, self.guard_budget,
                    [(self.chains[c].reuse_count, self.chains[c].sticky)
                     for c in sorted(guarded)][:4])
        for key, _ in candidates:
            del self.blocks[key]
            self._drop_meta(key)
        return candidates
'''


def main() -> None:
    if not CHAIN_FILE.exists():
        raise SystemExit("[FATAL] chain.py not found")
    ast.parse(CHAIN_V3)
    CHAIN_FILE.write_text(CHAIN_V3)
    ast.parse(CHAIN_FILE.read_text())
    from importlib import import_module, invalidate_caches
    invalidate_caches()
    import_module("vllm.v1.kv_offload.cpu.policies.chain").ChainCachePolicy(64)
    print("[OK] chain.py upgraded to v3, import + instantiation passed")


if __name__ == "__main__":
    main()
