#!/usr/bin/env python3
"""Upgrade the installed chain.py to v1.
v0 lesson: under symmetric workloads the "head-run length" sort key degenerates,
and the secondary key (colder dies first) = LRU.
v1 policy (the correct answer for cyclic revisit):
  1) whole-chain in/out (eliminate fragmented partial hits)
  2) eviction order = most-recently-touched chain dies first (MRU-style -- under
     cyclic workloads the coldest one returns soonest)
  3) guard the oldest GUARD chains; if satisfying n would break a guarded chain
     -> return None
     => prepare_store atomic failure = write rejection = stable resident set
        (scan-resistant)
Usage: python chain_v1_install.py   (chain.py must already be installed by setup_chain_policy.py)
"""
import ast
from pathlib import Path

import vllm

CHAIN_FILE = Path(vllm.__file__).parent / "v1/kv_offload/cpu/policies/chain.py"

CHAIN_V1 = '''# SPDX-License-Identifier: Apache-2.0
"""ChainCachePolicy v1: whole-chain MRU eviction + guard oldest chains + write-rejection stable residency.
Applicability: cyclic / round-robin revisit workloads (multi-turn sessions).
v0 -> v1: under symmetric workloads v0's chain-length sort degenerates to LRU; v1
reverses the direction and uses prepare_store's atomic failure to implement write
rejection, converging to a stable resident set.
"""
import itertools
import logging
from collections.abc import Iterable

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy

logger = logging.getLogger("vllm")

GUARD_CHAINS = 2  # guard the oldest K chains


class ChainCachePolicy(CachePolicy):

    def __init__(self, cache_capacity: int):
        self.blocks: dict[OffloadKey, BlockStatus] = {}
        self.meta: dict[OffloadKey, tuple[int, int]] = {}
        self.chains: dict[int, dict[int, OffloadKey]] = {}
        self.chain_touch: dict[int, int] = {}
        self._clock = itertools.count()
        self._chain_ids = itertools.count()
        self._pending_chain: int | None = None

    def _assign(self, key: OffloadKey, cid: int, pos: int) -> None:
        old = self.meta.get(key)
        if old is not None:
            ocid, opos = old
            self.chains.get(ocid, {}).pop(opos, None)
            if not self.chains.get(ocid):
                self.chains.pop(ocid, None)
                self.chain_touch.pop(ocid, None)
        self.meta[key] = (cid, pos)
        self.chains.setdefault(cid, {})[pos] = key

    def _drop_meta(self, key: OffloadKey) -> None:
        m = self.meta.pop(key, None)
        if m is None:
            return
        cid, pos = m
        ch = self.chains.get(cid)
        if ch is not None:
            ch.pop(pos, None)
            if not ch:
                self.chains.pop(cid, None)
                self.chain_touch.pop(cid, None)

    # ---- CachePolicy interface ----
    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.blocks.get(key)

    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.blocks[key] = block
        if key not in self.meta:
            if self._pending_chain is None:
                self._pending_chain = next(self._chain_ids)
                self.chain_touch[self._pending_chain] = next(self._clock)
            ch = self.chains.get(self._pending_chain, {})
            pos = (max(ch) + 1) if ch else 0
            self._assign(key, self._pending_chain, pos)

    def remove(self, key: OffloadKey) -> None:
        del self.blocks[key]
        self._drop_meta(key)

    def touch(self, keys: Iterable[OffloadKey]) -> None:
        self._pending_chain = None
        ks = [k for k in keys if k in self.blocks]
        if not ks:
            return
        cid = next(self._chain_ids)
        self.chain_touch[cid] = next(self._clock)
        for pos, k in enumerate(ks):
            self._assign(k, cid, pos)

    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []
        self._pending_chain = None
        # the oldest GUARD chains are guarded; the rest are evicted whole-chain by "newest dies first" (MRU)
        by_age = sorted(self.chains.keys(), key=lambda c: self.chain_touch.get(c, 0))
        guarded = set(by_age[:GUARD_CHAINS])
        order = [c for c in reversed(by_age) if c not in guarded]
        candidates: list[tuple[OffloadKey, BlockStatus]] = []
        for cid in order:
            for pos in sorted(self.chains[cid], reverse=True):
                key = self.chains[cid][pos]
                block = self.blocks.get(key)
                if block is None or block.ref_cnt != 0 or key in protected:
                    continue
                candidates.append((key, block))
                if len(candidates) == n:
                    break
            if len(candidates) == n:
                break
        if len(candidates) < n:
            logger.info("[CHAIN-REJECT] need=%d got=%d chains=%d guarded=%d",
                        n, len(candidates), len(self.chains), len(guarded))
            return None  # write rejection: guarded chains stay intact, the new store is dropped
        logger.info("[CHAIN-EVICT] n=%d chains=%d guarded=%d", n,
                    len(self.chains), len(guarded))
        for key, _ in candidates:
            del self.blocks[key]
            self._drop_meta(key)
        return candidates
'''


def main() -> None:
    if not CHAIN_FILE.exists():
        raise SystemExit("[FATAL] chain.py not found, run setup_chain_policy.py first")
    ast.parse(CHAIN_V1)
    CHAIN_FILE.write_text(CHAIN_V1)
    ast.parse(CHAIN_FILE.read_text())
    from importlib import import_module, invalidate_caches
    invalidate_caches()
    import_module("vllm.v1.kv_offload.cpu.policies.chain").ChainCachePolicy(64)
    print("[OK] chain.py upgraded to v1, import + instantiation passed")


if __name__ == "__main__":
    main()
