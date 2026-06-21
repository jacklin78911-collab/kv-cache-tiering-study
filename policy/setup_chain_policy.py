#!/usr/bin/env python3
"""Install ChainCachePolicy v0 into vllm/v1/kv_offload/cpu/policies/ and register it.
Policy idea: a block's value in a prefix chain depends monotonically on its
predecessors (a hit is a contiguous head-run of the chain), so eviction should
  1) sacrifice whole short chains first (short head-run = low expected value)
  2) within a chain, retreat from the tail, protecting the head
Chain identification (v0 heuristic): touch() carries the request prefix's ordered
key sequence -> chain and positions learned directly; insert() batch adjacency as
a supplement. The strict version (req_context directly wired) is left to v1.
Discipline: .bak / uniqueness / ast verification before and after / undo.
Usage: python setup_chain_policy.py / python setup_chain_policy.py undo
"""
import ast
import shutil
import sys
from pathlib import Path

import vllm

POLICY_DIR = Path(vllm.__file__).parent / "v1/kv_offload/cpu/policies"
CHAIN_FILE = POLICY_DIR / "chain.py"
MANAGER = Path(vllm.__file__).parent / "v1/kv_offload/cpu/manager.py"

CHAIN_SRC = '''# SPDX-License-Identifier: Apache-2.0
"""ChainCachePolicy v0: chain-aware eviction.
Value model: a prefix hit is a contiguous head-run of the chain; the value of
block k depends on 0..k-1 all being present.
Eviction order: sort chains by (head-run length asc, last-touch asc) -> short/cold
chains leave first, whole-chain; within a chain retreat by position desc -> protect
the head.
Chain identification: touch(ordered prefix keys) is the primary signal; insert
batch adjacency is a supplement.
"""
import itertools
from collections.abc import Iterable

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


class ChainCachePolicy(CachePolicy):

    def __init__(self, cache_capacity: int):
        self.blocks: dict[OffloadKey, BlockStatus] = {}
        # key -> (chain_id, pos)
        self.meta: dict[OffloadKey, tuple[int, int]] = {}
        # chain_id -> {pos -> key}
        self.chains: dict[int, dict[int, OffloadKey]] = {}
        self.chain_touch: dict[int, int] = {}   # chain_id -> logical clock
        self._clock = itertools.count()
        self._chain_ids = itertools.count()
        self._pending_chain: int | None = None  # for insert batch adjacency

    # ---- internal ----
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

    def _head_run_len(self, cid: int) -> int:
        """Head-run length: number of consecutive positions starting from the minimum."""
        ch = self.chains.get(cid)
        if not ch:
            return 0
        positions = sorted(ch)
        run, prev = 1, positions[0]
        for p in positions[1:]:
            if p == prev + 1:
                run += 1
                prev = p
            else:
                break
        return run

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
        # ordered prefix sequence = ground-truth signal for the chain: rebuild chain membership and positions
        self._pending_chain = None  # touch acts as an insert batch boundary
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
        self._pending_chain = None  # evict also acts as a batch boundary
        # chain ordering: shorter head-run first, ties broken by colder first
        order = sorted(
            self.chains.keys(),
            key=lambda c: (self._head_run_len(c), self.chain_touch.get(c, 0)),
        )
        candidates: list[tuple[OffloadKey, BlockStatus]] = []
        for cid in order:
            # within a chain, retreat from the tail
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
            return None
        for key, _ in candidates:
            del self.blocks[key]
            self._drop_meta(key)
        return candidates
'''

REG_OLD_IMPORT = "from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy"
REG_NEW_IMPORT = (REG_OLD_IMPORT
                  + "\nfrom vllm.v1.kv_offload.cpu.policies.chain import ChainCachePolicy")
REG_OLD_DICT = '    "arc": ARCCachePolicy,'
REG_NEW_DICT = '    "arc": ARCCachePolicy,\n    "chain": ChainCachePolicy,'


def undo() -> None:
    b = MANAGER.with_suffix(".py.bak-chain")
    if b.exists():
        shutil.copy2(b, MANAGER)
        print(f"[OK] restored {MANAGER.name}")
    if CHAIN_FILE.exists():
        CHAIN_FILE.unlink()
        print(f"[OK] deleted {CHAIN_FILE.name}")


def install() -> None:
    ast.parse(CHAIN_SRC)
    src = MANAGER.read_text()
    ast.parse(src)
    if "ChainCachePolicy" in src:
        print("[SKIP] manager.py already registers chain")
    else:
        for old in (REG_OLD_IMPORT, REG_OLD_DICT):
            if src.count(old) != 1:
                sys.exit(f"[FATAL] manager.py target fragment not unique: {old[:50]!r}")
        b = MANAGER.with_suffix(".py.bak-chain")
        if not b.exists():
            shutil.copy2(MANAGER, b)
            print(f"[OK] backup -> {b.name}")
        src = src.replace(REG_OLD_IMPORT, REG_NEW_IMPORT, 1)
        src = src.replace(REG_OLD_DICT, REG_NEW_DICT, 1)
        ast.parse(src)
        MANAGER.write_text(src)
        ast.parse(MANAGER.read_text())
        print(f"[OK] registry updated {MANAGER}")
    CHAIN_FILE.write_text(CHAIN_SRC)
    ast.parse(CHAIN_FILE.read_text())
    print(f"[OK] wrote {CHAIN_FILE}")
    # smoke test: import + instantiate
    from importlib import import_module, invalidate_caches
    invalidate_caches()
    mod = import_module("vllm.v1.kv_offload.cpu.policies.chain")
    mod.ChainCachePolicy(64)
    print("[OK] import + instantiation smoke test passed")


if __name__ == "__main__":
    undo() if (len(sys.argv) > 1 and sys.argv[1] == "undo") else install()
