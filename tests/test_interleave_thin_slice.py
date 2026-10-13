# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Owner(s): ["module: cpp"]

"""
Interleaved tensor round trip on 1p0 in VF mode.

It drives one large **interleaved** (multi-chunk) device allocation through the
**eager** copy path — ``x.to("spyre")`` (H2D) then ``.cpu()`` (D2H) — under
the emulated 4-domain topology of ``FLEX_DEVICE=EMUL1p5``, and requires the
round-tripped values to be bit-exact and to match a ``Bind{0}`` baseline.

There is **no compute op and no torch.compile**: the compiled JobPlan builder
requires single-chunk addresses.

## Why one large fp32 tensor

* **Large:** in VF mode flex moves each chunk in DMA slices of at most
  ``FLEX_DATA_TRANSFER_BUFFER_SIZE`` bytes (256 MiB by default). The tensor is
  sized so that each of its 4 chunks is larger than one slice, so every chunk
  is split into several slices. A smaller tensor would only ever produce one
  slice per chunk.
* **fp32:** ``float32`` is ``IEEE_FP32`` on both host and device, so the H2D/D2H
  conversion is a pure layout shuffle and the round trip must be bit-exact.

## Running this test

The emulated topology is a flex setting read when the device runtime starts, so
it must be set **on the pytest command line**:

```bash
FLEX_DEVICE=EMUL1p5 python -m pytest tests/test_interleave_thin_slice.py -v -s
```

``FLEX_DEVICE=EMUL1p5`` submits through the VF path and reports 4 memory domains
(``FLEX_EMUL_MAX_REGIONS`` defaults to 8). Under any other device flex reports a
single domain and the whole class **skips**, which is what happens in a default
``pytest tests/`` sweep.

Interleaved placement itself is requested in-process via an env gate
(``TORCH_SPYRE_EMULATE_INTERLEAVE`` / ``_spyre_debug_set_emulate_interleave``).

Both placements are exercised in **one** process, on purpose: only one process
may hold the Spyre device (see ``torch_spyre/__init__.py`` — "Spyre can't be
used by more than one process"), so a subprocess baseline is not possible. The
baseline therefore differs from the interleaved run only in placement.
"""

import contextlib
import hashlib
import time
import unittest

import torch
from torch_spyre import _C
from torch.testing._internal.common_utils import run_tests, TestCase

SEED = 20260815

RUN_COMMAND = "FLEX_DEVICE=EMUL1p5 python -m pytest tests/test_interleave_thin_slice.py"

# flex's default DMA slice size in VF mode (RuntimeConfig::DataTransferBufferSize).
DMA_SLICE_BYTES = 256 * 1024 * 1024

# fp32, 10241 x 32800 = ~1.25 GiB, i.e. ~336 MB per chunk at 4 domains, so every
# chunk takes two DMA slices. The last dim is 1025 sticks (32 fp32 each) and the
# row count is odd, so the stick count does not divide by 4: flex rounds each
# chunk up to the 128 B device alignment and total_size() exceeds the storage.
SHAPE = [10241, 32800]
DTYPE = torch.float32


@contextlib.contextmanager
def _interleaved_placement(enabled: bool):
    """Force interleaved (or Bind{0}) placement for allocations in this block."""
    previous = _C._spyre_debug_set_emulate_interleave(enabled)
    try:
        yield
    finally:
        _C._spyre_debug_set_emulate_interleave(previous)


def _chunks(dev: torch.Tensor) -> list:
    """Per-chunk layout of a device tensor's allocation, as plain dicts.

    Geometry comes from get_composite_address; the segment id (None if the
    chunk's region has no tensor segment) from a debug binding.
    Copied out at once so no handle keeps the allocation alive.
    """
    geometry = _C.get_composite_address(dev).chunks()
    segments = _C._spyre_debug_chunk_segment_ids(dev)
    return [
        {
            "domain_id": c.domain_id,
            "region_id": c.region_id,
            "offset": c.offset,
            "size": c.size,
            "segment_id": segment,
        }
        for c, segment in zip(geometry, segments, strict=True)
    ]


def _digest(tensor: torch.Tensor) -> str:
    # Hash the tensor's buffer in place; tobytes() would copy ~1.25 GiB.
    return hashlib.sha256(tensor.contiguous().numpy().data).hexdigest()


def _max_relative_error(back: torch.Tensor, host: torch.Tensor) -> float:
    """Largest relative deviation, in units of the host magnitude.

    Exact zeros round-trip to exact zeros, so clamping the denominator only
    guards the division; it cannot mask a real difference.
    """
    return float(((back - host).abs() / host.abs().clamp_min(2.0**-14)).max().item())


def _round_trip(interleaved: bool) -> dict:
    """Allocate + H2D + D2H the test tensor and report what happened.

    Only the allocation and upload run under the placement gate; the download
    happens with the gate restored, so the round trip is driven by the tensor's
    own address rather than by the flag still being set.
    """
    torch.manual_seed(SEED)
    host = torch.randn(SHAPE, dtype=DTYPE)

    start = time.perf_counter()
    with _interleaved_placement(interleaved):
        dev = host.to("spyre")
    h2d_seconds = time.perf_counter() - start

    chunks = _chunks(dev)
    start = time.perf_counter()
    back = dev.cpu()
    d2h_seconds = time.perf_counter() - start
    del dev

    exact = bool(torch.equal(back, host))
    return {
        "host_bytes": host.untyped_storage().nbytes(),
        "chunks": chunks,
        "chunk_bytes_total": sum(c["size"] for c in chunks),
        "exact": exact,
        # Only computed on a mismatch: it allocates several tensor-sized buffers.
        "max_rel_err": 0.0 if exact else _max_relative_error(back, host),
        "digest": _digest(back),
        "h2d_seconds": h2d_seconds,
        "d2h_seconds": d2h_seconds,
    }


class TestInterleaveThinSlice(TestCase):
    """Interleaved round trip under the emulated 4-domain topology (VF only)."""

    num_domains = 0
    interleaved: dict = {}
    baseline: dict = {}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.spyre.is_initialized():
            torch.spyre._lazy_init()

        cls.num_domains = _C._spyre_debug_num_memory_domains()
        if cls.num_domains < 2:
            raise unittest.SkipTest(
                "needs the emulated multi-domain topology of FLEX_DEVICE=EMUL1p5 "
                f"(VF); flex reports {cls.num_domains} memory domain(s). "
                f"Re-run as: {RUN_COMMAND}"
            )

        # The device work runs once and every test asserts on the result, so the
        # suite does not re-upload the tensor per assertion.
        cls.interleaved = _round_trip(interleaved=True)
        cls.baseline = _round_trip(interleaved=False)

    def test_allocation_is_interleaved_across_all_domains(self):
        """The tensor must be N chunks, one per domain, all device-addressable.

        Without this the numeric tests below would pass vacuously on a
        single-chunk allocation if the interleave gate ever stopped firing.
        """
        chunks = self.interleaved["chunks"]
        self.assertEqual(len(chunks), self.num_domains)
        # One chunk per domain, each in its own region.
        self.assertEqual(
            sorted(c["domain_id"] for c in chunks), list(range(self.num_domains))
        )
        self.assertEqual(len({c["region_id"] for c in chunks}), self.num_domains)
        # A region without a tensor segment has no device address and must
        # never be dispatched.
        for chunk in chunks:
            self.assertIsNotNone(chunk["segment_id"], msg=str(chunk))
        # Chunks are equal-sized and cover the whole tensor.
        self.assertEqual(len({c["size"] for c in chunks}), 1)
        self.assertGreaterEqual(
            self.interleaved["chunk_bytes_total"], self.interleaved["host_bytes"]
        )

    def test_baseline_allocation_is_single_chunk(self):
        """With the gate off, allocation must stay exactly Bind{0}."""
        self.assertEqual(len(self.baseline["chunks"]), 1)
        self.assertEqual(self.baseline["chunks"][0]["domain_id"], 0)

    def test_per_chunk_transfer_exceeds_dma_slice(self):
        """Each chunk must be larger than one DMA slice.

        That is what makes flex split every chunk into several slices; assert it
        so the coverage cannot silently rot if the shape is edited.
        """
        for chunk in self.interleaved["chunks"]:
            self.assertGreater(chunk["size"], DMA_SLICE_BYTES, msg=str(chunk))

    def test_round_trip_is_exact(self):
        """H2D → D2H must be bit-identical: fp32 is IEEE_FP32 on both sides."""
        self.assertTrue(
            self.interleaved["exact"],
            msg=(
                "round trip differs; max relative error "
                f"{self.interleaved['max_rel_err']:.3e}"
            ),
        )

    def test_interleaved_matches_bind_baseline(self):
        """Interleaved round-trip values must equal the Bind{0} baseline.

        Stronger than the round-trip check on its own: it rules out a
        scatter/gather bug that is self-consistent (same wrong permutation both
        ways) but still disagrees with the single-chunk path.
        """
        self.assertEqual(self.interleaved["digest"], self.baseline["digest"])

    def test_layout_report(self):
        """Not an assertion — print the observed layout and timings."""
        for name, case in (("interleaved", self.interleaved), ("bind0", self.baseline)):
            print(
                f"\n{name} shape={SHAPE} dtype={DTYPE}"
                f" host_bytes={case['host_bytes']}"
                f" chunk_bytes_total={case['chunk_bytes_total']}"
                f" exact={case['exact']}"
                f" h2d={case['h2d_seconds']:.2f}s d2h={case['d2h_seconds']:.2f}s"
            )
            for chunk in case["chunks"]:
                print(
                    f"  domain={chunk['domain_id']} region={chunk['region_id']}"
                    f" offset={chunk['offset']} size={chunk['size']}"
                    f" segment={chunk['segment_id']}"
                )


if __name__ == "__main__":
    run_tests()
