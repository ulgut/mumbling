# Benchmark guide

| File | Responsibility |
|:--|:--|
| `helpers.rs` | Shared deterministic positions, query batches, and measurement clocks. `helpers::mumbling` and `helpers::roaring` contain implementation-specific construction helpers. |
| `perf.rs` | Individual operations: parse, lookup, iteration, construction, serialization, and set operations. Optional isolated allocation audit. |
| `e2e.rs` | Complete in-memory delete-vector pipelines, including allocation and teardown. |

## Run

```sh
# Wall-clock latency and throughput (Criterion).
cargo bench --bench perf
cargo bench --bench e2e

# Allocation traffic and retained heap; no latency results from this run.
cargo bench --bench perf --features bench-memory

# Process CPU time, including user and kernel time (macOS/Linux).
cargo bench --bench perf --features bench-cpu -- 'serialize/format-specific'

# Smoke-check every workload without statistical timing.
cargo bench --bench perf -- --test
cargo bench --bench e2e -- --test
```

Wall-clock results go to `target/criterion`; CPU results go to
`target/criterion-cpu`. CPU throughput means operations per **CPU second**;
it is not wall-clock throughput, hardware cycles, or an instruction count.
Run CPU and memory modes separately. If both features are enabled, `perf`
runs only the memory audit. The tracking allocator is never installed by `e2e`.

For before/after comparisons, filter to the affected operation and use
Criterion's `--save-baseline NAME` and `--baseline NAME`. Avoid concurrent
builds/benchmarks, use the same compiler and machine, and rerun improvements
before accepting them. Defaults are 30 samples, 1 second warmup, and 3 seconds
measurement per benchmark. The committed result summary records the estimates
used in the top-level README; it is not a cross-machine performance guarantee.

## End-to-end workloads

Each pipeline runs against 100,000 scattered unique positions in the full
2,097,152-position universe and against the contiguous range `0..100_000`.

| Pipeline | Included work | Throughput denominator |
|:--|:--|:--|
| Build and serialize | Consume positions, build the bitmap, encode into a fresh vector, drop the builder and output | Input positions |
| Parse and filter | Validate the serialized bitmap, filter a batch into a fresh vector of surviving row positions, drop the reader and output | 16,384 queried rows |
| Parse and enumerate | Validate the serialized bitmap and materialize all deleted positions into a fresh vector, drop the reader and output | Deleted positions |

The filter batch comprises 64 spans of 256 consecutive rows, with span starts
spaced evenly across the universe. Neither all-hit nor all-miss behavior is
assumed. Both implementations receive identical queries and value sets.
Preflight assertions verify that they enumerate identical positions and produce
identical surviving rows. Setup and these checks are outside timing.

These are bitmap pipelines with resident inputs, not complete Iceberg scans:
no disk, network, Parquet decoding, or payload-row copying is simulated.

## Roaring comparison policy

- Sorted inputs use `RoaringBitmap::from_sorted_iter`.
- Arbitrary-input construction measures both direct insertion and copying,
  sorting, deduplicating, and bulk construction. The latter includes all that
  preprocessing in timing. The README uses the faster measured strategy.
- Read-only fixtures call `optimize()` before timing to permit run containers.
  End-to-end writers include optimization in their measured work.
- Repeated serialization permits **both** formats to cache their serialized
  bytes. Amortized-copy benchmarks include the initial encoding for both.
  Cached-copy microbenchmarks exclude it for both.
- Parsing validates both formats. Mumbling retains borrowed wire bytes;
  Roaring allocates its own decoded representation. Memory comparisons must
  include Mumbling's required input backing, not just its reader index.
- Latency includes destruction for both implementations. Prepared-copy timing
  compares equivalent copies, not a borrowed slice against encoding work.

## Serialization investigation

For the scattered 100K fixture, Mumbling's builder stores dense words but the
wire format predominantly requires sparse byte positions. Encoding enumerates
set bits across 32,768 words, computes 8,192 descriptors, and compresses 32
PFOR chunks. Roaring's 32 containers already hold arrays of 16-bit positions;
its serializer writes those positions and metadata without this conversion.

An isolated release-mode stage probe found sparse payload extraction was the
largest stage (roughly 84 µs), followed by PFOR encoding (27 µs) and descriptor
calculation (15 µs). These are standalone diagnostic timings, not additive
Criterion estimates: changed code layout and inlining affect them. Supplying
precomputed descriptors did not improve the payload probe (roughly 98 µs).
The temporary probe was removed; `serialize/layout` retains reproducible public
API comparisons of uniform sparse, dense, and full-universe inputs.

`serialize/format-specific` compares allocating, buffer-reuse, size, preparation,
and cached-copy operations. Reusing the output buffer did not remove the
one-shot gap. The proposed next target is sparse extraction, with randomized
and uniform layouts checked together. No wire-format change or speculative
descriptor cache was introduced in this cleanup.

## Architecture and profiling status

ARM NEON and x86 SSE2 tests ran locally. AVX2 and AVX-512 kernels compile and
have tests that execute when the host supports them; this Mac's x86 execution
environment exposes neither extension. Native AVX2/AVX-512 correctness and
performance runs remain pending. Wider vectors are not assumed faster.

For Instruments or another native profiler, use the fixed-work entry point:

```sh
cargo bench --bench perf -- --profile mumbling 100000
cargo bench --bench perf -- --profile roaring 100000
```

Profile the resulting `target/release/deps/perf-*` executable directly to avoid
including Cargo in the trace. This mode encodes the same 100K-position fixture
into a reused output buffer. It includes setup and 100 warmup iterations before
the separately named `profile_loop`; restrict analysis to that loop.

Hardware-counter profiling is unfinished: the local Linux VM exposes no CPU
PMU; Instruments' `xctrace` was terminated with exit 137 and samply returned
`Unknown(1100)`. No branch-miss or cache-miss conclusion is claimed from the
latency measurements alone.
