# mumbling

A safe, `no_std + alloc`, efficient Rust implementation of the [Apache Iceberg Mumbling
bitmap format][spec].

Requires Rust 1.89 or later. AArch64 uses NEON; x86-64 bulk validation detects
AVX-512BW, AVX2, or SSE2 at runtime, including OS register-state support.
Other architectures use scalar code. Tiny x86 lookups retain SSE2 to avoid
per-query dispatch. The timings below measure the AArch64 implementation.

## Benchmarks vs Roaring

Local Apple Silicon measurements, Rust 1.92.0 and Roaring 0.11.3. Times are
Criterion medians; lower is better. Both formats receive identical values.

### End-to-end latency

Each bitmap contains 100,000 deleted positions: either scattered throughout
`0..2_097_152` or the contiguous range `0..100_000`. These are in-memory
pipelines, including construction/parsing, output allocation, and teardown.

| Pipeline | Input | Mumbling | Roaring |
|:--|:--|--:|--:|
| Build + serialize | Scattered deletes | 274 µs | 1.05 ms |
| Build + serialize | Contiguous deletes | 156 µs | 431 µs |
| Parse + filter 16K rows | Scattered deletes | 105 µs | 251 µs |
| Parse + filter 16K rows | Contiguous deletes | 14.3 µs | 19.4 µs |
| Parse + enumerate | Scattered deletes | 171 µs | 282 µs |
| Parse + enumerate | Contiguous deletes | 125 µs | 309 µs |

Roaring writers use the faster measured strategy: direct insertion or sorting
and bulk construction, including preprocessing and `optimize()` in the timing.
Read-only Roaring fixtures are optimized before timing. No disk or network IO
is included. Filtering materializes surviving positions; enumeration
materializes all deleted positions.

### Operation latency

| Operation | Workload | Mumbling | Roaring |
|:--|:--|--:|--:|
| Parse | 8,192 values, one every 256 positions | 3.19 µs | 4.7 µs |
| Parse | 253,952 values, 31 per 256-position region | 7.45 µs | 7.85 µs |
| Lookup batch | 16,384 hits on 31,744 scattered positions | 23.7 µs | 25.1 µs |
| Build from unsorted input | 100,000 scattered positions; Roaring sort + bulk | 150 µs | 1.02 ms |
| Build from sorted input | Same 100,000 positions, already sorted | 181 µs | 277 µs |
| Serialize once | 100,000 scattered positions; fresh output | 124 µs | 40.6 µs |
| Copy cached encoding | Same bitmap; reused output, encoding excluded for both | 0.93 µs | 2.41 µs |

Lookup positions span `0..262_144`, with 31 values spaced eight positions apart
per 256-position region. Cached bytes are available to **both** formats; the
cached-copy row is not a comparison against Roaring re-encoding each time.

### Throughput

Derived from the corresponding scattered-input latency above:

| Pipeline | Mumbling | Roaring |
|:--|--:|--:|
| Build + serialize | 364M positions/s | 95M positions/s |
| Parse + filter | 157M rows/s | 65.3M rows/s |
| Parse + enumerate | 585M positions/s | 354M positions/s |

### Memory

For the same 100,000 scattered positions:

| Representation | Mumbling | Roaring |
|:--|--:|--:|
| Serialized bytes | 104,519 B | 200,264 B |
| Parsed bitmap, retained heap | 40,964 B | 201,280 B |
| Parsed bitmap + required backing bytes | 145,483 B | 201,280 B |

The combined reader figure counts a right-sized, unshared serialized input for
Mumbling; Roaring can release its input after parsing. These totals exclude
stack structs, allocator bookkeeping, and spare capacity in input vectors.
The memory audit also reports actual output capacity and builder allocation
traffic for both Roaring construction strategies.

See [the benchmark guide](benches/README.md) for commands, workload definitions,
CPU and memory modes, and the serialization investigation. Recorded estimates
are in [the result summary](benches/results.json). One-shot serialization
remains slower than Roaring; no encoder optimization is claimed in this pass.

## Read and validate

`MumblingBitmapV1` borrows the wire bytes. `parse` validates the complete
encoding once, including PFOR descriptors, payload lengths, container
cardinality, sparse ordering, dense occupancy, and trailing bytes.

```rust
use mumbling::v1::MumblingBitmapV1;

# fn read(serialized: &[u8]) -> Result<(), mumbling::Error> {
let bitmap = MumblingBitmapV1::parse(serialized)?;

assert_eq!(bitmap.as_bytes(), serialized);
println!("{} values in {} containers", bitmap.len(), bitmap.container_count());

if bitmap.contains(42) {
    // Position 42 is set.
}

for position in &bitmap {
    // Positions are yielded in ascending order.
    let _ = position;
}
# Ok(())
# }
```

The reader keeps the payload zero-copy and allocates descriptor and container
offset indexes during parsing. `contains`, iteration, and `as_bytes` do not
allocate.

Both reader and builder expose Roaring-style `len() -> u64` and `is_empty()`.
The existing `cardinality() -> u32` remains available for version-1 counts.
Construction is fallible because Mumbling has a smaller, fixed universe;
`parse` borrows bytes and serialization appends to a vector to support `no_std`.

## Construct and mutate

`MumblingBitmapV1Builder` accepts individual positions, half-open ranges,
arbitrary iterators, and strictly increasing iterators. Fallible operations
reject positions outside the version-1 universe.

```rust
use mumbling::v1::MumblingBitmapV1Builder;

# fn build() -> Result<(), Box<dyn std::error::Error>> {
let mut bitmap = MumblingBitmapV1Builder::try_from_iter([7, 1, 7, 42])?;
assert_eq!(bitmap.len(), 3);

assert!(bitmap.insert(100)?);
assert!(bitmap.remove(7)?);
bitmap.insert_range(200..300)?;
bitmap.remove_range(240..260)?;

let sorted = MumblingBitmapV1Builder::from_sorted_iter([1, 10, 100])?;
assert_eq!(sorted.len(), 3);
# Ok(())
# }
```

`from_sorted_iter` rejects duplicates and out-of-order input. `clear` retains
allocated storage for reuse. `contains`, `cardinality`, `is_empty`,
`is_disjoint`, `is_subset`, and `is_superset` provide queries on the builder.

## Set operations

Builders implement union (`|`/`|=`), intersection (`&`/`&=`), difference
(`-`/`-=`), and symmetric difference (`^`/`^=`). Cardinality-only variants
avoid materializing a result.

```rust
use mumbling::v1::MumblingBitmapV1Builder;

# fn sets() -> Result<(), mumbling::v1::PositionOutOfRange> {
let left = MumblingBitmapV1Builder::try_from([1, 2, 3])?;
let right = MumblingBitmapV1Builder::try_from([3, 4, 5])?;

let union = &left | &right;
let intersection = &left & &right;

assert_eq!(union.len(), 5);
assert_eq!(intersection.len(), 1);
assert_eq!(left.union_len(&right), 5);
assert_eq!(left.intersection_len(&right), 1);
assert_eq!(left.difference_len(&right), 2);
assert_eq!(left.symmetric_difference_len(&right), 4);
# Ok(())
# }
```

## Serialize

`serialize` returns a new version-1 wire image. `serialize_into` appends to an
existing `Vec<u8>`, and `serialized_size` computes the exact output size without
allocating.

```rust
use mumbling::v1::{MumblingBitmapV1, MumblingBitmapV1Builder};

# fn encode() -> Result<(), Box<dyn std::error::Error>> {
let bitmap = MumblingBitmapV1Builder::from_sorted_iter([1, 3, 5, 100])?;
let encoded = bitmap.serialize();
let decoded = MumblingBitmapV1::parse(&encoded)?;
assert!(decoded.iter().eq([1, 3, 5, 100]));

let mut output = Vec::with_capacity(bitmap.serialized_size());
bitmap.serialize_into(&mut output);
assert_eq!(output, encoded);
# Ok(())
# }
```

For repeated writes, `prepare_serialization` returns an owned snapshot. It can
outlive the builder and is unaffected by subsequent mutations. Its `as_bytes`
is zero-copy; `serialize` and `serialize_into` copy the cached bytes, and
`into_bytes` transfers ownership without copying. Preparation itself does the
same encoding work as one call to `serialize`.

```rust
use mumbling::v1::MumblingBitmapV1Builder;

# fn write(_: &[u8]) {}
# fn prepared() -> Result<(), mumbling::v1::PositionOutOfRange> {
let bitmap = MumblingBitmapV1Builder::try_from([1, 2, 3])?;
let prepared = bitmap.prepare_serialization();

write(prepared.as_bytes());
drop(bitmap); // The snapshot owns its bytes.
write(prepared.as_bytes());
let encoded = prepared.into_bytes();
write(&encoded);
# Ok(())
# }
```

[spec]: https://iceberg.apache.org/mumbling-spec
