# mumbling

A safe, `no_std + alloc`, efficient Rust implementation of the [Apache Iceberg Mumbling
bitmap format][spec].

## Benchmarks vs Roaring

Representative Criterion medians on Apple Silicon, using release builds and
identical value sets for both implementations:

| Operation | Workload | Mumbling latency | Mumbling throughput | Roaring latency | Roaring throughput | Result |
|:--|:--|--:|--:|--:|--:|--:|
| Parse | Sparse full-universe bitmap: 8,192 values, one set bit every 256 positions | 3.29 µs | 304K bitmaps/s | 4.86 µs | 206K bitmaps/s | **1.48× faster** |
| Parse | Near-threshold sparse bitmap: 253,952 values, 31 set bits in every 256-position region | 7.61 µs | 131K bitmaps/s | 12.10 µs | 82.6K bitmaps/s | **1.59× faster** |
| Lookup | 16,384 successful queries against 31,744 values in `0..262_144`; 31 values spaced 8 positions apart per 256-position region | 19.65 µs/batch | 834M queries/s | 25.20 µs/batch | 650M queries/s | **1.28× faster** |
| Iterate | 1,024 values spread one per 256-position region | 81 ns | 12.6B values/s | 1.82 µs | 563M values/s | **22.5× faster** |
| Iterate | 262,144 values filling every tested region | 317 µs | 827M values/s | 366 µs | 716M values/s | **1.15× faster** |
| Construct | 100,000 uniformly distributed positions supplied in random order | 152 µs | 658M values/s | 6.71 ms | 14.9M values/s | **44.1× faster** |
| Construct | Entire `0..2_097_152` universe as one range | 2.17 ns | O(1) lazy range | 652 ns | — | **300× faster** |
| Intersection | Two full-universe strided bitmaps, materializing the result | 9.38 µs | 224B positions/s | 11.74 µs | 179B positions/s | **1.25× faster** |
| Intersection cardinality | Same inputs, counting without materializing | 4.68 µs | 448B positions/s | 4.84 µs | 433B positions/s | **1.04× faster** |
| Serialize | 100,000 uniformly distributed values, one new output buffer | 146 µs | 685M values/s | 40.6 µs | 2.46B values/s | **3.59× slower** |
| Serialize prepared | Prepare once and write four output copies | 144 µs | 27.8K writes/s | 161 µs | 24.8K writes/s | **1.11× faster** |

Throughput is derived from the corresponding median latency. “Positions/s”
for set operations is the fixed 2,097,152-position universe processed per
operation; the full-range construction throughput is intentionally omitted
because Mumbling represents that range lazily in constant time.
Iteration benchmarks consume the values by summing them; their throughput
includes any compiler vectorization of that operation. Intersection inputs
contain every third and every fifth position, respectively.

### Memory

The same 100,000 uniformly distributed values were used for wire size and live
heap measurements:

| Representation | Mumbling | Roaring | Mumbling relative to Roaring |
|:--|--:|--:|--:|
| Serialized bytes | 104,519 B | 200,264 B | **1.92× smaller** |
| Parsed bitmap, retained heap | 40,964 B | 201,280 B | **4.91× less heap** |
| Parsed bitmap, including required backing bytes | 145,483 B | 201,280 B | **1.38× smaller** |
| Mutable builder, retained heap | 262,144 B | 263,424 B | **1,280 B smaller** |
| Construction allocation traffic | 262,144 B in 1 allocation | 526,432 B in 356 allocations | **2.01× fewer bytes** |

Retained-heap figures exclude the small stack-resident Rust structs and
allocator bookkeeping. The combined reader figure assumes Mumbling's borrowed
104,519-byte input is resident and not shared; a memory-mapped or shared input
would not add that entire amount to the reader's incremental heap footprint.

The full-range result uses the builder's lazy contiguous-prefix representation.
Prepared serialization amortizes one encoding pass across repeated writes;
one-shot serialization remains a known performance gap.

Run the comparison suite with `cargo bench --bench perf` and the memory audit
with `cargo bench --bench allocations`. Results are machine-dependent; the
x86-64 SSE2 paths are implemented but are not represented by the Apple Silicon
measurements above.

## Read and validate

`MumblingBitmapV1` borrows the wire bytes. `parse` validates the complete
encoding once, including PFOR descriptors, payload lengths, container
cardinality, sparse ordering, dense occupancy, and trailing bytes.

```rust
use mumbling::v1::MumblingBitmapV1;

# fn read(serialized: &[u8]) -> Result<(), mumbling::Error> {
let bitmap = MumblingBitmapV1::parse(serialized)?;

assert_eq!(bitmap.as_bytes(), serialized);
println!("{} values in {} containers", bitmap.cardinality(), bitmap.container_count());

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

## Construct and mutate

`MumblingBitmapV1Builder` accepts individual positions, half-open ranges,
arbitrary iterators, and strictly increasing iterators. Fallible operations
reject positions outside the version-1 universe.

```rust
use mumbling::v1::MumblingBitmapV1Builder;

# fn build() -> Result<(), Box<dyn std::error::Error>> {
let mut bitmap = MumblingBitmapV1Builder::try_from_iter([7, 1, 7, 42])?;
assert_eq!(bitmap.cardinality(), 3);

assert!(bitmap.insert(100)?);
assert!(bitmap.remove(7)?);
bitmap.insert_range(200..300)?;
bitmap.remove_range(240..260)?;

let sorted = MumblingBitmapV1Builder::from_sorted_iter([1, 10, 100])?;
assert_eq!(sorted.cardinality(), 3);
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

assert_eq!(union.cardinality(), 5);
assert_eq!(intersection.cardinality(), 1);
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

For repeated writes of an unchanged bitmap, `prepare_serialization` owns a
cached wire image while borrowing the builder. Its `as_bytes` is zero-copy;
`serialize` and `serialize_into` copy the cached bytes.

```rust
use mumbling::v1::MumblingBitmapV1Builder;

# fn write(_: &[u8]) {}
# fn prepared() -> Result<(), mumbling::v1::PositionOutOfRange> {
let bitmap = MumblingBitmapV1Builder::try_from([1, 2, 3])?;
let prepared = bitmap.prepare_serialization();

write(prepared.as_bytes());
write(prepared.as_bytes());
# Ok(())
# }
```

[spec]: https://iceberg.apache.org/mumbling-spec
