//! Operation-level latency/throughput, or an isolated allocation audit.
mod helpers;

use std::hint::black_box;
use std::time::Duration;

use ::mumbling::MAX_VALUES;
use ::mumbling::v1::{MumblingBitmapV1, MumblingBitmapV1Builder};
use ::roaring::RoaringBitmap;
use criterion::{BenchmarkId, Throughput, criterion_group};

use helpers::{
    LARGE_CONTAINER_COUNT, MEDIUM_CONTAINER_COUNT, QUERY_COUNT, lookup_queries, mumbling,
    random_positions, roaring, strided_lookup_queries, strided_positions, uniform_positions,
};

fn parse_boundaries(c: &mut helpers::BenchCriterion) {
    let mut group = c.benchmark_group("parse/container-boundaries");
    for container_count in [255, 256, 257, LARGE_CONTAINER_COUNT] {
        for cardinality in [1, 31, 32, 33, 256] {
            let positions = uniform_positions(container_count, cardinality);
            let encoded = mumbling::from_sorted_positions(&positions).serialize();
            let roaring = roaring::optimized(&positions);
            let mut roaring_encoded = Vec::with_capacity(roaring.serialized_size());
            roaring.serialize_into(&mut roaring_encoded).unwrap();

            group.throughput(Throughput::Bytes(encoded.len() as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("mumbling/c{container_count}"), cardinality),
                &encoded,
                |b, encoded| {
                    b.iter(|| {
                        let bitmap = MumblingBitmapV1::parse(black_box(encoded)).unwrap();
                        black_box(bitmap.cardinality())
                    });
                },
            );
            group.throughput(Throughput::Bytes(roaring_encoded.len() as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("roaring/c{container_count}"), cardinality),
                &roaring_encoded,
                |b, encoded| {
                    b.iter(|| {
                        let bitmap =
                            RoaringBitmap::deserialize_from(black_box(&encoded[..])).unwrap();
                        black_box(bitmap.len())
                    });
                },
            );
        }
    }
    group.finish();
}

fn contains(c: &mut helpers::BenchCriterion) {
    let mut group = c.benchmark_group("contains/semantic-comparison");
    group.throughput(Throughput::Elements(QUERY_COUNT as u64));

    for cardinality in [1, 31, 32, 33, 256] {
        let positions = uniform_positions(MEDIUM_CONTAINER_COUNT, cardinality);
        let encoded = mumbling::from_sorted_positions(&positions).serialize();
        let bitmap = MumblingBitmapV1::parse(&encoded).unwrap();
        let roaring = roaring::optimized(&positions);

        for (kind, queries) in query_sets(cardinality) {
            group.bench_with_input(
                BenchmarkId::new("mumbling", format!("cardinality-{cardinality}/{kind}")),
                &queries,
                |b, queries| b.iter(|| lookup_mumbling(black_box(&bitmap), black_box(queries))),
            );
            group.bench_with_input(
                BenchmarkId::new("roaring", format!("cardinality-{cardinality}/{kind}")),
                &queries,
                |b, queries| b.iter(|| lookup_roaring(black_box(&roaring), black_box(queries))),
            );
        }
    }

    let positions = strided_positions(MEDIUM_CONTAINER_COUNT, 31, 8);
    let encoded = mumbling::from_sorted_positions(&positions).serialize();
    let bitmap = MumblingBitmapV1::parse(&encoded).unwrap();
    let roaring = roaring::optimized(&positions);
    for (kind, queries) in [
        ("hit", strided_lookup_queries(31, 8, true)),
        ("miss-in-range", strided_lookup_queries(31, 8, false)),
    ] {
        group.bench_with_input(
            BenchmarkId::new("mumbling", format!("cardinality-31-scattered/{kind}")),
            &queries,
            |b, queries| b.iter(|| lookup_mumbling(black_box(&bitmap), black_box(queries))),
        );
        group.bench_with_input(
            BenchmarkId::new("roaring", format!("cardinality-31-scattered/{kind}")),
            &queries,
            |b, queries| b.iter(|| lookup_roaring(black_box(&roaring), black_box(queries))),
        );
    }
    group.finish();
}

fn iteration(c: &mut helpers::BenchCriterion) {
    let mut group = c.benchmark_group("iterate/semantic-comparison");
    for cardinality in [1, 31, 32, 33, 256] {
        let positions = uniform_positions(MEDIUM_CONTAINER_COUNT, cardinality);
        let encoded = mumbling::from_sorted_positions(&positions).serialize();
        let bitmap = MumblingBitmapV1::parse(&encoded).unwrap();
        let roaring = roaring::optimized(&positions);
        group.throughput(Throughput::Elements(positions.len() as u64));

        group.bench_function(BenchmarkId::new("mumbling", cardinality), |b| {
            b.iter(|| {
                black_box(&bitmap)
                    .into_iter()
                    .fold(0_u64, |sum, value| sum.wrapping_add(u64::from(value)))
            });
        });
        group.bench_function(BenchmarkId::new("roaring", cardinality), |b| {
            b.iter(|| {
                black_box(&roaring)
                    .iter()
                    .fold(0_u64, |sum, value| sum.wrapping_add(u64::from(value)))
            });
        });
    }
    group.finish();
}

fn construction(c: &mut helpers::BenchCriterion) {
    let random = random_positions(100_000);
    let mut sorted = random.clone();
    sorted.sort_unstable();
    sorted.dedup();

    let mut group = c.benchmark_group("construct/semantic-comparison");
    group.throughput(Throughput::Elements(random.len() as u64));
    group.bench_function("mumbling/random-insert", |b| {
        b.iter(|| {
            MumblingBitmapV1Builder::try_from_iter(black_box(&random).iter().copied()).unwrap()
        });
    });
    group.bench_function("roaring/random-insert", |b| {
        b.iter(|| {
            black_box(&random)
                .iter()
                .copied()
                .collect::<RoaringBitmap>()
        });
    });
    group.bench_function("roaring/random-sort-bulk", |b| {
        b.iter(|| helpers::roaring::from_unsorted_bulk(black_box(&random)));
    });

    group.throughput(Throughput::Elements(sorted.len() as u64));
    group.bench_function("mumbling/sorted", |b| {
        b.iter(|| {
            MumblingBitmapV1Builder::from_sorted_iter(black_box(&sorted).iter().copied()).unwrap()
        });
    });
    group.bench_function("roaring/sorted", |b| {
        b.iter(|| RoaringBitmap::from_sorted_iter(black_box(&sorted).iter().copied()).unwrap());
    });

    group.throughput(Throughput::Elements(MAX_VALUES as u64));
    group.bench_function("mumbling/full-range", |b| {
        b.iter(|| {
            let mut bitmap = MumblingBitmapV1Builder::new();
            bitmap.insert_range(0..MAX_VALUES).unwrap();
            bitmap
        });
    });
    group.bench_function("roaring/full-range", |b| {
        b.iter(|| {
            let mut bitmap = RoaringBitmap::new();
            bitmap.insert_range(0..MAX_VALUES);
            bitmap
        });
    });
    group.finish();
}

fn serialization(c: &mut helpers::BenchCriterion) {
    let positions = random_positions(100_000);
    let mumbling = mumbling::from_positions(&positions);
    let roaring = roaring::optimized(&positions);
    let mumbling_size = mumbling.serialized_size();
    let prepared = mumbling.prepare_serialization();
    let roaring_size = roaring.serialized_size();
    let mut roaring_prepared = Vec::with_capacity(roaring_size);
    roaring.serialize_into(&mut roaring_prepared).unwrap();
    println!("100k random encoded size: mumbling={mumbling_size} B, roaring={roaring_size} B");

    let mut group = c.benchmark_group("serialize/format-specific");
    group.throughput(Throughput::Elements(1));
    group.bench_function("mumbling/allocate", |b| {
        b.iter(|| black_box(&mumbling).serialize());
    });
    let mut mumbling_output = Vec::with_capacity(mumbling_size);
    group.bench_function("mumbling/reuse", |b| {
        b.iter(|| {
            mumbling_output.clear();
            black_box(&mumbling).serialize_into(black_box(&mut mumbling_output));
            black_box(&mumbling_output);
        });
    });
    group.bench_function("mumbling/size", |b| {
        b.iter(|| black_box(&mumbling).serialized_size());
    });
    group.bench_function("mumbling/prepare", |b| {
        b.iter(|| black_box(&mumbling).prepare_serialization());
    });
    group.bench_function("mumbling/prepared-allocate", |b| {
        b.iter(|| black_box(&prepared).serialize());
    });
    let mut prepared_output = Vec::with_capacity(prepared.serialized_size());
    group.bench_function("mumbling/prepared-reuse", |b| {
        b.iter(|| {
            prepared_output.clear();
            black_box(&prepared).serialize_into(black_box(&mut prepared_output));
            black_box(&prepared_output);
        });
    });
    group.bench_function("mumbling/prepared-size", |b| {
        b.iter(|| black_box(&prepared).serialized_size());
    });
    group.bench_function("mumbling/prepared-as-bytes", |b| {
        b.iter(|| black_box(black_box(&prepared).as_bytes()));
    });

    group.bench_function("roaring/allocate", |b| {
        b.iter(|| {
            let mut output = Vec::with_capacity(roaring_size);
            black_box(&roaring).serialize_into(&mut output).unwrap();
            output
        });
    });
    let mut roaring_output = Vec::with_capacity(roaring_size);
    group.bench_function("roaring/reuse", |b| {
        b.iter(|| {
            roaring_output.clear();
            black_box(&roaring)
                .serialize_into(black_box(&mut roaring_output))
                .unwrap();
            black_box(&roaring_output);
        });
    });
    group.bench_function("roaring/size", |b| {
        b.iter(|| black_box(&roaring).serialized_size());
    });
    group.bench_function("roaring/prepared-allocate", |b| {
        b.iter(|| black_box(&roaring_prepared).clone());
    });
    group.bench_function("roaring/prepared-reuse", |b| {
        b.iter(|| {
            roaring_output.clear();
            roaring_output.extend_from_slice(black_box(&roaring_prepared));
            black_box(&roaring_output);
        });
    });
    group.finish();

    let mut group = c.benchmark_group("serialize/amortized-copies");
    for copies in [1, 2, 4, 8] {
        group.throughput(Throughput::Elements(copies as u64));
        group.bench_function(BenchmarkId::new("mumbling", copies), |b| {
            b.iter(|| {
                let prepared = black_box(&mumbling).prepare_serialization();
                let mut output = Vec::with_capacity(prepared.serialized_size());
                for _ in 0..copies {
                    output.clear();
                    prepared.serialize_into(black_box(&mut output));
                    black_box(&output);
                }
            });
        });
        group.bench_function(BenchmarkId::new("roaring", copies), |b| {
            b.iter(|| {
                let mut prepared = Vec::with_capacity(roaring_size);
                black_box(&roaring).serialize_into(&mut prepared).unwrap();
                let mut output = Vec::with_capacity(roaring_size);
                for _ in 0..copies {
                    output.clear();
                    output.extend_from_slice(black_box(&prepared));
                    black_box(&output);
                }
            });
        });
    }
    group.finish();
}

fn set_operations(c: &mut helpers::BenchCriterion) {
    let left_positions: Vec<_> = (0..MAX_VALUES).step_by(3).collect();
    let right_positions: Vec<_> = (0..MAX_VALUES).step_by(5).collect();
    let mumbling_left = mumbling::from_sorted_positions(&left_positions);
    let mumbling_right = mumbling::from_sorted_positions(&right_positions);
    let roaring_left = roaring::optimized(&left_positions);
    let roaring_right = roaring::optimized(&right_positions);

    let mut group = c.benchmark_group("set/semantic-comparison");
    group.throughput(Throughput::Elements(MAX_VALUES as u64));
    group.bench_function("mumbling/union", |b| {
        b.iter(|| black_box(&mumbling_left) | black_box(&mumbling_right));
    });
    group.bench_function("roaring/union", |b| {
        b.iter(|| black_box(&roaring_left) | black_box(&roaring_right));
    });
    group.bench_function("mumbling/intersection", |b| {
        b.iter(|| black_box(&mumbling_left) & black_box(&mumbling_right));
    });
    group.bench_function("roaring/intersection", |b| {
        b.iter(|| black_box(&roaring_left) & black_box(&roaring_right));
    });
    group.bench_function("mumbling/difference", |b| {
        b.iter(|| black_box(&mumbling_left) - black_box(&mumbling_right));
    });
    group.bench_function("roaring/difference", |b| {
        b.iter(|| black_box(&roaring_left) - black_box(&roaring_right));
    });
    group.bench_function("mumbling/xor", |b| {
        b.iter(|| black_box(&mumbling_left) ^ black_box(&mumbling_right));
    });
    group.bench_function("roaring/xor", |b| {
        b.iter(|| black_box(&roaring_left) ^ black_box(&roaring_right));
    });
    group.bench_function("mumbling/intersection-len", |b| {
        b.iter(|| black_box(&mumbling_left).intersection_len(black_box(&mumbling_right)));
    });
    group.bench_function("roaring/intersection-len", |b| {
        b.iter(|| black_box(&roaring_left).intersection_len(black_box(&roaring_right)));
    });
    group.bench_function("mumbling/union-len", |b| {
        b.iter(|| black_box(&mumbling_left).union_len(black_box(&mumbling_right)));
    });
    group.bench_function("roaring/union-len", |b| {
        b.iter(|| black_box(&roaring_left).union_len(black_box(&roaring_right)));
    });
    group.bench_function("mumbling/difference-len", |b| {
        b.iter(|| black_box(&mumbling_left).difference_len(black_box(&mumbling_right)));
    });
    group.bench_function("roaring/difference-len", |b| {
        b.iter(|| black_box(&roaring_left).difference_len(black_box(&roaring_right)));
    });
    group.bench_function("mumbling/xor-len", |b| {
        b.iter(|| black_box(&mumbling_left).symmetric_difference_len(black_box(&mumbling_right)));
    });
    group.bench_function("roaring/xor-len", |b| {
        b.iter(|| black_box(&roaring_left).symmetric_difference_len(black_box(&roaring_right)));
    });
    group.finish();
}

/// Distinguishes descriptor compression, sparse extraction, and dense copying
/// without exposing implementation internals in the public library API.
fn serialization_layouts(c: &mut helpers::BenchCriterion) {
    let mut group = c.benchmark_group("serialize/layout");
    group.throughput(Throughput::Elements(1));
    for (name, positions) in [
        (
            "uniform-sparse-12",
            uniform_positions(LARGE_CONTAINER_COUNT, 12),
        ),
        (
            "uniform-sparse-31",
            uniform_positions(LARGE_CONTAINER_COUNT, 31),
        ),
        (
            "uniform-dense-32",
            uniform_positions(LARGE_CONTAINER_COUNT, 32),
        ),
        ("full-universe", (0..MAX_VALUES).collect()),
    ] {
        let bitmap = mumbling::from_sorted_positions(&positions);
        let roaring = roaring::optimized(&positions);
        let encoded = bitmap.serialize();
        assert!(
            MumblingBitmapV1::parse(&encoded)
                .unwrap()
                .iter()
                .eq(roaring.iter())
        );
        group.bench_function(BenchmarkId::new("mumbling", name), |b| {
            b.iter(|| black_box(&bitmap).serialize());
        });
        group.bench_function(BenchmarkId::new("roaring", name), |b| {
            b.iter(|| {
                let mut output = Vec::with_capacity(roaring.serialized_size());
                black_box(&roaring).serialize_into(&mut output).unwrap();
                output
            });
        });
    }
    group.finish();
}

fn query_sets(cardinality: usize) -> Vec<(&'static str, Vec<u32>)> {
    let mut sets = Vec::with_capacity(2);
    sets.push(("hit", lookup_queries(cardinality, true)));
    if cardinality != 256 {
        sets.push(("miss", lookup_queries(cardinality, false)));
    }
    sets
}

fn lookup_mumbling(bitmap: &MumblingBitmapV1<'_>, queries: &[u32]) -> u32 {
    queries
        .iter()
        .map(|&position| u32::from(bitmap.contains(position)))
        .sum()
}

fn lookup_roaring(bitmap: &RoaringBitmap, queries: &[u32]) -> u32 {
    queries
        .iter()
        .map(|&position| u32::from(bitmap.contains(position)))
        .sum()
}

criterion_group! {
    name = benches;
    config = helpers::criterion()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(30)
        .noise_threshold(0.02);
    targets = parse_boundaries, contains, iteration, construction, serialization, serialization_layouts, set_operations
}
fn main() {
    if std::env::args().nth(1).as_deref() == Some("--profile") {
        if cfg!(feature = "bench-memory") {
            panic!("profile without the tracking allocator");
        }
        profile_serialization();
        return;
    }
    if cfg!(feature = "bench-memory") {
        #[cfg(feature = "bench-memory")]
        memory::run();
    } else {
        benches();
        helpers::criterion().configure_from_args().final_summary();
    }
}

/// Fixed work for Instruments, samply, or perf: fixture preparation and a short
/// warmup precede the named loop. No Criterion calibration/reporting runs here.
fn profile_serialization() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        args.len() == 4,
        "usage: --profile mumbling|roaring ITERATIONS"
    );
    let iterations: u64 = args[3].parse().expect("positive iteration count");
    assert!(iterations > 0);
    let positions = helpers::random_positions(100_000);
    match args[2].as_str() {
        "mumbling" => {
            let bitmap = helpers::mumbling::from_positions(&positions);
            let mut output = Vec::with_capacity(bitmap.serialized_size());
            let mut encode = || {
                output.clear();
                black_box(&bitmap).serialize_into(black_box(&mut output));
                black_box(&output);
            };
            for _ in 0..100 {
                encode();
            }
            profile_loop(iterations, &mut encode);
        }
        "roaring" => {
            let bitmap = helpers::roaring::optimized(&positions);
            let mut output = Vec::with_capacity(bitmap.serialized_size());
            let mut encode = || {
                output.clear();
                black_box(&bitmap)
                    .serialize_into(black_box(&mut output))
                    .unwrap();
                black_box(&output);
            };
            for _ in 0..100 {
                encode();
            }
            profile_loop(iterations, &mut encode);
        }
        _ => panic!("implementation must be mumbling or roaring"),
    }
}

#[inline(never)]
fn profile_loop(iterations: u64, encode: &mut impl FnMut()) {
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        encode();
    }
    eprintln!("{iterations} serializations in {:?}", start.elapsed());
}

#[cfg(feature = "bench-memory")]
mod memory {
    //! Allocation audit used only by `perf --features bench-memory`.

    use std::hint::black_box;

    use ::mumbling::v1::{MumblingBitmapV1, MumblingBitmapV1Builder};
    use ::roaring::RoaringBitmap;

    use crate::helpers::{mumbling, random_positions, roaring};

    #[global_allocator]
    static ALLOC: dhat::Alloc = dhat::Alloc;

    const REPETITIONS: u64 = 100;

    pub(super) fn run() {
        let positions = random_positions(100_000);
        let builder = mumbling::from_positions(&positions);
        let encoded = builder.serialize();
        let roaring = roaring::optimized(&positions);
        let mut roaring_encoded = Vec::with_capacity(roaring.serialized_size());
        roaring.serialize_into(&mut roaring_encoded).unwrap();
        let prepared = builder.prepare_serialization();
        let mut output = Vec::with_capacity(builder.serialized_size());
        builder.serialize_into(&mut output);
        output.clear();
        let _profiler = dhat::Profiler::builder().testing().build();

        println!(
            "wire bytes: mumbling={}, roaring={}",
            encoded.len(),
            roaring_encoded.len()
        );
        println!(
            "output Vec capacity: mumbling={}, roaring={}",
            encoded.capacity(),
            roaring_encoded.capacity()
        );
        println!("allocation traffic per operation over {REPETITIONS} repetitions");
        measure("parse/mumbling", || {
            black_box(MumblingBitmapV1::parse(black_box(&encoded)).unwrap());
        });
        measure("parse/roaring", || {
            black_box(RoaringBitmap::deserialize_from(black_box(&roaring_encoded[..])).unwrap());
        });
        measure("construct/mumbling", || {
            black_box(
                MumblingBitmapV1Builder::try_from_iter(black_box(&positions).iter().copied())
                    .unwrap(),
            );
        });
        measure("construct/roaring", || {
            black_box(
                black_box(&positions)
                    .iter()
                    .copied()
                    .collect::<RoaringBitmap>(),
            );
        });
        measure("construct/roaring-bulk", || {
            black_box(roaring::from_unsorted_bulk(black_box(&positions)));
        });
        measure("serialize/mumbling", || {
            black_box(black_box(&builder).serialize());
        });
        measure("serialize/roaring", || {
            let mut output = Vec::with_capacity(roaring.serialized_size());
            black_box(&roaring).serialize_into(&mut output).unwrap();
            black_box(output);
        });
        measure("serialize/reuse", || {
            output.clear();
            black_box(&builder).serialize_into(black_box(&mut output));
            black_box(&output);
        });
        measure("serialize/prepared", || {
            output.clear();
            prepared.serialize_into(black_box(&mut output));
            black_box(&output);
        });

        println!("retained heap for one live bitmap");
        measure_retained("reader/mumbling", || {
            MumblingBitmapV1::parse(black_box(&encoded)).unwrap()
        });
        measure_retained("reader/roaring", || {
            RoaringBitmap::deserialize_from(black_box(&roaring_encoded[..])).unwrap()
        });
        measure_retained("builder/mumbling", || {
            MumblingBitmapV1Builder::try_from_iter(black_box(&positions).iter().copied()).unwrap()
        });
        measure_retained("builder/roaring", || {
            black_box(&positions)
                .iter()
                .copied()
                .collect::<RoaringBitmap>()
        });
        measure_retained("builder/roaring-bulk", || {
            roaring::from_unsorted_bulk(black_box(&positions))
        });
    }

    fn measure_retained<T>(name: &str, operation: impl FnOnce() -> T) {
        let before = dhat::HeapStats::get();
        let value = operation();
        let after = dhat::HeapStats::get();
        println!(
            "{name:<24} {:>8} allocations  {:>12} bytes",
            after.curr_blocks - before.curr_blocks,
            after.curr_bytes - before.curr_bytes,
        );
        black_box(&value);
    }

    fn measure(name: &str, mut operation: impl FnMut()) {
        let before = dhat::HeapStats::get();
        for _ in 0..REPETITIONS {
            operation();
        }
        let after = dhat::HeapStats::get();
        let blocks = after.total_blocks - before.total_blocks;
        let bytes = after.total_bytes - before.total_bytes;
        println!(
            "{name:<24} {:>8.2} allocations/op  {:>12.2} bytes/op",
            blocks as f64 / REPETITIONS as f64,
            bytes as f64 / REPETITIONS as f64
        );
    }
}
