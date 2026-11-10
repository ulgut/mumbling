mod support;

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mumbling::MAX_VALUES;
use mumbling::v1::{MumblingBitmapV1, MumblingBitmapV1Builder};
use roaring::RoaringBitmap;

use support::{
    LARGE_CONTAINER_COUNT, MEDIUM_CONTAINER_COUNT, QUERY_COUNT, lookup_queries, mumbling,
    mumbling_sorted, random_positions, roaring, strided_lookup_queries, strided_positions,
    uniform_positions,
};

fn parse_boundaries(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse/container-boundaries");
    for container_count in [255, 256, 257, LARGE_CONTAINER_COUNT] {
        for cardinality in [1, 31, 32, 33, 256] {
            let positions = uniform_positions(container_count, cardinality);
            let encoded = mumbling_sorted(&positions).serialize();
            let roaring = roaring(&positions);
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

fn contains(c: &mut Criterion) {
    let mut group = c.benchmark_group("contains/semantic-comparison");
    group.throughput(Throughput::Elements(QUERY_COUNT as u64));

    for cardinality in [1, 31, 32, 33, 256] {
        let positions = uniform_positions(MEDIUM_CONTAINER_COUNT, cardinality);
        let encoded = mumbling_sorted(&positions).serialize();
        let bitmap = MumblingBitmapV1::parse(&encoded).unwrap();
        let roaring = roaring(&positions);

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
    let encoded = mumbling_sorted(&positions).serialize();
    let bitmap = MumblingBitmapV1::parse(&encoded).unwrap();
    let roaring = roaring(&positions);
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

fn iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("iterate/semantic-comparison");
    for cardinality in [1, 31, 32, 33, 256] {
        let positions = uniform_positions(MEDIUM_CONTAINER_COUNT, cardinality);
        let encoded = mumbling_sorted(&positions).serialize();
        let bitmap = MumblingBitmapV1::parse(&encoded).unwrap();
        let roaring = roaring(&positions);
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

fn construction(c: &mut Criterion) {
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

    group.throughput(Throughput::Elements(sorted.len() as u64));
    group.bench_function("mumbling/sorted", |b| {
        b.iter(|| {
            MumblingBitmapV1Builder::from_sorted_iter(black_box(&sorted).iter().copied()).unwrap()
        });
    });
    group.bench_function("roaring/sorted", |b| {
        b.iter(|| {
            black_box(&sorted)
                .iter()
                .copied()
                .collect::<RoaringBitmap>()
        });
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

fn serialization(c: &mut Criterion) {
    let positions = random_positions(100_000);
    let mumbling = mumbling(&positions);
    let roaring = roaring(&positions);
    let mumbling_size = mumbling.serialized_size();
    let prepared = mumbling.prepare_serialization();
    let roaring_size = roaring.serialized_size();
    println!("100k random encoded size: mumbling={mumbling_size} B, roaring={roaring_size} B");

    let mut group = c.benchmark_group("serialize/format-specific");
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
    group.finish();

    let mut group = c.benchmark_group("serialize/amortized-copies");
    for copies in [1, 2, 4, 8] {
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
                let mut output = Vec::with_capacity(roaring_size);
                for _ in 0..copies {
                    output.clear();
                    black_box(&roaring)
                        .serialize_into(black_box(&mut output))
                        .unwrap();
                    black_box(&output);
                }
            });
        });
    }
    group.finish();
}

fn set_operations(c: &mut Criterion) {
    let left_positions: Vec<_> = (0..MAX_VALUES).step_by(3).collect();
    let right_positions: Vec<_> = (0..MAX_VALUES).step_by(5).collect();
    let mumbling_left = mumbling_sorted(&left_positions);
    let mumbling_right = mumbling_sorted(&right_positions);
    let roaring_left = roaring(&left_positions);
    let roaring_right = roaring(&right_positions);

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
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(30)
        .noise_threshold(0.02);
    targets = parse_boundaries, contains, iteration, construction, serialization, set_operations
}
criterion_main!(benches);
