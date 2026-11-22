//! Complete in-memory delete-vector pipelines, including allocation and teardown.
//! Fixture generation and correctness checks are outside the measured region.
mod helpers;

use std::hint::black_box;
use std::time::Duration;

use ::mumbling::{MAX_VALUES, MumblingBitmapV1};
use ::roaring::RoaringBitmap;
use criterion::{BenchmarkId, Throughput, criterion_group};
use helpers::BenchCriterion;

fn pipelines(c: &mut BenchCriterion) {
    for (name, positions) in [
        ("scattered-100k", helpers::random_positions(100_000)),
        ("contiguous-100k", (0..100_000).collect()),
    ] {
        let builder = helpers::mumbling::from_positions(&positions);
        let encoded = builder.serialize();
        let roaring = helpers::roaring::optimized(&positions);
        let mut roaring_encoded = Vec::with_capacity(roaring.serialized_size());
        roaring.serialize_into(&mut roaring_encoded).unwrap();
        let bitmap = MumblingBitmapV1::parse(&encoded).unwrap();
        assert!(bitmap.iter().eq(roaring.iter()));
        assert_eq!(helpers::roaring::from_unsorted_bulk(&positions), roaring);

        let mut write = c.benchmark_group("e2e/build-and-serialize");
        write.throughput(Throughput::Elements(positions.len() as u64));
        write.bench_function(BenchmarkId::new("mumbling", name), |b| {
            b.iter(|| helpers::mumbling::from_positions(black_box(&positions)).serialize());
        });
        // Both Roaring strategies include construction, optimize, exact size,
        // output allocation, serialization, and destruction of the builder.
        for (strategy, build) in [
            (
                "roaring-insert",
                helpers::roaring::from_positions as fn(&[u32]) -> RoaringBitmap,
            ),
            ("roaring-sort-bulk", helpers::roaring::from_unsorted_bulk),
        ] {
            write.bench_function(BenchmarkId::new(strategy, name), |b| {
                b.iter(|| {
                    let mut bitmap = build(black_box(&positions));
                    bitmap.optimize();
                    let mut output = Vec::with_capacity(bitmap.serialized_size());
                    bitmap.serialize_into(&mut output).unwrap();
                    output
                });
            });
        }
        write.finish();

        // Model a 16K-row scan batch. Spread batches across the universe so the
        // contiguous fixture contains both deleted and surviving rows.
        let rows: Vec<u32> = (0..64)
            .flat_map(|batch| (0..256).map(move |row| batch * (MAX_VALUES / 64) + row))
            .collect();
        let expected: Vec<_> = rows
            .iter()
            .copied()
            .filter(|&row| !roaring.contains(row))
            .collect();
        assert_eq!(
            rows.iter()
                .copied()
                .filter(|&row| !bitmap.contains(row))
                .collect::<Vec<_>>(),
            expected
        );
        let mut filter = c.benchmark_group("e2e/parse-and-filter");
        filter.throughput(Throughput::Elements(rows.len() as u64));
        filter.bench_function(BenchmarkId::new("mumbling", name), |b| {
            b.iter(|| {
                let bitmap = MumblingBitmapV1::parse(black_box(&encoded)).unwrap();
                black_box(&rows)
                    .iter()
                    .copied()
                    .filter(|&row| !bitmap.contains(row))
                    .collect::<Vec<_>>()
            });
        });
        filter.bench_function(BenchmarkId::new("roaring", name), |b| {
            b.iter(|| {
                let bitmap =
                    RoaringBitmap::deserialize_from(black_box(&roaring_encoded[..])).unwrap();
                black_box(&rows)
                    .iter()
                    .copied()
                    .filter(|&row| !bitmap.contains(row))
                    .collect::<Vec<_>>()
            });
        });
        filter.finish();

        let mut enumerate = c.benchmark_group("e2e/parse-and-enumerate");
        enumerate.throughput(Throughput::Elements(bitmap.cardinality() as u64));
        enumerate.bench_function(BenchmarkId::new("mumbling", name), |b| {
            b.iter(|| {
                MumblingBitmapV1::parse(black_box(&encoded))
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>()
            });
        });
        enumerate.bench_function(BenchmarkId::new("roaring", name), |b| {
            b.iter(|| {
                RoaringBitmap::deserialize_from(black_box(&roaring_encoded[..]))
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>()
            });
        });
        enumerate.finish();
    }
}

criterion_group! {
    name = benches;
    config = helpers::criterion()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(30);
    targets = pipelines
}
fn main() {
    benches();
    helpers::criterion().configure_from_args().final_summary();
}
