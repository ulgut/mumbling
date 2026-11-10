mod support;

use std::hint::black_box;

use mumbling::v1::{MumblingBitmapV1, MumblingBitmapV1Builder};
use roaring::RoaringBitmap;

use support::{mumbling, random_positions, roaring};

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

const REPETITIONS: u64 = 100;

fn main() {
    let positions = random_positions(100_000);
    let builder = mumbling(&positions);
    let encoded = builder.serialize();
    let roaring = roaring(&positions);
    let mut roaring_encoded = Vec::with_capacity(roaring.serialized_size());
    roaring.serialize_into(&mut roaring_encoded).unwrap();
    let prepared = builder.prepare_serialization();
    let mut output = Vec::with_capacity(builder.serialized_size());
    builder.serialize_into(&mut output);
    output.clear();
    let _profiler = dhat::Profiler::builder().testing().build();

    println!("allocation traffic per operation over {REPETITIONS} repetitions");
    measure("parse/mumbling", || {
        black_box(MumblingBitmapV1::parse(black_box(&encoded)).unwrap());
    });
    measure("parse/roaring", || {
        black_box(RoaringBitmap::deserialize_from(black_box(&roaring_encoded[..])).unwrap());
    });
    measure("construct/mumbling", || {
        black_box(
            MumblingBitmapV1Builder::try_from_iter(black_box(&positions).iter().copied()).unwrap(),
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
    measure("serialize/mumbling", || {
        black_box(black_box(&builder).serialize());
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
