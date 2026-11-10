use std::hint::black_box;
use std::time::{Duration, Instant};

use mumbling::MAX_VALUES;
use mumbling::v1::{MumblingBitmapV1, MumblingBitmapV1Builder};

const CONTAINER_COUNT: usize = 1_024;
const QUERY_COUNT: usize = 4_096;
const SAMPLE_COUNT: usize = 15;
const CALIBRATION_TIME: Duration = Duration::from_millis(25);

fn main() {
    let empty = uniform_bitmap(CONTAINER_COUNT, 0);
    let sparse1 = uniform_bitmap(CONTAINER_COUNT, 1);
    let sparse31 = uniform_bitmap(CONTAINER_COUNT, 31);
    let dense32 = uniform_bitmap(CONTAINER_COUNT, 32);
    let dense33 = uniform_bitmap(CONTAINER_COUNT, 33);
    let dense256 = uniform_bitmap(CONTAINER_COUNT, 256);

    println!("Mumbling bitmap core benchmarks");
    println!("containers: {CONTAINER_COUNT}, samples: {SAMPLE_COUNT}");
    println!(
        "encoded bytes: empty={}, sparse-1={}, sparse-31={}, dense-32={}, dense-33={}, dense-256={}",
        empty.len(),
        sparse1.len(),
        sparse31.len(),
        dense32.len(),
        dense33.len(),
        dense256.len()
    );
    println!();

    benchmark_parse("parse/empty", &empty);
    benchmark_parse("parse/sparse-1", &sparse1);
    benchmark_parse("parse/sparse-31", &sparse31);
    benchmark_parse("parse/dense-32", &dense32);
    benchmark_parse("parse/dense-33", &dense33);
    benchmark_parse("parse/dense-256", &dense256);

    let empty = MumblingBitmapV1::parse(&empty).unwrap();
    let sparse1 = MumblingBitmapV1::parse(&sparse1).unwrap();
    let sparse31 = MumblingBitmapV1::parse(&sparse31).unwrap();
    let dense32 = MumblingBitmapV1::parse(&dense32).unwrap();
    let dense33 = MumblingBitmapV1::parse(&dense33).unwrap();
    let dense256 = MumblingBitmapV1::parse(&dense256).unwrap();

    let empty_misses = queries(0, false);
    let sparse1_hits = queries(1, true);
    let sparse1_misses = queries(1, false);
    let sparse_hits = queries(31, true);
    let sparse_misses = queries(31, false);
    let dense_hits = queries(32, true);
    let dense_misses = queries(32, false);
    let dense33_hits = queries(33, true);
    let dense33_misses = queries(33, false);
    let full_hits = queries(256, true);

    benchmark_contains("contains/empty/miss", &empty, &empty_misses);
    benchmark_contains("contains/sparse-1/hit", &sparse1, &sparse1_hits);
    benchmark_contains("contains/sparse-1/miss", &sparse1, &sparse1_misses);
    benchmark_contains("contains/sparse-31/hit", &sparse31, &sparse_hits);
    benchmark_contains("contains/sparse-31/miss", &sparse31, &sparse_misses);
    benchmark_contains("contains/dense-32/hit", &dense32, &dense_hits);
    benchmark_contains("contains/dense-32/miss", &dense32, &dense_misses);
    benchmark_contains("contains/dense-33/hit", &dense33, &dense33_hits);
    benchmark_contains("contains/dense-33/miss", &dense33, &dense33_misses);
    benchmark_contains("contains/dense-256/hit", &dense256, &full_hits);

    benchmark_iter("iter/sparse-1", &sparse1);
    benchmark_iter("iter/sparse-31", &sparse31);
    benchmark_iter("iter/dense-32", &dense32);
    benchmark_iter("iter/dense-33", &dense33);
    benchmark_iter("iter/dense-256", &dense256);

    let random_positions = construction_positions(100_000);
    let mut sorted_positions = random_positions.clone();
    sorted_positions.sort_unstable();
    sorted_positions.dedup();
    benchmark_construction("construct/random", &random_positions, false);
    benchmark_construction("construct/sorted", &sorted_positions, true);
    benchmark_range_construction();

    let random_builder = MumblingBitmapV1Builder::try_from_iter(random_positions).unwrap();
    let sparse_builder = uniform_builder(CONTAINER_COUNT, 31);
    let dense_builder = uniform_builder(CONTAINER_COUNT, 32);
    benchmark_serialization("serialize/random-100k", &random_builder);
    benchmark_serialization("serialize/sparse-31", &sparse_builder);
    benchmark_serialization("serialize/dense-32", &dense_builder);
    benchmark_reused_serialization("serialize-into/random-100k", &random_builder);
    benchmark_serialized_size("serialized-size/random-100k", &random_builder);
    benchmark_round_trip("roundtrip/random-100k", &random_builder);

    let left = MumblingBitmapV1Builder::from_sorted_iter((0..MAX_VALUES).step_by(3)).unwrap();
    let right = MumblingBitmapV1Builder::from_sorted_iter((0..MAX_VALUES).step_by(5)).unwrap();
    benchmark_set_operations(&left, &right);
}

fn benchmark_parse(name: &str, bytes: &[u8]) {
    measure(name, 1, "bitmap", || {
        let bitmap = MumblingBitmapV1::parse(black_box(bytes)).unwrap();
        black_box(bitmap.cardinality());
    });
}

fn benchmark_contains(name: &str, bitmap: &MumblingBitmapV1<'_>, queries: &[u32]) {
    measure(name, queries.len(), "query", || {
        let mut matches = 0_u32;
        for &position in black_box(queries) {
            matches += u32::from(black_box(bitmap).contains(position));
        }
        black_box(matches);
    });
}

fn benchmark_iter(name: &str, bitmap: &MumblingBitmapV1<'_>) {
    measure(name, bitmap.cardinality() as usize, "value", || {
        let mut checksum = 0_u64;
        for position in black_box(bitmap) {
            checksum = checksum.wrapping_add(u64::from(position));
        }
        black_box(checksum);
    });
}

fn benchmark_construction(name: &str, positions: &[u32], sorted: bool) {
    measure(name, positions.len(), "insert", || {
        let bitmap = if sorted {
            MumblingBitmapV1Builder::from_sorted_iter(black_box(positions).iter().copied()).unwrap()
        } else {
            MumblingBitmapV1Builder::try_from_iter(black_box(positions).iter().copied()).unwrap()
        };
        black_box(bitmap);
    });
}

fn benchmark_range_construction() {
    measure(
        "construct/range-full",
        MAX_VALUES as usize,
        "insert",
        || {
            let mut bitmap = MumblingBitmapV1Builder::new();
            black_box(bitmap.insert_range(0..MAX_VALUES).unwrap());
            black_box(bitmap);
        },
    );
}

fn benchmark_set_operations(left: &MumblingBitmapV1Builder, right: &MumblingBitmapV1Builder) {
    measure("set/union", 1, "bitmap", || {
        black_box(black_box(left) | black_box(right));
    });
    measure("set/intersection", 1, "bitmap", || {
        black_box(black_box(left) & black_box(right));
    });
    measure("set/difference", 1, "bitmap", || {
        black_box(black_box(left) - black_box(right));
    });
    measure("set/xor", 1, "bitmap", || {
        black_box(black_box(left) ^ black_box(right));
    });
    measure("set/intersection-len", 1, "bitmap", || {
        black_box(black_box(left).intersection_len(black_box(right)));
    });
    measure("set/is-disjoint-early", 1, "bitmap", || {
        black_box(black_box(left).is_disjoint(black_box(right)));
    });
    measure("set/is-subset-early", 1, "bitmap", || {
        black_box(black_box(left).is_subset(black_box(right)));
    });

    let evens = MumblingBitmapV1Builder::from_sorted_iter((0..MAX_VALUES).step_by(2)).unwrap();
    let odds = MumblingBitmapV1Builder::from_sorted_iter((1..MAX_VALUES).step_by(2)).unwrap();
    measure("set/is-disjoint-full", 1, "bitmap", || {
        black_box(black_box(&evens).is_disjoint(black_box(&odds)));
    });
    measure("set/is-subset-full", 1, "bitmap", || {
        black_box(black_box(&evens).is_subset(black_box(&evens)));
    });
}

fn benchmark_serialization(name: &str, bitmap: &MumblingBitmapV1Builder) {
    measure(name, 1, "bitmap", || {
        black_box(black_box(bitmap).serialize());
    });
}

fn benchmark_reused_serialization(name: &str, bitmap: &MumblingBitmapV1Builder) {
    let mut output = Vec::with_capacity(bitmap.serialized_size());
    measure(name, 1, "bitmap", || {
        output.clear();
        black_box(bitmap).serialize_into(&mut output);
        black_box(&output);
    });
}

fn benchmark_serialized_size(name: &str, bitmap: &MumblingBitmapV1Builder) {
    measure(name, 1, "bitmap", || {
        black_box(black_box(bitmap).serialized_size());
    });
}

fn benchmark_round_trip(name: &str, bitmap: &MumblingBitmapV1Builder) {
    measure(name, 1, "bitmap", || {
        let encoded = black_box(bitmap).serialize();
        let decoded = MumblingBitmapV1::parse(black_box(&encoded)).unwrap();
        black_box(decoded.cardinality());
    });
}

fn measure(mut name: &str, operations: usize, unit: &str, mut benchmark: impl FnMut()) {
    for _ in 0..3 {
        benchmark();
    }

    let mut batch_size = 1_usize;
    loop {
        let start = Instant::now();
        for _ in 0..batch_size {
            benchmark();
        }
        if start.elapsed() >= CALIBRATION_TIME {
            break;
        }
        batch_size = batch_size.saturating_mul(2);
        assert!(
            batch_size != usize::MAX,
            "benchmark is too fast to calibrate"
        );
    }

    let mut samples = Vec::with_capacity(SAMPLE_COUNT);
    for _ in 0..SAMPLE_COUNT {
        let start = Instant::now();
        for _ in 0..batch_size {
            benchmark();
        }
        let elapsed = start.elapsed().as_secs_f64();
        samples.push(elapsed * 1e9 / (batch_size * operations) as f64);
    }
    samples.sort_unstable_by(f64::total_cmp);

    let median = samples[SAMPLE_COUNT / 2];
    let low = samples[SAMPLE_COUNT / 10];
    let high = samples[SAMPLE_COUNT - 1 - SAMPLE_COUNT / 10];
    let throughput = 1e3 / median;

    if name.len() > 29 {
        name = &name[..29];
    }
    println!(
        "{name:<29} {median:>9.2} ns/{unit:<6} [{low:>8.2}, {high:>8.2}]  {throughput:>8.2} M {unit}/s"
    );
}

/// Builds a valid bitmap whose containers all have the same cardinality. The
/// descriptor array uses the compact PFOR representation for a constant chunk.
fn uniform_bitmap(container_count: usize, cardinality: usize) -> Vec<u8> {
    assert!(container_count <= 8_192);
    assert!(cardinality <= 256);

    let total_cardinality = container_count * cardinality;
    let descriptor = cardinality.min(32) as u8;
    let descriptor_chunks = container_count.div_ceil(256);
    let container_len = usize::from(descriptor);
    let mut bytes = Vec::with_capacity(6 + descriptor_chunks * 3 + container_count * container_len);

    bytes.push(1);
    bytes.extend_from_slice(&(total_cardinality as u32).to_le_bytes()[..3]);
    bytes.extend_from_slice(&(container_count as u16).to_le_bytes());

    for _ in 0..descriptor_chunks {
        bytes.extend_from_slice(&[0, 0, descriptor]);
    }

    for _ in 0..container_count {
        if cardinality < 32 {
            bytes.extend(0..cardinality as u8);
        } else {
            let full_bytes = cardinality / 8;
            let remaining_bits = cardinality % 8;
            bytes.extend(std::iter::repeat_n(0xff, full_bytes));
            if remaining_bits != 0 {
                bytes.push(0xff << (8 - remaining_bits));
            }
            bytes.extend(std::iter::repeat_n(
                0,
                32 - full_bytes - usize::from(remaining_bits != 0),
            ));
        }
    }

    bytes
}

/// Produces deterministic queries distributed across all containers.
fn queries(container_cardinality: usize, hits: bool) -> Vec<u32> {
    let mut state = 0x9e37_79b9_u32;
    let mut positions = Vec::with_capacity(QUERY_COUNT);

    for _ in 0..QUERY_COUNT {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let container = (state as usize % CONTAINER_COUNT) as u32;
        let local = if hits {
            state.rotate_left(13) as usize % container_cardinality
        } else {
            container_cardinality + state.rotate_left(13) as usize % (256 - container_cardinality)
        };
        positions.push((container << 8) | local as u32);
    }

    positions
}

fn construction_positions(count: usize) -> Vec<u32> {
    let mut state = 0x243f_6a88_u32;
    let mut positions = Vec::with_capacity(count);
    for _ in 0..count {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        positions.push(state % MAX_VALUES);
    }
    positions
}

fn uniform_builder(container_count: usize, cardinality: usize) -> MumblingBitmapV1Builder {
    MumblingBitmapV1Builder::from_sorted_iter(
        (0..container_count).flat_map(|container| {
            (0..cardinality).map(move |local| (container * 256 + local) as u32)
        }),
    )
    .unwrap()
}
