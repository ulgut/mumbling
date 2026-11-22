//! Deterministic fixtures shared by microbenchmarks and complete pipelines.
#![allow(dead_code)]

use ::mumbling::MAX_VALUES;

#[cfg(not(feature = "bench-cpu"))]
pub type Timer = criterion::measurement::WallTime;
#[cfg(feature = "bench-cpu")]
pub use cpu::Timer;
pub type BenchCriterion = criterion::Criterion<Timer>;

#[cfg(all(feature = "bench-cpu", not(unix)))]
compile_error!("bench-cpu requires a Unix process CPU clock");

pub fn criterion() -> BenchCriterion {
    #[cfg(feature = "bench-cpu")]
    let timer = cpu::Timer;
    #[cfg(not(feature = "bench-cpu"))]
    let timer = criterion::measurement::WallTime;
    let criterion = criterion::Criterion::default().with_measurement(timer);
    // Keep results from different clocks in separate directories.
    #[cfg(feature = "bench-cpu")]
    let criterion = criterion.output_directory(std::path::Path::new("target/criterion-cpu"));
    criterion
}

#[cfg(feature = "bench-cpu")]
mod cpu {
    use criterion::measurement::{Measurement, ValueFormatter, WallTime};

    /// Process CPU nanoseconds (user + kernel), excluding time descheduled.
    #[derive(Default)]
    pub struct Timer;

    fn now() -> u64 {
        let mut time = core::mem::MaybeUninit::<libc::timespec>::uninit();
        // SAFETY: clock_gettime writes one timespec to a valid pointer. Only
        // read the result after success; no uninitialized padding is inspected.
        let status =
            unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, time.as_mut_ptr()) };
        assert_eq!(status, 0, "process CPU clock unavailable");
        // SAFETY: successful clock_gettime initialized both fields.
        let time = unsafe { time.assume_init() };
        time.tv_sec as u64 * 1_000_000_000 + time.tv_nsec as u64
    }

    impl Measurement for Timer {
        type Intermediate = u64;
        type Value = u64;
        fn start(&self) -> u64 {
            now()
        }
        fn end(&self, start: u64) -> u64 {
            now() - start
        }
        fn add(&self, a: &u64, b: &u64) -> u64 {
            a + b
        }
        fn zero(&self) -> u64 {
            0
        }
        fn to_f64(&self, value: &u64) -> f64 {
            *value as f64
        }
        fn formatter(&self) -> &dyn ValueFormatter {
            WallTime.formatter()
        }
    }
}

pub const LARGE_CONTAINER_COUNT: usize = 8_192;
pub const MEDIUM_CONTAINER_COUNT: usize = 1_024;
pub const QUERY_COUNT: usize = 16_384;

pub fn uniform_positions(container_count: usize, cardinality: usize) -> Vec<u32> {
    assert!(container_count <= LARGE_CONTAINER_COUNT);
    assert!(cardinality <= 256);
    (0..container_count)
        .flat_map(|container| (0..cardinality).map(move |local| ((container << 8) | local) as u32))
        .collect()
}

pub fn strided_positions(container_count: usize, cardinality: usize, stride: usize) -> Vec<u32> {
    assert!(cardinality != 0);
    assert!((cardinality - 1) * stride < 256);
    (0..container_count)
        .flat_map(|container| {
            (0..cardinality).map(move |local| ((container << 8) | (local * stride)) as u32)
        })
        .collect()
}

pub fn random_positions(count: usize) -> Vec<u32> {
    let mut state = 0x243f_6a88_u32;
    let mut positions = Vec::with_capacity(count);
    for _ in 0..count {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        positions.push(state % MAX_VALUES);
    }
    positions
}

pub fn lookup_queries(container_cardinality: usize, hits: bool) -> Vec<u32> {
    assert!(container_cardinality <= 256);
    assert!(container_cardinality != 0 || !hits);
    assert!(container_cardinality != 256 || hits);
    let mut state = 0x9e37_79b9_u32;
    let mut positions = Vec::with_capacity(QUERY_COUNT);

    for _ in 0..QUERY_COUNT {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let container = state as usize % MEDIUM_CONTAINER_COUNT;
        let local = if hits {
            state.rotate_left(13) as usize % container_cardinality
        } else {
            container_cardinality + state.rotate_left(13) as usize % (256 - container_cardinality)
        };
        positions.push(((container << 8) | local) as u32);
    }
    positions
}

pub fn strided_lookup_queries(cardinality: usize, stride: usize, hits: bool) -> Vec<u32> {
    assert!((cardinality - 1) * stride + usize::from(!hits) < 256);
    let mut state = 0x9e37_79b9_u32;
    let mut positions = Vec::with_capacity(QUERY_COUNT);
    for _ in 0..QUERY_COUNT {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let container = state as usize % MEDIUM_CONTAINER_COUNT;
        let index = state.rotate_left(13) as usize % cardinality;
        let local = index * stride + usize::from(!hits);
        positions.push(((container << 8) | local) as u32);
    }
    positions
}

pub mod mumbling {
    use ::mumbling::MumblingBitmapV1Builder;

    pub fn from_positions(positions: &[u32]) -> MumblingBitmapV1Builder {
        MumblingBitmapV1Builder::try_from_iter(positions.iter().copied()).unwrap()
    }

    pub fn from_sorted_positions(positions: &[u32]) -> MumblingBitmapV1Builder {
        MumblingBitmapV1Builder::from_sorted_iter(positions.iter().copied()).unwrap()
    }
}

pub mod roaring {
    use ::roaring::RoaringBitmap;

    pub fn from_positions(positions: &[u32]) -> RoaringBitmap {
        positions.iter().copied().collect()
    }

    pub fn from_sorted_positions(positions: &[u32]) -> RoaringBitmap {
        RoaringBitmap::from_sorted_iter(positions.iter().copied()).unwrap()
    }

    /// Read-only fixtures amortize run-container optimization outside timing.
    pub fn optimized(positions: &[u32]) -> RoaringBitmap {
        let mut bitmap = from_positions(positions);
        bitmap.optimize();
        bitmap
    }

    /// Includes the copy, sorting, and deduplication required by arbitrary input.
    pub fn from_unsorted_bulk(positions: &[u32]) -> RoaringBitmap {
        let mut sorted = positions.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        RoaringBitmap::from_sorted_iter(sorted).unwrap()
    }
}
