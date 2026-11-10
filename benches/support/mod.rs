#![allow(dead_code)]

use mumbling::MAX_VALUES;
use mumbling::v1::MumblingBitmapV1Builder;
use roaring::RoaringBitmap;

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

pub fn mumbling(positions: &[u32]) -> MumblingBitmapV1Builder {
    MumblingBitmapV1Builder::try_from_iter(positions.iter().copied()).unwrap()
}

pub fn mumbling_sorted(positions: &[u32]) -> MumblingBitmapV1Builder {
    MumblingBitmapV1Builder::from_sorted_iter(positions.iter().copied()).unwrap()
}

pub fn roaring(positions: &[u32]) -> RoaringBitmap {
    positions.iter().copied().collect()
}
