use alloc::vec;
use alloc::vec::Vec;

use super::MumblingBitmapV1;
use crate::Error;

#[test]
fn large_sparse_ordering_checks_all_pairs_as_unsigned_bytes() {
    for len in 17..=31 {
        let sorted: Vec<u8> = (0..len).map(|index| (index * 8) as u8).collect();
        assert!(super::simd::is_strictly_increasing_large(&sorted));
        for index in 1..len {
            let mut invalid = sorted.clone();
            invalid[index] = invalid[index - 1];
            assert!(!super::simd::is_strictly_increasing_large(&invalid));
            invalid.swap(index - 1, index);
            invalid[index - 1] = u8::MAX;
            assert!(!super::simd::is_strictly_increasing_large(&invalid));
        }
    }
}

#[test]
#[should_panic]
fn large_sparse_ordering_rejects_short_slices() {
    super::simd::is_strictly_increasing_large(&[0; 16]);
}

#[test]
#[should_panic]
fn large_sparse_ordering_rejects_oversized_slices() {
    super::simd::is_strictly_increasing_large(&[0; 32]);
}

fn sparse_bitmap() -> Vec<u8> {
    vec![
        1, 3, 0, 0, 1, 0, // header: v1, cardinality 3, one container
        0, 0, 3, // one PFOR descriptor whose minimum/value is 3
        0, 34, 255, // sparse payload
    ]
}

fn dense_bitmap() -> Vec<u8> {
    let mut bytes = vec![
        1, 32, 0, 0, 1, 0, // header: v1, cardinality 32, one container
        0, 0, 32, // one PFOR descriptor whose minimum/value is 32
    ];
    bytes.extend_from_slice(&[0xff; 4]);
    bytes.extend_from_slice(&[0; 28]);
    bytes
}

#[test]
fn looks_up_sparse_positions() {
    let bytes = sparse_bitmap();
    let bitmap = MumblingBitmapV1::parse(&bytes).unwrap();

    assert_eq!(bitmap.cardinality(), 3);
    assert_eq!(bitmap.container_count(), 1);
    assert!(bitmap.contains(0));
    assert!(bitmap.contains(34));
    assert!(bitmap.contains(255));
    assert!(!bitmap.contains(1));
    assert!(!bitmap.contains(256));
}

#[test]
fn looks_up_dense_positions_msb_first() {
    let bytes = dense_bitmap();
    let bitmap = MumblingBitmapV1::parse(&bytes).unwrap();

    for position in 0..32 {
        assert!(bitmap.contains(position));
    }
    assert!(!bitmap.contains(32));
    assert!(!bitmap.contains(255));
}

#[test]
fn looks_up_across_container_offsets() {
    let mut bytes = vec![
        1, 33, 0, 0, 3, 0, // header: cardinality 33, three containers
        8, 0, 0, 1, 0, 32, // descriptors [sparse(1), empty, dense]
        7,  // sparse payload for container 0
    ];
    bytes.extend_from_slice(&[0xff; 4]);
    bytes.extend_from_slice(&[0; 28]);

    let bitmap = MumblingBitmapV1::parse(&bytes).unwrap();
    assert!(bitmap.contains(7));
    assert!(!bitmap.contains(256 + 7));
    assert!(bitmap.contains(512));
    assert!(bitmap.contains(512 + 31));
    assert!(!bitmap.contains(512 + 32));
}

#[test]
fn parses_empty_bitmap() {
    let bytes = [1, 0, 0, 0, 0, 0];
    let bitmap = MumblingBitmapV1::parse(&bytes).unwrap();
    assert_eq!(bitmap.cardinality(), 0);
    assert_eq!(bitmap.container_count(), 0);
    assert!(!bitmap.contains(0));
}

#[test]
fn iterates_sparse_and_dense_positions_in_order() {
    let mut bytes = vec![
        1, 35, 0, 0, 3, 0, // header: cardinality 35, three containers
        8, 0, 0, 3, 0, 32, // descriptors [sparse(3), empty, dense]
        0, 34, 255, // sparse payload
    ];
    bytes.extend_from_slice(&[0x80; 32]);

    let bitmap = MumblingBitmapV1::parse(&bytes).unwrap();
    let expected = [0, 34, 255]
        .into_iter()
        .chain((0..256).step_by(8).map(|position| 512 + position))
        .collect::<Vec<_>>();
    let mut iter = bitmap.iter();

    assert_eq!(iter.size_hint(), (35, Some(35)));
    assert_eq!(iter.next(), Some(0));
    assert_eq!(iter.len(), 34);
    assert_eq!(iter.collect::<Vec<_>>(), expected[1..]);
    assert_eq!((&bitmap).into_iter().collect::<Vec<_>>(), expected);
}

#[test]
fn empty_iterator_is_fused() {
    let bytes = [1, 0, 0, 0, 0, 0];
    let bitmap = MumblingBitmapV1::parse(&bytes).unwrap();
    let mut iter = bitmap.iter();

    assert_eq!(iter.len(), 0);
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);
}

#[test]
fn rejects_invalid_containers() {
    let mut unsorted = sparse_bitmap();
    unsorted[9..].copy_from_slice(&[34, 0, 255]);
    assert!(matches!(
        MumblingBitmapV1::parse(&unsorted),
        Err(Error::UnsortedSparseContainer)
    ));

    let mut underfull = dense_bitmap();
    underfull[1] = 1;
    underfull[9..].fill(0);
    underfull[9] = 0x80;
    assert!(matches!(
        MumblingBitmapV1::parse(&underfull),
        Err(Error::UnderfullDenseContainer)
    ));

    let mut large_sparse = vec![
        1, 62, 0, 0, 2, 0, // header: 62 values, two containers
        0, 0, 31, // two identical sparse(31) descriptors
    ];
    large_sparse.extend(0..31);
    large_sparse.extend(0..31);
    large_sparse.swap(9 + 31 + 14, 9 + 31 + 15);
    assert!(matches!(
        MumblingBitmapV1::parse(&large_sparse),
        Err(Error::UnsortedSparseContainer)
    ));
}

#[test]
fn rejects_bad_metadata_and_framing() {
    let mut wrong_cardinality = sparse_bitmap();
    wrong_cardinality[1] = 2;
    assert!(matches!(
        MumblingBitmapV1::parse(&wrong_cardinality),
        Err(Error::CardinalityMismatch {
            declared: 2,
            actual: 3
        })
    ));

    let mut trailing = sparse_bitmap();
    trailing.push(0);
    assert!(matches!(
        MumblingBitmapV1::parse(&trailing),
        Err(Error::TrailingBytes)
    ));

    assert!(matches!(
        MumblingBitmapV1::parse(&[2, 0, 0, 0, 0, 0]),
        Err(Error::UnsupportedVersion(2))
    ));
}
