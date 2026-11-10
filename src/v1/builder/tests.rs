use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;
use std::collections::BTreeSet;

use super::{FromSortedError, MumblingBitmapV1Builder, PositionOutOfRange, RangeOutOfRange};
use crate::v1::MumblingBitmapV1;
use crate::{MAX_POSITION, MAX_VALUES};

#[test]
fn inserts_removes_and_checks_boundaries() {
    let mut bitmap = MumblingBitmapV1Builder::new();
    for position in [0, 63, 64, 255, 256, MAX_POSITION] {
        assert_eq!(bitmap.insert(position), Ok(true));
        assert_eq!(bitmap.insert(position), Ok(false));
        assert!(bitmap.contains(position));
    }
    assert_eq!(bitmap.cardinality(), 6);
    assert_eq!(
        bitmap.insert(MAX_VALUES),
        Err(PositionOutOfRange {
            position: MAX_VALUES
        })
    );
    assert!(!bitmap.contains(MAX_VALUES));

    assert_eq!(bitmap.remove(MAX_POSITION), Ok(true));
    assert_eq!(bitmap.remove(MAX_POSITION), Ok(false));
    assert_eq!(bitmap.cardinality(), 5);
}

#[test]
fn validates_sorted_construction() {
    let bitmap = MumblingBitmapV1Builder::from_sorted_iter([1, 2, 100]).unwrap();
    assert_eq!(bitmap.cardinality(), 3);
    assert_eq!(
        MumblingBitmapV1Builder::from_sorted_iter([1, 1]),
        Err(FromSortedError::NotStrictlyIncreasing {
            previous: 1,
            current: 1
        })
    );
    assert!(matches!(
        MumblingBitmapV1Builder::from_sorted_iter([2, 1]),
        Err(FromSortedError::NotStrictlyIncreasing { .. })
    ));
}

#[test]
fn inserts_and_removes_ranges_wordwise() {
    let mut bitmap = MumblingBitmapV1Builder::new();
    assert_eq!(bitmap.insert_range(1..130), Ok(129));
    assert_eq!(bitmap.insert_range(64..192), Ok(62));
    assert_eq!(bitmap.cardinality(), 191);
    assert!(bitmap.contains(1));
    assert!(bitmap.contains(191));
    assert!(!bitmap.contains(0));
    assert!(!bitmap.contains(192));

    assert_eq!(bitmap.remove_range(63..129), Ok(66));
    assert_eq!(bitmap.cardinality(), 125);
    assert!(bitmap.contains(62));
    assert!(!bitmap.contains(63));
    assert!(!bitmap.contains(128));
    assert!(bitmap.contains(129));

    assert_eq!(bitmap.insert_range(10..10), Ok(0));
    assert_eq!(bitmap.remove_range(10..10), Ok(0));
    assert_eq!(bitmap.insert_range(0..MAX_VALUES), Ok(MAX_VALUES - 125));
    assert_eq!(bitmap.cardinality(), MAX_VALUES);
    assert_eq!(bitmap.remove_range(0..MAX_VALUES), Ok(MAX_VALUES));
    assert!(bitmap.is_empty());

    let reversed = Range { start: 10, end: 9 };
    assert_eq!(
        bitmap.insert_range(reversed),
        Err(RangeOutOfRange { start: 10, end: 9 })
    );
    assert_eq!(
        bitmap.remove_range(0..MAX_VALUES + 1),
        Err(RangeOutOfRange {
            start: 0,
            end: MAX_VALUES + 1
        })
    );
}

#[test]
fn preserves_lazy_prefix_semantics() {
    let mut prefix = MumblingBitmapV1Builder::new();
    assert_eq!(prefix.insert_range(0..130), Ok(130));
    assert!(prefix.words.is_empty());
    assert_eq!(prefix.prefix_len, 130);
    assert!(prefix.contains(0));
    assert!(prefix.contains(129));
    assert!(!prefix.contains(130));

    let materialized = MumblingBitmapV1Builder::from_sorted_iter(0..130).unwrap();
    assert_eq!(prefix, materialized);
    assert_eq!(prefix.serialize(), materialized.serialize());

    assert_eq!(prefix.remove_range(120..200), Ok(10));
    assert_eq!(prefix.prefix_len, 120);
    assert_eq!(prefix.insert(120), Ok(true));
    assert_eq!(prefix.prefix_len, 121);
    assert_eq!(prefix.remove(64), Ok(true));
    assert_eq!(prefix.prefix_len, 0);
    assert!(!prefix.contains(64));
    assert_eq!(prefix.cardinality(), 120);

    let mut left = MumblingBitmapV1Builder::new();
    left.insert_range(0..100).unwrap();
    let right = MumblingBitmapV1Builder::try_from([50, 150]).unwrap();
    assert_eq!(left.intersection_len(&right), 1);
    assert_eq!((&left | &right).cardinality(), 101);
    assert_eq!((&left & &right).cardinality(), 1);
    assert_eq!((&left - &right).cardinality(), 99);
    assert_eq!((&left ^ &right).cardinality(), 100);
}

#[test]
fn performs_set_operations() {
    let left = MumblingBitmapV1Builder::try_from([1, 2, 64, 10_000]).unwrap();
    let right = MumblingBitmapV1Builder::try_from([2, 3, 64, MAX_POSITION]).unwrap();

    assert_bitmap_eq(&(&left | &right), [1, 2, 3, 64, 10_000, MAX_POSITION]);
    assert_bitmap_eq(&(&left & &right), [2, 64]);
    assert_bitmap_eq(&(&left - &right), [1, 10_000]);
    assert_bitmap_eq(&(&left ^ &right), [1, 3, 10_000, MAX_POSITION]);
    assert_eq!(left.union_len(&right), 6);
    assert_eq!(left.intersection_len(&right), 2);
    assert_eq!(left.difference_len(&right), 2);
    assert_eq!(left.symmetric_difference_len(&right), 4);
    assert!(!left.is_disjoint(&right));
    assert!(
        MumblingBitmapV1Builder::try_from([1, 64])
            .unwrap()
            .is_subset(&left)
    );
    assert!(left.is_superset(&MumblingBitmapV1Builder::try_from([1, 64]).unwrap()));
    assert!(
        MumblingBitmapV1Builder::try_from([7, 8])
            .unwrap()
            .is_disjoint(&left)
    );
    assert!(!right.is_subset(&left));
}

#[test]
fn serializes_known_sparse_and_dense_bitmaps() {
    let sparse = MumblingBitmapV1Builder::try_from([0, 34, 255]).unwrap();
    assert_eq!(
        sparse.serialize(),
        [
            1, 3, 0, 0, 1, 0, // header
            0, 0, 3, // descriptor 3
            0, 34, 255, // sparse payload
        ]
    );

    let dense = MumblingBitmapV1Builder::from_sorted_iter(0..32).unwrap();
    let mut expected = vec![
        1, 32, 0, 0, 1, 0, // header
        0, 0, 32, // descriptor 32
    ];
    expected.extend_from_slice(&[0xff; 4]);
    expected.extend_from_slice(&[0; 28]);
    assert_eq!(dense.serialize(), expected);
}

#[test]
fn round_trips_container_and_descriptor_boundaries() {
    let positions = [0, 30, 31, 32, 255, 256, 257, 65_535, 65_536, MAX_POSITION];
    let bitmap = MumblingBitmapV1Builder::try_from(positions).unwrap();
    let encoded = bitmap.serialize();
    assert_eq!(bitmap.serialized_size(), encoded.len());
    let prepared = bitmap.prepare_serialization();
    assert_eq!(prepared.serialized_size(), encoded.len());
    assert_eq!(prepared.serialize(), encoded);
    let mut appended = vec![9, 8, 7];
    prepared.serialize_into(&mut appended);
    assert_eq!(&appended[3..], encoded);
    let decoded = MumblingBitmapV1::parse(&encoded).unwrap();

    assert_eq!(decoded.iter().collect::<Vec<_>>(), positions);
}

#[test]
fn switches_wire_container_at_32_positions() {
    let mut bitmap = MumblingBitmapV1Builder::from_sorted_iter(0..31).unwrap();
    let sparse = bitmap.serialize();
    assert_eq!(MumblingBitmapV1::parse(&sparse).unwrap().cardinality(), 31);

    bitmap.insert(31).unwrap();
    let dense = bitmap.serialize();
    assert_eq!(MumblingBitmapV1::parse(&dense).unwrap().cardinality(), 32);
    assert_eq!(dense.len(), sparse.len() + 1);

    bitmap.remove(31).unwrap();
    assert_eq!(bitmap.serialize(), sparse);
}

#[test]
fn agrees_with_btree_set_model() {
    let mut bitmap = MumblingBitmapV1Builder::new();
    let mut model = BTreeSet::new();
    let mut state = 0x1234_5678_u32;

    for step in 0..20_000 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let position = state % MAX_VALUES;
        if step % 3 == 0 {
            assert_eq!(bitmap.remove(position).unwrap(), model.remove(&position));
        } else {
            assert_eq!(bitmap.insert(position).unwrap(), model.insert(position));
        }
        assert_eq!(bitmap.cardinality() as usize, model.len());
    }

    for position in 0..MAX_VALUES {
        assert_eq!(bitmap.contains(position), model.contains(&position));
    }

    let encoded = bitmap.serialize();
    let decoded = MumblingBitmapV1::parse(&encoded).unwrap();
    assert!(decoded.iter().eq(model.iter().copied()));
}

fn assert_bitmap_eq<const N: usize>(bitmap: &MumblingBitmapV1Builder, expected: [u32; N]) {
    assert_eq!(bitmap.cardinality() as usize, N);
    for position in 0..MAX_VALUES {
        assert_eq!(bitmap.contains(position), expected.contains(&position));
    }
}
