//! A safe implementation of the Apache Iceberg Mumbling bitmap format.
//!
//! [`MumblingBitmapV1`] validates a serialized version-1 bitmap once, decodes
//! its PFOR descriptor array, and builds container offsets. Point lookups are
//! then allocation-free and require no scanning across preceding containers.

#![no_std]
#![deny(missing_docs)]

extern crate alloc;
#[cfg(test)]
extern crate std;

mod error;
pub mod v1;

pub use error::Error;
pub use v1::{
    FromSortedError, MumblingBitmapV1, MumblingBitmapV1Builder, MumblingBitmapV1Iter,
    MumblingBitmapV1SerializationPlan, PositionOutOfRange, RangeOutOfRange,
};

/// Maximum number of values representable by a version-1 Mumbling bitmap.
pub const MAX_VALUES: u32 = 2_097_152;

/// Maximum valid position in a version-1 Mumbling bitmap.
pub const MAX_POSITION: u32 = MAX_VALUES - 1;
