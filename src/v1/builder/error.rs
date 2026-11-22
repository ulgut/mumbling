//! Errors for version-1 bitmap construction.

use core::fmt;

use crate::MAX_VALUES;

/// Error returned when a position is outside the version-1 universe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PositionOutOfRange {
    /// The rejected position.
    pub position: u32,
}

impl fmt::Display for PositionOutOfRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "position {} is outside the Mumbling version-1 range 0..{}",
            self.position, MAX_VALUES
        )
    }
}

impl core::error::Error for PositionOutOfRange {}

/// Error returned when a half-open range is reversed or outside the v1 universe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RangeOutOfRange {
    /// Inclusive start of the rejected range.
    pub start: u32,
    /// Exclusive end of the rejected range.
    pub end: u32,
}

impl fmt::Display for RangeOutOfRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "range {}..{} is outside the Mumbling version-1 range 0..{}",
            self.start, self.end, MAX_VALUES
        )
    }
}

impl core::error::Error for RangeOutOfRange {}

/// Error returned while constructing from an allegedly sorted iterator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum FromSortedError {
    /// A position is outside the version-1 universe.
    PositionOutOfRange(PositionOutOfRange),
    /// Two adjacent positions are duplicated or decreasing.
    NotStrictlyIncreasing {
        /// The previously accepted position.
        previous: u32,
        /// The rejected position.
        current: u32,
    },
}

impl fmt::Display for FromSortedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PositionOutOfRange(error) => error.fmt(f),
            Self::NotStrictlyIncreasing { previous, current } => write!(
                f,
                "positions are not strictly increasing: {current} follows {previous}"
            ),
        }
    }
}

impl core::error::Error for FromSortedError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::PositionOutOfRange(error) => Some(error),
            Self::NotStrictlyIncreasing { .. } => None,
        }
    }
}
