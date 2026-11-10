use core::fmt;

/// An error found while validating a serialized Mumbling bitmap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// The input ended before a complete field or section could be read.
    Truncated,
    /// Bytes remain after the final container.
    TrailingBytes,
    /// The format version is not supported.
    UnsupportedVersion(u8),
    /// The header declares more than 8,192 containers.
    TooManyContainers(u16),
    /// A PFOR header contains invalid bit widths or an exception count.
    InvalidPforHeader,
    /// A PFOR exception offset is duplicated or outside its chunk.
    InvalidPforExceptionOffset,
    /// A PFOR exception is not representable as an unsigned byte.
    InvalidPforValue,
    /// A descriptor uses a container type not defined by version 1.
    UnsupportedDescriptor(u8),
    /// Sparse positions are not strictly increasing.
    UnsortedSparseContainer,
    /// A dense container has fewer than 32 set bits.
    UnderfullDenseContainer,
    /// The header cardinality differs from the decoded containers.
    CardinalityMismatch {
        /// Cardinality stored in the header.
        declared: u32,
        /// Cardinality counted from the container payloads.
        actual: u32,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("truncated Mumbling bitmap"),
            Self::TrailingBytes => f.write_str("trailing bytes after Mumbling bitmap"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported Mumbling bitmap version {version}")
            }
            Self::TooManyContainers(count) => {
                write!(f, "container count {count} exceeds 8192")
            }
            Self::InvalidPforHeader => f.write_str("invalid PFOR header"),
            Self::InvalidPforExceptionOffset => f.write_str("invalid PFOR exception offset"),
            Self::InvalidPforValue => f.write_str("invalid PFOR value"),
            Self::UnsupportedDescriptor(descriptor) => {
                write!(f, "unsupported container descriptor 0x{descriptor:02X}")
            }
            Self::UnsortedSparseContainer => {
                f.write_str("sparse container positions are not strictly increasing")
            }
            Self::UnderfullDenseContainer => {
                f.write_str("dense container contains fewer than 32 positions")
            }
            Self::CardinalityMismatch { declared, actual } => write!(
                f,
                "cardinality mismatch: header declares {declared}, decoded {actual}"
            ),
        }
    }
}

impl core::error::Error for Error {}
