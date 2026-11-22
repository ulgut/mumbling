//! Reader and format constants for Mumbling bitmap version 1.

use alloc::vec::Vec;

use crate::{Error, MAX_VALUES};

mod builder;
mod pfor;
mod simd;
#[cfg(test)]
mod tests;

use simd::{
    contains_sparse, is_strictly_increasing_large, validate_uniform_dense, validate_uniform_sparse,
};

pub use builder::{
    FromSortedError, MumblingBitmapV1Builder, MumblingBitmapV1SerializationPlan,
    PositionOutOfRange, RangeOutOfRange,
};

const VERSION: u8 = 1;
const HEADER_LEN: usize = 6;
const MAX_CONTAINERS: u16 = 8_192;
const DENSE_DESCRIPTOR: u8 = 0x20;
const DESCRIPTOR_TYPE_MASK: u8 = 0xe0;
const SPARSE_LENGTH_MASK: u8 = 0x1f;

/// A validated, read-only view of a serialized Mumbling bitmap.
///
/// Parsing borrows the serialized bytes, decodes the PFOR descriptor array,
/// and builds a container-offset index. The descriptor and offset indexes use
/// one allocation each; container payloads remain zero-copy.
#[derive(Debug)]
pub struct MumblingBitmapV1<'a> {
    bytes: &'a [u8],
    descriptors: Vec<u8>,
    offsets: Vec<u32>,
    payload: &'a [u8],
    cardinality: u32,
    uniform_descriptor: Option<u8>,
}

impl<'a> MumblingBitmapV1<'a> {
    /// Parses and strictly validates a complete version-1 Mumbling bitmap.
    ///
    /// This checks PFOR structure, container types and sizes, sparse ordering,
    /// dense-container cardinality, total cardinality, and trailing bytes.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        let header = bytes.get(..HEADER_LEN).ok_or(Error::Truncated)?;
        if header[0] != VERSION {
            return Err(Error::UnsupportedVersion(header[0]));
        }

        let cardinality =
            u32::from(header[1]) | (u32::from(header[2]) << 8) | (u32::from(header[3]) << 16);
        let container_count = u16::from_le_bytes([header[4], header[5]]);

        if container_count > MAX_CONTAINERS {
            return Err(Error::TooManyContainers(container_count));
        }
        if cardinality > MAX_VALUES {
            return Err(Error::CardinalityMismatch {
                declared: cardinality,
                actual: 0,
            });
        }

        let (descriptors, descriptor_bytes) =
            pfor::decode(&bytes[HEADER_LEN..], usize::from(container_count))?;
        let payload_start = HEADER_LEN + descriptor_bytes;
        let payload = bytes.get(payload_start..).ok_or(Error::Truncated)?;

        let uniform_descriptor = descriptors
            .first()
            .copied()
            .filter(|descriptor| descriptors.iter().all(|candidate| candidate == descriptor));
        let (offsets, actual_cardinality) =
            index_and_validate(&descriptors, payload, uniform_descriptor)?;

        if actual_cardinality != cardinality {
            return Err(Error::CardinalityMismatch {
                declared: cardinality,
                actual: actual_cardinality,
            });
        }

        Ok(Self {
            bytes,
            descriptors,
            offsets,
            payload,
            cardinality,
            uniform_descriptor,
        })
    }

    /// Returns whether `position` is present in the bitmap.
    ///
    /// Positions outside the version-1 universe return `false`.
    #[inline]
    pub fn contains(&self, position: u32) -> bool {
        let container_index = (position >> 8) as usize;
        let Some(&descriptor) = self.descriptors.get(container_index) else {
            return false;
        };
        let offset = self.offsets[container_index] as usize;
        let position_in_container = position as u8;

        if descriptor & DESCRIPTOR_TYPE_MASK == DENSE_DESCRIPTOR {
            let byte = self.payload[offset + (usize::from(position_in_container) >> 3)];
            let mask = 0x80_u8 >> (position_in_container & 7);
            byte & mask != 0
        } else {
            let len = usize::from(descriptor & SPARSE_LENGTH_MASK);
            contains_sparse(&self.payload[offset..offset + len], position_in_container)
        }
    }

    /// Returns the number of positions, using Roaring's `u64` count convention.
    #[inline]
    pub const fn len(&self) -> u64 {
        self.cardinality as u64
    }

    /// Returns whether this bitmap contains no positions.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.cardinality == 0
    }

    /// Returns the version-1 cardinality as a `u32`.
    ///
    /// Prefer [`Self::len`] when sharing query code with Roaring.
    #[inline]
    pub const fn cardinality(&self) -> u32 {
        self.cardinality
    }

    /// Returns the number of logical 256-position containers.
    #[inline]
    pub fn container_count(&self) -> usize {
        self.descriptors.len()
    }

    /// Returns an iterator over all set positions in ascending order.
    #[inline]
    pub fn iter(&self) -> MumblingBitmapV1Iter<'_> {
        MumblingBitmapV1Iter {
            descriptors: &self.descriptors,
            offsets: &self.offsets,
            payload: self.payload,
            container_index: 0,
            descriptor: self.descriptors.first().copied().unwrap_or(0),
            payload_offset: 0,
            container_base: 0,
            local_index: 0,
            dense_word: 0,
            dense_word_index: 0,
            remaining: self.cardinality,
            singleton_sparse: self.uniform_descriptor == Some(1),
        }
    }

    /// Returns the complete validated serialized representation.
    #[inline]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }
}

/// An ascending iterator over positions in a [`MumblingBitmapV1`].
///
/// Sparse containers are traversed directly. Dense containers skip empty
/// 64-bit words and use `leading_zeros` to locate their next MSB-first set bit.
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct MumblingBitmapV1Iter<'a> {
    descriptors: &'a [u8],
    offsets: &'a [u32],
    payload: &'a [u8],
    container_index: usize,
    descriptor: u8,
    payload_offset: usize,
    container_base: u32,
    local_index: u16,
    dense_word: u64,
    dense_word_index: u8,
    remaining: u32,
    singleton_sparse: bool,
}

impl MumblingBitmapV1Iter<'_> {
    #[inline]
    fn advance_container(&mut self) {
        self.container_index += 1;
        self.container_base += 256;
        self.local_index = 0;
        self.dense_word = 0;
        self.dense_word_index = 0;

        if let Some(&descriptor) = self.descriptors.get(self.container_index) {
            self.descriptor = descriptor;
            self.payload_offset = self.offsets[self.container_index] as usize;
        }
    }
}

impl Iterator for MumblingBitmapV1Iter<'_> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.singleton_sparse {
            if self.remaining == 0 {
                return None;
            }
            let position = self.container_base + u32::from(self.payload[self.payload_offset]);
            self.container_base += 256;
            self.payload_offset += 1;
            self.remaining -= 1;
            return Some(position);
        }

        while self.remaining != 0 {
            if self.descriptor & DESCRIPTOR_TYPE_MASK == DENSE_DESCRIPTOR {
                loop {
                    if self.dense_word != 0 {
                        let bit_index = self.dense_word.leading_zeros();
                        self.dense_word &= !(1_u64 << (63 - bit_index));
                        let position = (u32::from(self.dense_word_index) - 1) * 64 + bit_index;
                        self.remaining -= 1;
                        return Some(self.container_base + position);
                    }

                    if self.dense_word_index == 4 {
                        break;
                    }

                    let start = self.payload_offset + usize::from(self.dense_word_index) * 8;
                    self.dense_word = u64::from_be_bytes(
                        self.payload[start..start + 8]
                            .try_into()
                            .expect("validated dense container"),
                    );
                    self.dense_word_index += 1;
                }
            } else {
                let len = u16::from(self.descriptor & SPARSE_LENGTH_MASK);
                if self.local_index < len {
                    let position = self.container_base
                        + u32::from(
                            self.payload[self.payload_offset + usize::from(self.local_index)],
                        );
                    self.local_index += 1;
                    self.remaining -= 1;
                    return Some(position);
                }
            }

            self.advance_container();
        }

        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for MumblingBitmapV1Iter<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.remaining as usize
    }
}

impl core::iter::FusedIterator for MumblingBitmapV1Iter<'_> {}

impl<'bitmap> IntoIterator for &'bitmap MumblingBitmapV1<'_> {
    type Item = u32;
    type IntoIter = MumblingBitmapV1Iter<'bitmap>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

fn container_len(descriptor: u8) -> Result<usize, Error> {
    match descriptor & DESCRIPTOR_TYPE_MASK {
        0 => Ok(usize::from(descriptor & SPARSE_LENGTH_MASK)),
        DENSE_DESCRIPTOR => Ok(32),
        _ => Err(Error::UnsupportedDescriptor(descriptor)),
    }
}

fn index_and_validate(
    descriptors: &[u8],
    payload: &[u8],
    uniform_descriptor: Option<u8>,
) -> Result<(Vec<u32>, u32), Error> {
    if let Some(descriptor) = uniform_descriptor {
        return index_and_validate_uniform(descriptor, descriptors.len(), payload);
    }

    let mut offsets = Vec::with_capacity(descriptors.len() + 1);
    offsets.push(0);
    let mut payload_len = 0_usize;
    let mut cardinality = 0_u32;
    for &descriptor in descriptors {
        let start = payload_len;
        payload_len = payload_len
            .checked_add(container_len(descriptor)?)
            .ok_or(Error::Truncated)?;
        let container = payload.get(start..payload_len).ok_or(Error::Truncated)?;

        if descriptor & DESCRIPTOR_TYPE_MASK == DENSE_DESCRIPTOR {
            let count = container.iter().map(|byte| byte.count_ones()).sum::<u32>();
            if count < 32 {
                return Err(Error::UnderfullDenseContainer);
            }
            cardinality += count;
        } else {
            let is_sorted = if container.len() >= 17 {
                is_strictly_increasing_large(container)
            } else {
                container.windows(2).all(|pair| pair[0] < pair[1])
            };
            if !is_sorted {
                return Err(Error::UnsortedSparseContainer);
            }
            cardinality += container.len() as u32;
        }

        offsets.push(u32::try_from(payload_len).map_err(|_| Error::Truncated)?);
    }
    validate_payload_len(payload.len(), payload_len)?;
    Ok((offsets, cardinality))
}

fn index_and_validate_uniform(
    descriptor: u8,
    container_count: usize,
    payload: &[u8],
) -> Result<(Vec<u32>, u32), Error> {
    let width = container_len(descriptor)?;
    let payload_len = width.checked_mul(container_count).ok_or(Error::Truncated)?;
    validate_payload_len(payload.len(), payload_len)?;

    let mut offsets = Vec::with_capacity(container_count + 1);
    offsets.extend((0..=container_count).map(|index| (index * width) as u32));

    let cardinality = if descriptor & DESCRIPTOR_TYPE_MASK == DENSE_DESCRIPTOR {
        validate_uniform_dense(payload)?
    } else {
        if !validate_uniform_sparse(payload, width) {
            return Err(Error::UnsortedSparseContainer);
        }
        (width * container_count) as u32
    };
    Ok((offsets, cardinality))
}

#[inline]
fn validate_payload_len(actual: usize, expected: usize) -> Result<(), Error> {
    match actual.cmp(&expected) {
        core::cmp::Ordering::Less => Err(Error::Truncated),
        core::cmp::Ordering::Greater => Err(Error::TrailingBytes),
        core::cmp::Ordering::Equal => Ok(()),
    }
}
