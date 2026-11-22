use alloc::vec::Vec;
use core::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Range, Sub, SubAssign,
};

use crate::MAX_VALUES;

use super::pfor::{self, VALUES_PER_CHUNK as DESCRIPTORS_PER_CHUNK};
use super::{HEADER_LEN, VERSION};

mod error;
#[cfg(test)]
mod tests;

pub use error::{FromSortedError, PositionOutOfRange, RangeOutOfRange};

const WORDS_PER_CONTAINER: usize = 4;
const MAX_WORDS: usize = MAX_VALUES as usize / u64::BITS as usize;

/// A mutable construction representation for Mumbling version 1.
///
/// General storage uses dense words, grows to the highest inserted position,
/// and is capped at 256 KiB. An initial `0..end` range remains an allocation-free
/// prefix until a mutation creates a hole. The wire-format sparse/dense decision
/// is intentionally deferred until serialization.
#[derive(Clone, Debug, Default)]
pub struct MumblingBitmapV1Builder {
    words: Vec<u64>,
    cardinality: u32,
    prefix_len: u32,
}

/// An owned snapshot of a version-1 wire image.
///
/// Preparing a plan performs serialization once. Subsequent builder mutations
/// do not affect the snapshot, which can also outlive the builder.
#[derive(Debug)]
pub struct MumblingBitmapV1SerializationPlan {
    encoded: Vec<u8>,
}

impl MumblingBitmapV1Builder {
    /// Creates an empty builder without allocating.
    #[inline]
    pub const fn new() -> Self {
        Self {
            words: Vec::new(),
            cardinality: 0,
            prefix_len: 0,
        }
    }

    /// Constructs a bitmap from positions in arbitrary order.
    ///
    /// Duplicate positions are ignored.
    ///
    /// # Errors
    ///
    /// Returns [`PositionOutOfRange`] for any position at or above [`MAX_VALUES`].
    pub fn try_from_iter<I>(positions: I) -> Result<Self, PositionOutOfRange>
    where
        I: IntoIterator<Item = u32>,
    {
        let positions = positions.into_iter();
        let mut bitmap = Self::new();
        bitmap.words.reserve(positions.size_hint().0.min(MAX_WORDS));
        for position in positions {
            validate_position(position)?;
            bitmap.insert_word(position);
        }
        Ok(bitmap)
    }

    /// Constructs a bitmap from strictly increasing positions.
    ///
    /// This rejects duplicates and out-of-order input rather than silently
    /// normalizing it.
    ///
    /// # Errors
    ///
    /// Returns [`FromSortedError`] for an out-of-range position or input that
    /// is not strictly increasing.
    pub fn from_sorted_iter<I>(positions: I) -> Result<Self, FromSortedError>
    where
        I: IntoIterator<Item = u32>,
    {
        let positions = positions.into_iter();
        let mut bitmap = Self::new();
        bitmap.words.reserve(positions.size_hint().0.min(MAX_WORDS));
        let mut previous = None;

        for position in positions {
            if position >= MAX_VALUES {
                return Err(FromSortedError::PositionOutOfRange(PositionOutOfRange {
                    position,
                }));
            }
            if let Some(previous) = previous
                && position <= previous
            {
                return Err(FromSortedError::NotStrictlyIncreasing {
                    previous,
                    current: position,
                });
            }
            let word_index = (position >> 6) as usize;
            if word_index >= bitmap.words.len() {
                bitmap.words.resize(word_index + 1, 0);
            }
            bitmap.words[word_index] |= 1_u64 << (position & 63);
            bitmap.cardinality += 1;
            previous = Some(position);
        }

        Ok(bitmap)
    }

    /// Inserts a position, returning whether it was newly inserted.
    ///
    /// # Errors
    ///
    /// Returns [`PositionOutOfRange`] without mutation if `position >= MAX_VALUES`.
    pub fn insert(&mut self, position: u32) -> Result<bool, PositionOutOfRange> {
        validate_position(position)?;
        if self.prefix_len != 0 {
            if position < self.prefix_len {
                return Ok(false);
            }
            if position == self.prefix_len {
                self.prefix_len += 1;
                self.cardinality += 1;
                return Ok(true);
            }
            self.materialize_prefix();
        }
        Ok(self.insert_word(position))
    }

    fn insert_word(&mut self, position: u32) -> bool {
        let word_index = (position >> 6) as usize;
        if word_index >= self.words.len() {
            self.words.resize(word_index + 1, 0);
        }

        let mask = 1_u64 << (position & 63);
        let was_present = self.words[word_index] & mask != 0;
        if !was_present {
            self.words[word_index] |= mask;
            self.cardinality += 1;
        }
        !was_present
    }

    /// Removes a position, returning whether it was present.
    ///
    /// # Errors
    ///
    /// Returns [`PositionOutOfRange`] without mutation if `position >= MAX_VALUES`.
    pub fn remove(&mut self, position: u32) -> Result<bool, PositionOutOfRange> {
        validate_position(position)?;
        if self.prefix_len != 0 {
            if position >= self.prefix_len {
                return Ok(false);
            }
            if position + 1 == self.prefix_len {
                self.prefix_len -= 1;
                self.cardinality -= 1;
                return Ok(true);
            }
            self.materialize_prefix();
        }
        let word_index = (position >> 6) as usize;
        let Some(word) = self.words.get_mut(word_index) else {
            return Ok(false);
        };

        let mask = 1_u64 << (position & 63);
        let was_present = *word & mask != 0;
        if was_present {
            *word &= !mask;
            self.cardinality -= 1;
            trim_trailing_zeros(&mut self.words);
        }
        Ok(was_present)
    }

    /// Inserts all positions in the half-open `start..end` range.
    ///
    /// Returns the number of newly inserted positions.
    ///
    /// # Errors
    ///
    /// Returns [`RangeOutOfRange`] without mutation if `start > end` or
    /// `end > MAX_VALUES`. Empty ranges are accepted.
    pub fn insert_range(&mut self, range: Range<u32>) -> Result<u32, RangeOutOfRange> {
        validate_range(&range)?;
        if range.is_empty() {
            return Ok(0);
        }

        if self.prefix_len != 0 {
            if range.start <= self.prefix_len {
                let previous = self.prefix_len;
                self.prefix_len = self.prefix_len.max(range.end);
                self.cardinality = self.prefix_len;
                return Ok(self.prefix_len - previous);
            }
            self.materialize_prefix();
        }

        if self.cardinality == 0 && range.start == 0 {
            debug_assert!(self.words.is_empty());
            self.prefix_len = range.end;
            self.cardinality = range.end;
            return Ok(range.end);
        }

        let first_word = (range.start >> 6) as usize;
        let last_word = ((range.end - 1) >> 6) as usize;
        if last_word >= self.words.len() {
            self.words.resize(last_word + 1, 0);
        }

        let mut inserted = 0;
        for word_index in first_word..=last_word {
            let mask = range_mask(word_index, &range);
            inserted += (mask & !self.words[word_index]).count_ones();
            self.words[word_index] |= mask;
        }
        self.cardinality += inserted;
        Ok(inserted)
    }

    /// Removes all positions in the half-open `start..end` range.
    ///
    /// Returns the number of removed positions.
    ///
    /// # Errors
    ///
    /// Returns [`RangeOutOfRange`] without mutation if `start > end` or
    /// `end > MAX_VALUES`. Empty ranges are accepted.
    pub fn remove_range(&mut self, range: Range<u32>) -> Result<u32, RangeOutOfRange> {
        validate_range(&range)?;
        if self.prefix_len != 0 {
            let overlap_end = range.end.min(self.prefix_len);
            if range.start >= overlap_end {
                return Ok(0);
            }
            if range.start == 0 && range.end >= self.prefix_len {
                let removed = self.prefix_len;
                self.prefix_len = 0;
                self.cardinality = 0;
                return Ok(removed);
            }
            if range.end >= self.prefix_len {
                let removed = self.prefix_len - range.start;
                self.prefix_len = range.start;
                self.cardinality = self.prefix_len;
                return Ok(removed);
            }
            self.materialize_prefix();
        }
        if range.is_empty() || self.words.is_empty() {
            return Ok(0);
        }

        let first_word = (range.start >> 6) as usize;
        if first_word >= self.words.len() {
            return Ok(0);
        }
        let last_word = (((range.end - 1) >> 6) as usize).min(self.words.len() - 1);
        let mut removed = 0;
        for word_index in first_word..=last_word {
            let mask = range_mask(word_index, &range);
            removed += (mask & self.words[word_index]).count_ones();
            self.words[word_index] &= !mask;
        }
        self.cardinality -= removed;
        trim_trailing_zeros(&mut self.words);
        Ok(removed)
    }

    /// Returns whether a valid position is present.
    ///
    /// Positions outside the version-1 universe return `false`.
    #[inline]
    pub fn contains(&self, position: u32) -> bool {
        if position >= MAX_VALUES {
            return false;
        }
        if self.prefix_len != 0 {
            return position < self.prefix_len;
        }
        let word_index = (position >> 6) as usize;
        self.words
            .get(word_index)
            .is_some_and(|word| word & (1_u64 << (position & 63)) != 0)
    }

    /// Returns the number of positions in the bitmap.
    #[inline]
    pub const fn cardinality(&self) -> u32 {
        self.cardinality
    }

    /// Returns the number of positions, using Roaring's `u64` count convention.
    #[inline]
    pub const fn len(&self) -> u64 {
        self.cardinality as u64
    }

    /// Returns whether the bitmap contains no positions.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.cardinality == 0
    }

    /// Removes all positions while retaining allocated storage.
    pub fn clear(&mut self) {
        self.words.clear();
        self.cardinality = 0;
        self.prefix_len = 0;
    }

    /// Returns whether this bitmap and `other` have no positions in common.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        if self.prefix_len != 0 || other.prefix_len != 0 {
            return self.intersection_len(other) == 0;
        }
        self.words
            .iter()
            .zip(&other.words)
            .all(|(left, right)| left & right == 0)
    }

    /// Returns whether every position in this bitmap is present in `other`.
    pub fn is_subset(&self, other: &Self) -> bool {
        if self.prefix_len != 0 || other.prefix_len != 0 {
            return (0..self.effective_word_len())
                .all(|index| self.word(index) & !other.word(index) == 0);
        }
        self.words
            .iter()
            .zip(&other.words)
            .all(|(left, right)| left & !right == 0)
            && self.words[other.words.len().min(self.words.len())..]
                .iter()
                .all(|word| *word == 0)
    }

    /// Returns whether every position in `other` is present in this bitmap.
    #[inline]
    pub fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }

    /// Serializes once and caches the complete wire image for repeated use.
    ///
    /// The returned snapshot is independent of this builder. Preparation does
    /// the same encoding work as [`Self::serialize`]; subsequent writes copy bytes.
    pub fn prepare_serialization(&self) -> MumblingBitmapV1SerializationPlan {
        MumblingBitmapV1SerializationPlan {
            encoded: self.serialize(),
        }
    }

    /// Serializes this bitmap using the Mumbling version-1 wire format.
    pub fn serialize(&self) -> Vec<u8> {
        let mut output = Vec::with_capacity(self.serialization_capacity_bound());
        self.serialize_into(&mut output);
        output
    }

    /// Appends a serialized Mumbling version-1 bitmap to `output`.
    ///
    /// Existing bytes in `output` are retained.
    pub fn serialize_into(&self, output: &mut Vec<u8>) {
        output.reserve(self.serialization_capacity_bound());
        let container_count = self.container_count();
        output.push(VERSION);
        output.extend_from_slice(&self.cardinality.to_le_bytes()[..3]);
        output.extend_from_slice(&(container_count as u16).to_le_bytes());

        let mut descriptors = [0_u8; DESCRIPTORS_PER_CHUNK];
        for chunk_start in (0..container_count).step_by(DESCRIPTORS_PER_CHUNK) {
            let chunk_len = (container_count - chunk_start).min(DESCRIPTORS_PER_CHUNK);
            self.fill_descriptors(chunk_start, &mut descriptors[..chunk_len]);
            pfor::encode(&descriptors[..chunk_len], output);
        }

        for container_index in 0..container_count {
            if self.descriptor(container_index) < 32 {
                self.serialize_sparse_container(container_index, output);
            } else {
                self.serialize_dense_container(container_index, output);
            }
        }
    }

    /// Returns the exact serialized length in bytes without allocating.
    ///
    /// This analyzes container descriptors and their compression; it is not a
    /// constant-time query. [`Self::serialize`] does not require calling it first.
    pub fn serialized_size(&self) -> usize {
        let container_count = self.container_count();
        let mut size = HEADER_LEN;
        let mut descriptors = [0_u8; DESCRIPTORS_PER_CHUNK];
        for chunk_start in (0..container_count).step_by(DESCRIPTORS_PER_CHUNK) {
            let chunk_len = (container_count - chunk_start).min(DESCRIPTORS_PER_CHUNK);
            self.fill_descriptors(chunk_start, &mut descriptors[..chunk_len]);
            size += pfor::encoded_len(&descriptors[..chunk_len]);
            size += descriptors[..chunk_len]
                .iter()
                .map(|&descriptor| usize::from(descriptor))
                .sum::<usize>();
        }
        size
    }

    /// Returns the cardinality of the intersection without allocating it.
    #[inline]
    pub fn intersection_len(&self, other: &Self) -> u32 {
        if self.prefix_len != 0 || other.prefix_len != 0 {
            return intersection_len_indexed(self, other);
        }
        intersection_len(&self.words, &other.words)
    }

    /// Returns the cardinality of the union without allocating it.
    #[inline]
    pub fn union_len(&self, other: &Self) -> u32 {
        self.cardinality + other.cardinality - self.intersection_len(other)
    }

    /// Returns the cardinality of `self - other` without allocating it.
    #[inline]
    pub fn difference_len(&self, other: &Self) -> u32 {
        self.cardinality - self.intersection_len(other)
    }

    /// Returns the cardinality of the symmetric difference without allocating it.
    #[inline]
    pub fn symmetric_difference_len(&self, other: &Self) -> u32 {
        self.cardinality + other.cardinality - 2 * self.intersection_len(other)
    }

    fn serialization_capacity_bound(&self) -> usize {
        let container_count = self.container_count();
        let descriptor_bound =
            container_count.div_ceil(DESCRIPTORS_PER_CHUNK) * 3 + (container_count * 6).div_ceil(8);
        let payload_bound = (self.cardinality as usize).min(container_count * 32);
        HEADER_LEN + descriptor_bound + payload_bound
    }

    #[inline]
    fn container_count(&self) -> usize {
        self.effective_word_len().div_ceil(WORDS_PER_CONTAINER)
    }

    fn fill_descriptors(&self, first_container: usize, descriptors: &mut [u8]) {
        for (offset, descriptor) in descriptors.iter_mut().enumerate() {
            *descriptor = self.descriptor(first_container + offset);
        }
    }

    fn descriptor(&self, container_index: usize) -> u8 {
        let first_word = container_index * WORDS_PER_CONTAINER;
        if self.prefix_len == 0 {
            return self.words[first_word..self.words.len().min(first_word + WORDS_PER_CONTAINER)]
                .iter()
                .map(|word| word.count_ones())
                .sum::<u32>()
                .min(32) as u8;
        }
        (first_word..(first_word + WORDS_PER_CONTAINER).min(self.effective_word_len()))
            .map(|index| self.word(index).count_ones())
            .sum::<u32>()
            .min(32) as u8
    }

    fn serialize_sparse_container(&self, container_index: usize, output: &mut Vec<u8>) {
        let first_word = container_index * WORDS_PER_CONTAINER;
        let output_start = output.len();
        let mut position_count = 0;
        let spare = output.spare_capacity_mut();
        if self.prefix_len == 0 {
            for word_offset in 0..WORDS_PER_CONTAINER {
                let mut word = self
                    .words
                    .get(first_word + word_offset)
                    .copied()
                    .unwrap_or(0);
                while word != 0 {
                    let bit = word.trailing_zeros();
                    let position = (word_offset as u32 * 64 + bit) as u8;
                    // SAFETY: `serialize_into` reserves the maximum possible payload
                    // before encoding. Sparse containers contain at most 31 values,
                    // and each loop iteration corresponds to one set bit.
                    unsafe {
                        spare.get_unchecked_mut(position_count).write(position);
                    }
                    position_count += 1;
                    word &= word - 1;
                }
            }
        } else {
            for word_offset in 0..WORDS_PER_CONTAINER {
                let mut word = self.word(first_word + word_offset);
                while word != 0 {
                    let bit = word.trailing_zeros();
                    let position = (word_offset as u32 * 64 + bit) as u8;
                    // SAFETY: same capacity and initialization invariant as the
                    // dense-word path above.
                    unsafe {
                        spare.get_unchecked_mut(position_count).write(position);
                    }
                    position_count += 1;
                    word &= word - 1;
                }
            }
        }
        // SAFETY: the preceding loop initialized exactly `position_count`
        // elements in the vector's spare capacity.
        unsafe {
            output.set_len(output_start + position_count);
        }
    }

    fn serialize_dense_container(&self, container_index: usize, output: &mut Vec<u8>) {
        let first_word = container_index * WORDS_PER_CONTAINER;
        if self.prefix_len == 0 {
            for word_offset in 0..WORDS_PER_CONTAINER {
                let word = self
                    .words
                    .get(first_word + word_offset)
                    .copied()
                    .unwrap_or(0);
                output.extend(word.to_le_bytes().map(u8::reverse_bits));
            }
            return;
        }
        for word_offset in 0..WORDS_PER_CONTAINER {
            let word = self.word(first_word + word_offset);
            output.extend(word.to_le_bytes().map(u8::reverse_bits));
        }
    }

    #[inline]
    fn effective_word_len(&self) -> usize {
        if self.prefix_len == 0 {
            self.words.len()
        } else {
            (self.prefix_len as usize).div_ceil(u64::BITS as usize)
        }
    }

    #[inline]
    fn word(&self, index: usize) -> u64 {
        if self.prefix_len == 0 {
            return self.words.get(index).copied().unwrap_or(0);
        }

        let full_words = (self.prefix_len >> 6) as usize;
        if index < full_words {
            u64::MAX
        } else if index == full_words && self.prefix_len & 63 != 0 {
            (1_u64 << (self.prefix_len & 63)) - 1
        } else {
            0
        }
    }

    fn materialize_prefix(&mut self) {
        if self.prefix_len == 0 {
            return;
        }
        debug_assert!(self.words.is_empty());
        let full_words = (self.prefix_len >> 6) as usize;
        self.words.resize(full_words, u64::MAX);
        let trailing_bits = self.prefix_len & 63;
        if trailing_bits != 0 {
            self.words.push((1_u64 << trailing_bits) - 1);
        }
        self.prefix_len = 0;
    }
}

#[inline]
fn intersection_len(left: &[u64], right: &[u64]) -> u32 {
    left.chunks(1024)
        .zip(right.chunks(1024))
        .map(|(left, right)| {
            left.iter()
                .zip(right)
                .map(|(left, right)| u64::from((left & right).count_ones()))
                .sum::<u64>()
        })
        .sum::<u64>() as u32
}

fn intersection_len_indexed(
    left: &MumblingBitmapV1Builder,
    right: &MumblingBitmapV1Builder,
) -> u32 {
    (0..left.effective_word_len().min(right.effective_word_len()))
        .map(|index| (left.word(index) & right.word(index)).count_ones())
        .sum()
}

impl MumblingBitmapV1SerializationPlan {
    /// Consumes the snapshot and transfers its wire bytes without copying.
    pub fn into_bytes(self) -> Vec<u8> {
        self.encoded
    }

    /// Returns the exact serialized length in constant time.
    #[inline]
    pub fn serialized_size(&self) -> usize {
        self.encoded.len()
    }

    /// Returns the cached wire image without copying.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.encoded
    }

    /// Copies the cached wire image into a newly allocated vector.
    pub fn serialize(&self) -> Vec<u8> {
        self.encoded.clone()
    }

    /// Appends the cached wire image to `output`.
    pub fn serialize_into(&self, output: &mut Vec<u8>) {
        output.extend_from_slice(&self.encoded);
    }
}

impl<const N: usize> TryFrom<[u32; N]> for MumblingBitmapV1Builder {
    type Error = PositionOutOfRange;

    fn try_from(positions: [u32; N]) -> Result<Self, Self::Error> {
        Self::try_from_iter(positions)
    }
}

impl TryFrom<&[u32]> for MumblingBitmapV1Builder {
    type Error = PositionOutOfRange;

    fn try_from(positions: &[u32]) -> Result<Self, Self::Error> {
        Self::try_from_iter(positions.iter().copied())
    }
}

impl BitOrAssign<&Self> for MumblingBitmapV1Builder {
    fn bitor_assign(&mut self, other: &Self) {
        self.materialize_prefix();
        if other.prefix_len != 0 {
            let mut materialized = other.clone();
            materialized.materialize_prefix();
            *self |= &materialized;
            return;
        }
        if self.words.len() < other.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        let mut cardinality = 0;
        for (left, right) in self.words.iter_mut().zip(&other.words) {
            *left |= right;
            cardinality += left.count_ones();
        }
        cardinality += self.words[other.words.len()..]
            .iter()
            .map(|word| word.count_ones())
            .sum::<u32>();
        self.cardinality = cardinality;
    }
}

impl BitAndAssign<&Self> for MumblingBitmapV1Builder {
    fn bitand_assign(&mut self, other: &Self) {
        self.materialize_prefix();
        if other.prefix_len != 0 {
            let mut materialized = other.clone();
            materialized.materialize_prefix();
            *self &= &materialized;
            return;
        }
        self.words.truncate(other.words.len());
        let mut cardinality = 0;
        for (left, right) in self.words.iter_mut().zip(&other.words) {
            *left &= right;
            cardinality += left.count_ones();
        }
        trim_trailing_zeros(&mut self.words);
        self.cardinality = cardinality;
    }
}

impl SubAssign<&Self> for MumblingBitmapV1Builder {
    fn sub_assign(&mut self, other: &Self) {
        self.materialize_prefix();
        if other.prefix_len != 0 {
            let mut materialized = other.clone();
            materialized.materialize_prefix();
            *self -= &materialized;
            return;
        }
        let mut cardinality = 0;
        for (left, right) in self.words.iter_mut().zip(&other.words) {
            *left &= !right;
            cardinality += left.count_ones();
        }
        cardinality += self.words[other.words.len().min(self.words.len())..]
            .iter()
            .map(|word| word.count_ones())
            .sum::<u32>();
        trim_trailing_zeros(&mut self.words);
        self.cardinality = cardinality;
    }
}

impl BitXorAssign<&Self> for MumblingBitmapV1Builder {
    fn bitxor_assign(&mut self, other: &Self) {
        self.materialize_prefix();
        if other.prefix_len != 0 {
            let mut materialized = other.clone();
            materialized.materialize_prefix();
            *self ^= &materialized;
            return;
        }
        if self.words.len() < other.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        let mut cardinality = 0;
        for (left, right) in self.words.iter_mut().zip(&other.words) {
            *left ^= right;
            cardinality += left.count_ones();
        }
        cardinality += self.words[other.words.len()..]
            .iter()
            .map(|word| word.count_ones())
            .sum::<u32>();
        trim_trailing_zeros(&mut self.words);
        self.cardinality = cardinality;
    }
}

macro_rules! impl_owned_set_op {
    ($trait:ident, $method:ident, $assign:tt) => {
        impl<'a, 'b> $trait<&'b MumblingBitmapV1Builder> for &'a MumblingBitmapV1Builder {
            type Output = MumblingBitmapV1Builder;

            fn $method(self, other: &'b MumblingBitmapV1Builder) -> Self::Output {
                let mut result = self.clone();
                result $assign other;
                result
            }
        }
    };
}

impl_owned_set_op!(BitOr, bitor, |=);
impl_owned_set_op!(BitAnd, bitand, &=);
impl_owned_set_op!(Sub, sub, -=);
impl_owned_set_op!(BitXor, bitxor, ^=);

impl PartialEq for MumblingBitmapV1Builder {
    fn eq(&self, other: &Self) -> bool {
        if self.cardinality != other.cardinality {
            return false;
        }
        if self.prefix_len == 0 && other.prefix_len == 0 {
            return self.words == other.words;
        }
        let word_len = self.effective_word_len().max(other.effective_word_len());
        (0..word_len).all(|index| self.word(index) == other.word(index))
    }
}

impl Eq for MumblingBitmapV1Builder {}

#[inline]
fn validate_position(position: u32) -> Result<(), PositionOutOfRange> {
    if position < MAX_VALUES {
        Ok(())
    } else {
        Err(PositionOutOfRange { position })
    }
}

#[inline]
fn validate_range(range: &Range<u32>) -> Result<(), RangeOutOfRange> {
    if range.start <= range.end && range.end <= MAX_VALUES {
        Ok(())
    } else {
        Err(RangeOutOfRange {
            start: range.start,
            end: range.end,
        })
    }
}

#[inline]
fn range_mask(word_index: usize, range: &Range<u32>) -> u64 {
    let word_start = word_index as u32 * u64::BITS;
    let low = range.start.saturating_sub(word_start).min(u64::BITS);
    let high = range.end.saturating_sub(word_start).min(u64::BITS);
    let below_high = if high == u64::BITS {
        u64::MAX
    } else {
        (1_u64 << high) - 1
    };
    below_high & (u64::MAX << low)
}

fn trim_trailing_zeros(words: &mut Vec<u64>) {
    while words.last() == Some(&0) {
        words.pop();
    }
}
