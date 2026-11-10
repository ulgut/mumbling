use alloc::vec::Vec;

use crate::Error;

pub(super) const VALUES_PER_CHUNK: usize = 256;

/// Encodes descriptor values using the version-1 PFOR layout.
pub(super) fn encode(values: &[u8], output: &mut Vec<u8>) {
    for chunk in values.chunks(VALUES_PER_CHUNK) {
        encode_chunk(chunk, output);
    }
}

/// Returns the exact number of bytes required to encode `values`.
pub(super) fn encoded_len(values: &[u8]) -> usize {
    values
        .chunks(VALUES_PER_CHUNK)
        .map(|chunk| 3 + encoding_for(chunk).data_size)
        .sum()
}

#[derive(Clone, Copy)]
struct Encoding {
    minimum: u8,
    primary_width: usize,
    exception_width: usize,
    exception_count: usize,
    data_size: usize,
}

fn encode_chunk(values: &[u8], output: &mut Vec<u8>) {
    let encoding = encoding_for(values);
    let Encoding {
        minimum,
        primary_width,
        exception_width,
        exception_count,
        ..
    } = encoding;

    output.push(((exception_width as u8) << 4) | primary_width as u8);
    output.push(exception_count as u8);
    output.push(minimum);

    pack(
        values
            .iter()
            .map(|&value| usize::from(value - minimum) & bit_mask(primary_width)),
        values.len(),
        primary_width,
        output,
    );

    for (offset, &value) in values.iter().enumerate() {
        if bits_required(value - minimum) > primary_width {
            output.push(offset as u8);
        }
    }
    pack(
        values
            .iter()
            .filter(|&&value| bits_required(value - minimum) > primary_width)
            .map(|&value| usize::from(value - minimum) >> primary_width),
        exception_count,
        exception_width,
        output,
    );
}

fn encoding_for(values: &[u8]) -> Encoding {
    let minimum = *values.iter().min().expect("PFOR chunks are non-empty");
    let maximum_normalized = values.iter().map(|value| value - minimum).max().unwrap();
    let maximum_width = bits_required(maximum_normalized);

    let mut best_width = 0;
    let mut best_exception_count = 0;
    let mut best_size = usize::MAX;
    for width in 0..=maximum_width {
        let exception_count = values
            .iter()
            .filter(|&&value| bits_required(value - minimum) > width)
            .count();
        if exception_count > u8::MAX as usize {
            continue;
        }
        let exception_width = maximum_width - width;
        let size = packed_len(values.len(), width)
            + exception_count
            + packed_len(exception_count, exception_width);
        if size < best_size {
            best_width = width;
            best_exception_count = exception_count;
            best_size = size;
        }
    }

    let exception_width = maximum_width - best_width;
    Encoding {
        minimum,
        primary_width: best_width,
        exception_width,
        exception_count: best_exception_count,
        data_size: best_size,
    }
}

/// Decodes exactly `value_count` descriptor values and returns the number of
/// input bytes consumed. The caller uses the consumed length to locate the
/// container payload section.
pub(super) fn decode(input: &[u8], value_count: usize) -> Result<(Vec<u8>, usize), Error> {
    let mut values = Vec::with_capacity(value_count);
    let mut cursor = 0;

    while values.len() < value_count {
        let chunk_len = (value_count - values.len()).min(VALUES_PER_CHUNK);
        let header = input.get(cursor..cursor + 3).ok_or(Error::Truncated)?;
        cursor += 3;

        let b1 = usize::from(header[0] & 0x0f);
        let b2 = usize::from(header[0] >> 4);
        let exception_count = usize::from(header[1]);
        let minimum = header[2];

        if b1 > 8
            || b2 > 8_usize.saturating_sub(b1)
            || exception_count > chunk_len
            || (b2 == 0 && exception_count != 0)
        {
            return Err(Error::InvalidPforHeader);
        }

        let primary_len = packed_len(chunk_len, b1);
        let primary = take(input, &mut cursor, primary_len)?;
        let exception_offsets = take(input, &mut cursor, exception_count)?;
        let exception_values_len = packed_len(exception_count, b2);
        let exception_values = take(input, &mut cursor, exception_values_len)?;

        let chunk_start = values.len();
        if b1 == 0 {
            values.resize(chunk_start + chunk_len, minimum);
        } else {
            for index in 0..chunk_len {
                let low = read_packed(primary, index, b1);
                let value = usize::from(minimum)
                    .checked_add(low)
                    .filter(|value| *value <= usize::from(u8::MAX))
                    .ok_or(Error::InvalidPforValue)?;
                values.push(value as u8);
            }
        }

        if exception_count != 0 {
            let mut seen = [false; VALUES_PER_CHUNK];
            for (exception_index, &offset) in exception_offsets.iter().enumerate() {
                let offset = usize::from(offset);
                if offset >= chunk_len || seen[offset] {
                    return Err(Error::InvalidPforExceptionOffset);
                }
                seen[offset] = true;

                let high = read_packed(exception_values, exception_index, b2);
                if high == 0 {
                    return Err(Error::InvalidPforValue);
                }

                let base = usize::from(values[chunk_start + offset]);
                let value = base
                    .checked_add(high << b1)
                    .filter(|value| *value <= usize::from(u8::MAX))
                    .ok_or(Error::InvalidPforValue)?;
                values[chunk_start + offset] = value as u8;
            }
        }
    }

    Ok((values, cursor))
}

#[inline]
const fn packed_len(value_count: usize, bit_width: usize) -> usize {
    (value_count * bit_width).div_ceil(8)
}

fn take<'a>(input: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8], Error> {
    let end = cursor.checked_add(len).ok_or(Error::Truncated)?;
    let bytes = input.get(*cursor..end).ok_or(Error::Truncated)?;
    *cursor = end;
    Ok(bytes)
}

/// Reads one MSB-first packed value. PFOR values are at most eight bits wide,
/// so they span at most two adjacent bytes.
#[inline]
fn read_packed(input: &[u8], index: usize, bit_width: usize) -> usize {
    if bit_width == 0 {
        return 0;
    }

    let bit_offset = index * bit_width;
    let byte_index = bit_offset >> 3;
    let offset_in_byte = bit_offset & 7;
    let first = usize::from(input[byte_index]);
    let second = input
        .get(byte_index + 1)
        .copied()
        .map(usize::from)
        .unwrap_or(0);
    let window = (first << 8) | second;
    let shift = 16 - offset_in_byte - bit_width;
    let mask = (1_usize << bit_width) - 1;
    (window >> shift) & mask
}

fn pack(
    values: impl IntoIterator<Item = usize>,
    value_count: usize,
    bit_width: usize,
    output: &mut Vec<u8>,
) {
    if bit_width == 0 {
        return;
    }

    let start = output.len();
    output.resize(start + packed_len(value_count, bit_width), 0);
    let mut bit_offset = 0;
    for (index, value) in values.into_iter().enumerate() {
        debug_assert!(index < value_count);
        debug_assert!(value <= bit_mask(bit_width));

        let byte_index = bit_offset >> 3;
        let offset_in_byte = bit_offset & 7;
        let window = value << (16 - offset_in_byte - bit_width);
        output[start + byte_index] |= (window >> 8) as u8;
        if offset_in_byte + bit_width > 8 {
            output[start + byte_index + 1] |= window as u8;
        }
        bit_offset += bit_width;
    }
}

#[inline]
fn bits_required(value: u8) -> usize {
    (u8::BITS - value.leading_zeros()) as usize
}

#[inline]
const fn bit_mask(bit_width: usize) -> usize {
    if bit_width == 0 {
        0
    } else {
        (1 << bit_width) - 1
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{decode, encode, encoded_len};
    use crate::Error;

    #[test]
    fn decodes_spec_examples() {
        assert_eq!(decode(&[0x00, 0x00, 0x00], 256).unwrap().0, [0; 256]);
        assert_eq!(decode(&[0x00, 0x00, 0x05], 51).unwrap().0, [5; 51]);
        assert_eq!(
            decode(&[0x80, 0x02, 0x00, 0x04, 0x07, 0xff, 0xfe], 8)
                .unwrap()
                .0,
            [0, 0, 0, 0, 255, 0, 0, 254]
        );
        assert_eq!(decode(&[0x02, 0x00, 0x06, 0x18], 3).unwrap().0, [6, 7, 8]);
        assert_eq!(
            decode(&[0x32, 0x01, 0x06, 0x09, 0x01, 0xe0], 4).unwrap().0,
            [6, 34, 8, 7]
        );
    }

    #[test]
    fn encodes_spec_examples() {
        assert_encoded(&[0; 256], &[0x00, 0x00, 0x00]);
        assert_encoded(&[5; 51], &[0x00, 0x00, 0x05]);
        assert_encoded(
            &[0, 0, 0, 0, 255, 0, 0, 254],
            &[0x80, 0x02, 0x00, 0x04, 0x07, 0xff, 0xfe],
        );
        assert_encoded(&[6, 7, 8], &[0x02, 0x00, 0x06, 0x18]);
        assert_encoded(&[6, 34, 8, 7], &[0x32, 0x01, 0x06, 0x09, 0x01, 0xe0]);
    }

    #[test]
    fn round_trips_arbitrary_bytes_and_chunk_lengths() {
        let mut state = 0x6a09_e667_u32;
        for length in [1, 2, 7, 8, 9, 255, 256, 257, 8_192] {
            let values = (0..length)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    state as u8
                })
                .collect::<Vec<_>>();
            let mut encoded = Vec::new();
            encode(&values, &mut encoded);
            let (decoded, consumed) = decode(&encoded, values.len()).unwrap();
            assert_eq!(decoded, values);
            assert_eq!(consumed, encoded.len());
        }
    }

    #[test]
    fn decodes_across_chunk_boundary() {
        let encoded = [
            0x00, 0x00, 0x03, // 256 values equal to 3
            0x00, 0x00, 0x04, // one value equal to 4
        ];
        let (values, consumed) = decode(&encoded, 257).unwrap();

        assert_eq!(values.len(), 257);
        assert!(values[..256].iter().all(|&value| value == 3));
        assert_eq!(values[256], 4);
        assert_eq!(consumed, encoded.len());
    }

    #[test]
    fn rejects_bad_exception_offsets() {
        let duplicate = [0x10, 0x02, 0x00, 0x00, 0x00, 0xc0];
        assert_eq!(
            decode(&duplicate, 2),
            Err(Error::InvalidPforExceptionOffset)
        );

        let outside_chunk = [0x10, 0x01, 0x00, 0x01, 0x80];
        assert_eq!(
            decode(&outside_chunk, 1),
            Err(Error::InvalidPforExceptionOffset)
        );
    }

    #[test]
    fn rejects_truncated_input() {
        assert_eq!(decode(&[0x02, 0x00, 0x06], 3), Err(Error::Truncated));
    }

    fn assert_encoded(values: &[u8], expected: &[u8]) {
        let mut encoded = Vec::new();
        encode(values, &mut encoded);
        assert_eq!(encoded, expected);
        assert_eq!(encoded_len(values), encoded.len());
        assert_eq!(decode(&encoded, values.len()).unwrap().0, values);
    }
}
