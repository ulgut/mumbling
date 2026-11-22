//! Internal lookup and validation kernels for version-1 containers.
//!
//! AArch64 uses NEON. x86-64 bulk validation selects AVX-512BW, AVX2, or SSE2
//! once per payload, checking CPU support and OS vector-register state.
//! Tiny lookups retain SSE2 to avoid dispatch on every query. Other targets
//! use scalar code. Unaligned input is supported on every path.
//! The reader validates payload framing before calling these helpers.

use crate::Error;

#[cfg(target_arch = "x86_64")]
mod x86;

#[inline]
/// Searches a validated sparse container for a local position.
///
/// The caller supplies at most 31 bytes in strictly increasing order. SIMD
/// compares the first and last 16 bytes when the length is at least 16;
/// these overlapping loads cover the entire container. Shorter containers
/// and scalar targets use binary search.
pub(super) fn contains_sparse(container: &[u8], position: u8) -> bool {
    #[cfg(target_arch = "aarch64")]
    if container.len() >= 16 {
        use core::arch::aarch64::{vceqq_u8, vdupq_n_u8, vld1q_u8, vmaxvq_u8};

        // SAFETY: both loads cover exactly 16 bytes within `container`. AArch64
        // guarantees NEON support, and these intrinsics permit unaligned loads.
        unsafe {
            let needle = vdupq_n_u8(position);
            let first = vld1q_u8(container.as_ptr());
            let last = vld1q_u8(container.as_ptr().add(container.len() - 16));
            return vmaxvq_u8(vceqq_u8(first, needle)) != 0
                || vmaxvq_u8(vceqq_u8(last, needle)) != 0;
        }
    }

    #[cfg(target_arch = "x86_64")]
    if container.len() >= 16 {
        use core::arch::x86_64::{
            __m128i, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_set1_epi8,
        };

        // SAFETY: both loads cover exactly 16 bytes within `container`.
        // SSE2 is part of the x86-64 baseline, and the loads are unaligned.
        unsafe {
            let needle = _mm_set1_epi8(position as i8);
            let first = _mm_loadu_si128(container.as_ptr().cast::<__m128i>());
            let last = _mm_loadu_si128(
                container
                    .as_ptr()
                    .add(container.len() - 16)
                    .cast::<__m128i>(),
            );
            return _mm_movemask_epi8(_mm_cmpeq_epi8(first, needle)) != 0
                || _mm_movemask_epi8(_mm_cmpeq_epi8(last, needle)) != 0;
        }
    }

    container.binary_search(&position).is_ok()
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
/// Validates dense-container occupancy and returns the total cardinality.
///
/// The caller has checked that `payload` consists of at most 8,192 complete
/// 32-byte containers. Each container must have at least 32 set bits;
/// otherwise this returns [`Error::UnderfullDenseContainer`]. Framing and
/// the header's declared total cardinality are checked by the reader.
/// NEON uses byte population counts; SSE2 uses parallel bit counting and
/// horizontal sums; the scalar fallback uses `count_ones`.
pub(super) fn validate_uniform_dense(payload: &[u8]) -> Result<u32, Error> {
    use core::arch::aarch64::{vaddlvq_u8, vcntq_u8, vld1q_u8};

    let mut cardinality = 0_u32;
    for container in payload.chunks_exact(32) {
        // SAFETY: each container has exactly 32 bytes, covering both loads.
        let count = unsafe {
            u32::from(vaddlvq_u8(vcntq_u8(vld1q_u8(container.as_ptr()))))
                + u32::from(vaddlvq_u8(vcntq_u8(vld1q_u8(container.as_ptr().add(16)))))
        };
        if count < 32 {
            return Err(Error::UnderfullDenseContainer);
        }
        cardinality += count;
    }
    Ok(cardinality)
}

#[cfg(target_arch = "x86_64")]
/// Dispatches once per payload after checking CPU and OS vector support.
pub(super) fn validate_uniform_dense(payload: &[u8]) -> Result<u32, Error> {
    x86::validate_dense(payload)
}

#[cfg(target_arch = "x86_64")]
#[inline(never)]
/// Validates dense-container occupancy and returns the total cardinality.
///
/// The caller has checked that `payload` consists of at most 8,192 complete
/// 32-byte containers. Each container must have at least 32 set bits;
/// otherwise this returns [`Error::UnderfullDenseContainer`]. Framing and
/// the header's declared total cardinality are checked by the reader.
/// NEON uses byte population counts; SSE2 uses parallel bit counting and
/// horizontal sums; the scalar fallback uses `count_ones`.
fn validate_uniform_dense_sse2(payload: &[u8]) -> Result<u32, Error> {
    let mut cardinality = 0;
    for container in payload.chunks_exact(32) {
        // SAFETY: each container has exactly 32 bytes, covering both loads.
        // SSE2 is part of the x86-64 baseline.
        let count = unsafe {
            x86_count_bits_16(container.as_ptr()) + x86_count_bits_16(container.as_ptr().add(16))
        };
        if count < 32 {
            return Err(Error::UnderfullDenseContainer);
        }
        cardinality += count;
    }
    Ok(cardinality)
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
/// Validates dense-container occupancy and returns the total cardinality.
///
/// The caller has checked that `payload` consists of at most 8,192 complete
/// 32-byte containers. Each container must have at least 32 set bits;
/// otherwise this returns [`Error::UnderfullDenseContainer`]. Framing and
/// the header's declared total cardinality are checked by the reader.
/// NEON uses byte population counts; SSE2 uses parallel bit counting and
/// horizontal sums; the scalar fallback uses `count_ones`.
pub(super) fn validate_uniform_dense(payload: &[u8]) -> Result<u32, Error> {
    let mut cardinality = 0;
    for container in payload.chunks_exact(32) {
        let count = container.iter().map(|byte| byte.count_ones()).sum::<u32>();
        if count < 32 {
            return Err(Error::UnderfullDenseContainer);
        }
        cardinality += count;
    }
    Ok(cardinality)
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
/// Checks strict ordering within every fixed-width sparse container.
///
/// The caller has checked `width <= 31` and exact payload framing:
/// for nonzero widths the length is a multiple of `width`; for zero width
/// the payload is empty. Empty and singleton containers are trivially sorted.
/// For widths 17..=31, SIMD compares the first and last 16 adjacent pairs,
/// covering every pair without comparing across container boundaries.
/// Smaller widths and scalar targets compare adjacent bytes directly.
pub(super) fn validate_uniform_sparse(payload: &[u8], width: usize) -> bool {
    use core::arch::aarch64::{vcltq_u8, vld1q_u8, vminvq_u8};

    if width <= 1 {
        return true;
    }
    if width < 17 {
        return payload
            .chunks_exact(width)
            .all(|values| values.windows(2).all(|pair| pair[0] < pair[1]));
    }

    payload.chunks_exact(width).all(|values| {
        // SAFETY: `width` is in 17..=31, so all four 16-byte loads are valid.
        unsafe {
            let first_left = vld1q_u8(values.as_ptr());
            let first_right = vld1q_u8(values.as_ptr().add(1));
            let last = width - 17;
            let last_left = vld1q_u8(values.as_ptr().add(last));
            let last_right = vld1q_u8(values.as_ptr().add(last + 1));
            vminvq_u8(vcltq_u8(first_left, first_right)) == u8::MAX
                && vminvq_u8(vcltq_u8(last_left, last_right)) == u8::MAX
        }
    })
}

#[cfg(target_arch = "x86_64")]
/// Dispatches once per payload after checking CPU and OS vector support.
pub(super) fn validate_uniform_sparse(payload: &[u8], width: usize) -> bool {
    x86::validate_sparse(payload, width)
}

#[cfg(target_arch = "x86_64")]
#[inline(never)]
/// Checks strict ordering within every fixed-width sparse container.
///
/// The caller has checked `width <= 31` and exact payload framing:
/// for nonzero widths the length is a multiple of `width`; for zero width
/// the payload is empty. Empty and singleton containers are trivially sorted.
/// For widths 17..=31, SIMD compares the first and last 16 adjacent pairs,
/// covering every pair without comparing across container boundaries.
/// Smaller widths and scalar targets compare adjacent bytes directly.
fn validate_uniform_sparse_sse2(payload: &[u8], width: usize) -> bool {
    if width <= 1 {
        return true;
    }
    if width < 17 {
        return payload
            .chunks_exact(width)
            .all(|values| values.windows(2).all(|pair| pair[0] < pair[1]));
    }

    payload.chunks_exact(width).all(|values| {
        // SAFETY: `width` is in 17..=31, so both sets of adjacent
        // comparisons address 17 readable bytes.
        unsafe {
            x86_16_pairs_increasing(values.as_ptr())
                && x86_16_pairs_increasing(values.as_ptr().add(width - 17))
        }
    })
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
/// Checks strict ordering within every fixed-width sparse container.
///
/// The caller has checked `width <= 31` and exact payload framing:
/// for nonzero widths the length is a multiple of `width`; for zero width
/// the payload is empty. Empty and singleton containers are trivially sorted.
/// For widths 17..=31, SIMD compares the first and last 16 adjacent pairs,
/// covering every pair without comparing across container boundaries.
/// Smaller widths and scalar targets compare adjacent bytes directly.
pub(super) fn validate_uniform_sparse(payload: &[u8], width: usize) -> bool {
    width <= 1
        || payload
            .chunks_exact(width)
            .all(|values| values.windows(2).all(|pair| pair[0] < pair[1]))
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
/// Checks strict unsigned-byte ordering in a 17..=31-byte sparse container.
///
/// SIMD compares two overlapping groups of 16 adjacent pairs. This covers
/// every pair, including comparisons across the unsigned sign-bit boundary.
/// The scalar fallback checks adjacent pairs directly.
///
/// # Panics
///
/// Panics unless `values.len()` is in 17..=31. This is a release-mode check
/// because the SIMD implementations use raw pointer loads.
pub(super) fn is_strictly_increasing_large(values: &[u8]) -> bool {
    use core::arch::aarch64::{vcltq_u8, vld1q_u8, vminvq_u8};

    assert!((17..=31).contains(&values.len()));
    // Compare the first and last 16 adjacent pairs. Sparse containers have
    // at most 31 values, so these ranges cover every pair and may overlap.
    // SAFETY: a length of at least 17 makes both 16-byte loads valid.
    unsafe {
        let first_left = vld1q_u8(values.as_ptr());
        let first_right = vld1q_u8(values.as_ptr().add(1));
        let last = values.len() - 17;
        let last_left = vld1q_u8(values.as_ptr().add(last));
        let last_right = vld1q_u8(values.as_ptr().add(last + 1));
        vminvq_u8(vcltq_u8(first_left, first_right)) == u8::MAX
            && vminvq_u8(vcltq_u8(last_left, last_right)) == u8::MAX
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(never)]
/// Checks strict unsigned-byte ordering in a 17..=31-byte sparse container.
///
/// SIMD compares two overlapping groups of 16 adjacent pairs. This covers
/// every pair, including comparisons across the unsigned sign-bit boundary.
/// The scalar fallback checks adjacent pairs directly.
///
/// # Panics
///
/// Panics unless `values.len()` is in 17..=31. This is a release-mode check
/// because the SIMD implementations use raw pointer loads.
pub(super) fn is_strictly_increasing_large(values: &[u8]) -> bool {
    assert!((17..=31).contains(&values.len()));
    // SAFETY: a length of at least 17 makes both adjacent 16-byte comparisons
    // valid. Together they cover every pair and may overlap.
    unsafe {
        x86_16_pairs_increasing(values.as_ptr())
            && x86_16_pairs_increasing(values.as_ptr().add(values.len() - 17))
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline]
/// Checks strict unsigned-byte ordering in a 17..=31-byte sparse container.
///
/// SIMD compares two overlapping groups of 16 adjacent pairs. This covers
/// every pair, including comparisons across the unsigned sign-bit boundary.
/// The scalar fallback checks adjacent pairs directly.
///
/// # Panics
///
/// Panics unless `values.len()` is in 17..=31. This is a release-mode check
/// because the SIMD implementations use raw pointer loads.
pub(super) fn is_strictly_increasing_large(values: &[u8]) -> bool {
    assert!((17..=31).contains(&values.len()));
    values.windows(2).all(|pair| pair[0] < pair[1])
}

#[cfg(target_arch = "x86_64")]
#[inline]
/// Counts the set bits in 16 bytes using baseline SSE2.
///
/// Successive masks accumulate counts into pairs, nibbles, and bytes;
/// `_mm_sad_epu8` then reduces them into two 64-bit lane sums.
///
/// # Safety
///
/// `bytes` must point to at least 16 initialized, readable bytes within one
/// allocation. Alignment is not required. SSE2 is baseline on x86-64.
#[target_feature(enable = "sse2")]
unsafe fn x86_count_bits_16(bytes: *const u8) -> u32 {
    use core::arch::x86_64::{
        __m128i, _mm_add_epi8, _mm_and_si128, _mm_loadu_si128, _mm_sad_epu8, _mm_set1_epi8,
        _mm_setzero_si128, _mm_srli_epi16, _mm_storeu_si128, _mm_sub_epi8,
    };

    // SAFETY: the caller guarantees that `bytes` addresses 16 readable bytes.
    let mut value = unsafe { _mm_loadu_si128(bytes.cast::<__m128i>()) };
    let pairs = _mm_and_si128(_mm_srli_epi16::<1>(value), _mm_set1_epi8(0x55));
    value = _mm_sub_epi8(value, pairs);
    let nibbles = _mm_add_epi8(
        _mm_and_si128(value, _mm_set1_epi8(0x33)),
        _mm_and_si128(_mm_srli_epi16::<2>(value), _mm_set1_epi8(0x33)),
    );
    let counts = _mm_and_si128(
        _mm_add_epi8(nibbles, _mm_srli_epi16::<4>(nibbles)),
        _mm_set1_epi8(0x0f),
    );
    let sums = _mm_sad_epu8(counts, _mm_setzero_si128());
    let mut lanes = [0_u64; 2];
    // SAFETY: `lanes` provides 16 writable bytes and unaligned stores are valid.
    unsafe { _mm_storeu_si128(lanes.as_mut_ptr().cast::<__m128i>(), sums) };
    (lanes[0] + lanes[1]) as u32
}

#[cfg(target_arch = "x86_64")]
#[inline]
/// Checks that 17 bytes are strictly increasing with 16 SSE2 comparisons.
///
/// XOR with 0x80 maps unsigned byte order to signed order. Every lane must
/// compare greater than its predecessor for the movemask to be 0xffff.
///
/// # Safety
///
/// `values` must point to at least 17 initialized, readable bytes within one
/// allocation. Alignment is not required. SSE2 is baseline on x86-64.
#[target_feature(enable = "sse2")]
unsafe fn x86_16_pairs_increasing(values: *const u8) -> bool {
    use core::arch::x86_64::{
        __m128i, _mm_cmpgt_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_set1_epi8, _mm_xor_si128,
    };

    // XOR with the sign bit maps unsigned byte order to signed byte order.
    let sign = _mm_set1_epi8(i8::MIN);
    // SAFETY: the caller guarantees 17 readable bytes from `values`.
    let left = unsafe { _mm_loadu_si128(values.cast::<__m128i>()) };
    let right = unsafe { _mm_loadu_si128(values.add(1).cast::<__m128i>()) };
    let increasing = _mm_cmpgt_epi8(_mm_xor_si128(right, sign), _mm_xor_si128(left, sign));
    _mm_movemask_epi8(increasing) == 0xffff
}
