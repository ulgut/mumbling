//! Runtime dispatch for bulk validation. No `std`, allocation, or global
//! target-cpu flags are needed; feature-specific functions cannot be called
//! before checking both CPUID and the OS-managed XCR0 register state.

use core::arch::x86_64::*;
use core::sync::atomic::{AtomicU8, Ordering};

use crate::Error;

const SSE2: u8 = 1;
const AVX2: u8 = 2;
const AVX512: u8 = 3;
static BACKEND: AtomicU8 = AtomicU8::new(0);

fn backend() -> u8 {
    let cached = BACKEND.load(Ordering::Relaxed);
    if cached != 0 {
        return cached;
    }
    let selected = detect();
    // The byte is the entire state; racing initializers compute the same value.
    // No other memory is published, so relaxed ordering suffices.
    BACKEND.store(selected, Ordering::Relaxed);
    selected
}

fn detect() -> u8 {
    // SAFETY: CPUID is available on every x86-64 CPU. XGETBV is executed only
    // when CPUID reports XSAVE, OSXSAVE, and AVX; otherwise return SSE2.
    unsafe {
        if __cpuid(0).eax < 7 {
            return SSE2;
        }
        let required = (1 << 26) | (1 << 27) | (1 << 28);
        if __cpuid(1).ecx & required != required {
            return SSE2;
        }
        select(__cpuid_count(7, 0).ebx, _xgetbv(0))
    }
}

/// Pure feature selection, separated so missing OS state is testable everywhere
/// an x86 build runs. AVX-512 also requires AVX2 for its 32-byte tail kernel.
fn select(features: u32, xcr0: u64) -> u8 {
    if xcr0 & 0x6 != 0x6 || features & (1 << 5) == 0 {
        return SSE2;
    }
    let avx512 = (1 << 16) | (1 << 30); // AVX512F + AVX512BW
    if xcr0 & 0xe6 == 0xe6 && features & avx512 == avx512 {
        AVX512
    } else {
        AVX2
    }
}

pub(super) fn validate_dense(payload: &[u8]) -> Result<u32, Error> {
    assert_eq!(payload.len() % 32, 0);
    // SAFETY: backend() verifies CPU features and OS state for each target.
    // Kernels load only complete containers; input may be unaligned.
    unsafe {
        match backend() {
            AVX512 => dense_avx512(payload),
            AVX2 => dense_avx2(payload),
            _ => super::validate_uniform_dense_sse2(payload),
        }
    }
}

pub(super) fn validate_sparse(payload: &[u8], width: usize) -> bool {
    assert!(width <= 31);
    assert!(if width == 0 {
        payload.is_empty()
    } else {
        payload.len() % width == 0
    });
    if width < 17 {
        return super::validate_uniform_sparse_sse2(payload, width);
    }
    // SAFETY: CPU/OS features are verified; each container has 17..=31 bytes.
    unsafe {
        match backend() {
            AVX512 => sparse_avx512(payload, width),
            AVX2 => sparse_avx2(payload, width),
            _ => super::validate_uniform_sparse_sse2(payload, width),
        }
    }
}

/// Counts each nibble with a lane-local lookup, then sums each 8-byte group.
///
/// # Safety
/// Requires AVX2 with OS YMM state enabled. `payload` must contain complete
/// 32-byte containers, with at most 8,192 containers (the v1 universe).
#[target_feature(enable = "avx2")]
unsafe fn dense_avx2(payload: &[u8]) -> Result<u32, Error> {
    let table = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    ));
    let mask = _mm256_set1_epi8(15);
    let mut total = 0;
    for bytes in payload.chunks_exact(32) {
        // SAFETY: this chunk provides exactly 32 readable bytes.
        let value = unsafe { _mm256_loadu_si256(bytes.as_ptr().cast()) };
        let low = _mm256_shuffle_epi8(table, _mm256_and_si256(value, mask));
        let high =
            _mm256_shuffle_epi8(table, _mm256_and_si256(_mm256_srli_epi16::<4>(value), mask));
        let sums = _mm256_sad_epu8(_mm256_add_epi8(low, high), _mm256_setzero_si256());
        let mut lanes = [0_u64; 4];
        // SAFETY: lanes provides 32 writable bytes; unaligned stores are valid.
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), sums) };
        let count = lanes.iter().sum::<u64>() as u32;
        if count < 32 {
            return Err(Error::UnderfullDenseContainer);
        }
        total += count;
    }
    Ok(total)
}

/// Processes two adjacent containers per vector while validating their counts
/// independently. Combining counts would incorrectly accept underfull neighbors.
///
/// # Safety
/// Requires AVX2, AVX512F/BW and OS YMM/ZMM/opmask state. `payload` has the same
/// framing and maximum length requirements as `dense_avx2`.
#[target_feature(enable = "avx2,avx512f,avx512bw")]
unsafe fn dense_avx512(payload: &[u8]) -> Result<u32, Error> {
    let table = _mm512_broadcast_i32x4(_mm_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    ));
    let mask = _mm512_set1_epi8(15);
    let mut total = 0;
    let mut chunks = payload.chunks_exact(64);
    for bytes in &mut chunks {
        // SAFETY: the pair provides exactly 64 readable bytes.
        let value = unsafe { _mm512_loadu_si512(bytes.as_ptr().cast()) };
        let low = _mm512_shuffle_epi8(table, _mm512_and_si512(value, mask));
        let high =
            _mm512_shuffle_epi8(table, _mm512_and_si512(_mm512_srli_epi16::<4>(value), mask));
        let sums = _mm512_sad_epu8(_mm512_add_epi8(low, high), _mm512_setzero_si512());
        let mut lanes = [0_u64; 8];
        // SAFETY: lanes provides 64 writable bytes; unaligned stores are valid.
        unsafe { _mm512_storeu_si512(lanes.as_mut_ptr().cast(), sums) };
        for container in lanes.chunks_exact(4) {
            let count = container.iter().sum::<u64>() as u32;
            if count < 32 {
                return Err(Error::UnderfullDenseContainer);
            }
            total += count;
        }
    }
    // SAFETY: the tail is empty or one complete 32-byte container; AVX2 is enabled.
    Ok(total + unsafe { dense_avx2(chunks.remainder()) }?)
}

/// Compares overlapping first/last groups of adjacent bytes in two 128-bit
/// halves. XOR maps unsigned ordering into the signed AVX2 byte comparison.
///
/// # Safety
/// Requires AVX2 with OS YMM state. `width` is 17..=31 and divides payload length.
#[target_feature(enable = "avx2")]
unsafe fn sparse_avx2(payload: &[u8], width: usize) -> bool {
    let sign = _mm256_set1_epi8(i8::MIN);
    for bytes in payload.chunks_exact(width) {
        // SAFETY: all four 16-byte loads lie within this 17..=31-byte container.
        let (left, right) = unsafe {
            let ptr = bytes.as_ptr();
            (
                _mm256_set_m128i(
                    _mm_loadu_si128(ptr.add(width - 17).cast()),
                    _mm_loadu_si128(ptr.cast()),
                ),
                _mm256_set_m128i(
                    _mm_loadu_si128(ptr.add(width - 16).cast()),
                    _mm_loadu_si128(ptr.add(1).cast()),
                ),
            )
        };
        let ordered =
            _mm256_cmpgt_epi8(_mm256_xor_si256(right, sign), _mm256_xor_si256(left, sign));
        if _mm256_movemask_epi8(ordered) != -1 {
            return false;
        }
    }
    true
}

/// Masked loads read exactly the adjacent pairs, without touching bytes beyond
/// the container, including at allocation/page boundaries.
///
/// # Safety
/// Requires AVX512F/BW with OS ZMM/opmask state. `width` is 17..=31 and divides
/// payload length. Inactive lanes are excluded from the comparison result.
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn sparse_avx512(payload: &[u8], width: usize) -> bool {
    let mask = (1_u64 << (width - 1)) - 1;
    for bytes in payload.chunks_exact(width) {
        // SAFETY: only width-1 bytes are enabled at ptr and ptr+1; both ranges
        // are within the current container. Masked-off bytes are not accessed.
        let (left, right) = unsafe {
            (
                _mm512_maskz_loadu_epi8(mask, bytes.as_ptr().cast()),
                _mm512_maskz_loadu_epi8(mask, bytes.as_ptr().add(1).cast()),
            )
        };
        if _mm512_mask_cmplt_epu8_mask(mask, left, right) != mask {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn detection_requires_cpu_and_os_support() {
        let avx2 = 1 << 5;
        let all = avx2 | (1 << 16) | (1 << 30);
        assert_eq!(select(0, 0xe6), SSE2);
        for state in [0, 2, 4] {
            assert_eq!(select(all, state), SSE2);
        }
        assert_eq!(select(avx2, 6), AVX2);
        for state in [6, 0x26, 0x66, 0xa6, 0xc6] {
            assert_eq!(select(all, state), AVX2);
        }
        assert_eq!(select(all, 0xe6), AVX512);
    }

    #[test]
    fn supported_kernels_match_scalar() {
        let mut dense_kernels: alloc::vec::Vec<unsafe fn(&[u8]) -> Result<u32, Error>> =
            vec![super::super::validate_uniform_dense_sse2];
        let mut sparse_kernels: alloc::vec::Vec<unsafe fn(&[u8], usize) -> bool> =
            vec![super::super::validate_uniform_sparse_sse2];
        if std::is_x86_feature_detected!("avx2") {
            dense_kernels.push(dense_avx2);
            sparse_kernels.push(sparse_avx2);
        }
        if std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
        {
            dense_kernels.push(dense_avx512);
            sparse_kernels.push(sparse_avx512);
        }
        std::eprintln!(
            "checking {} x86 kernel tiers (SSE2, then supported AVX2/AVX512)",
            dense_kernels.len()
        );
        for byte in 0..=255_u8 {
            let bytes = [byte; 96];
            let count = byte.count_ones() * 32;
            let expected = if count < 32 {
                Err(Error::UnderfullDenseContainer)
            } else {
                Ok(count * 3)
            };
            for kernel in &dense_kernels {
                // SAFETY: supported CPU, three complete containers.
                assert_eq!(unsafe { kernel(&bytes) }, expected);
            }
        }
        for count in [0, 1, 2, 3, 7] {
            for offset in 0..64 {
                let mut bytes = vec![0xff; offset + count * 32];
                for kernel in &dense_kernels {
                    // SAFETY: only supported kernels were selected above.
                    assert_eq!(unsafe { kernel(&bytes[offset..]) }, Ok(count as u32 * 256));
                }
                for bad in 0..count {
                    bytes[offset + bad * 32..offset + (bad + 1) * 32].fill(0);
                    for kernel in &dense_kernels {
                        // SAFETY: same selected CPU and complete-container framing.
                        assert_eq!(
                            unsafe { kernel(&bytes[offset..]) },
                            Err(Error::UnderfullDenseContainer)
                        );
                    }
                    bytes[offset + bad * 32..offset + (bad + 1) * 32].fill(0xff);
                }
            }
        }
        for width in 17..=31 {
            for start in 0..=256 - width {
                let mut bytes: alloc::vec::Vec<u8> =
                    (start..start + width).map(|v| v as u8).collect();
                for kernel in &sparse_kernels {
                    // SAFETY: supported CPU, one container of the required width.
                    assert!(unsafe { kernel(&bytes, width) });
                }
                for bad in 1..width {
                    let saved = bytes[bad];
                    bytes[bad] = bytes[bad - 1];
                    for kernel in &sparse_kernels {
                        // SAFETY: malformed ordering does not change load bounds.
                        assert!(!unsafe { kernel(&bytes, width) });
                    }
                    bytes[bad] = saved;
                }
            }
        }
    }
}
