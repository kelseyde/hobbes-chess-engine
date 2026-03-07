/// Returns the permutation order needed to counteract the cross-lane behaviour
/// of the `packus` SIMD instruction.
///
/// - AVX-512: `packus` operates on 128-bit lanes within a 512-bit register,
///   producing output in lane order `[0, 2, 4, 6, 1, 3, 5, 7]`.
/// - AVX-2: `packus` operates on 128-bit lanes within a 256-bit register,
///   producing output in lane order `[0, 2, 1, 3]`.
/// - NEON: 128-bit only, no cross-lane permutation needed.
/// - Scalar: no permutation needed.

#[cfg(target_feature = "avx512f")]
pub const PERMUTE_ORDER: [u8; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

#[cfg(target_feature = "avx2")]
pub const PERMUTE_ORDER: [u8; 4] = [0, 2, 1, 3];

#[cfg(target_arch = "aarch64")]
pub const PERMUTE_ORDER: [u8; 1] = [0];

#[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_arch = "aarch64")))]
pub const PERMUTE_ORDER: [u8; 0] = [];

pub const fn permute_order() -> &'static [u8] {
    &PERMUTE_ORDER
}

/// Returns whether the network weights need to be permuted for the current
/// target architecture. Permutation is required when `packus` operates across
/// multiple 128-bit lanes (AVX2 and AVX512).
pub const fn needs_permuting() -> bool {
    if cfg!(target_feature = "avx512f") {
        return true;
    }
    if cfg!(target_feature = "avx2") {
        return true;
    }
    false
}
