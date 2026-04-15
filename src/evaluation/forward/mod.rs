use hobbes_nnue_arch::{L1_SIZE, L2_SIZE, L3_SIZE};

/// Trait grouping for four forward pass functions in Hobbes' multilayer NNUE inference. There
/// is both a `Vectorised` and `Scalar` implementation of this trait; the appropriate one is
/// selected at compile time.
pub trait Forward {
    unsafe fn activate_l0(us: &[i16; L1_SIZE], them: &[i16; L1_SIZE]) -> [u8; L1_SIZE];

    unsafe fn propagate_l1(input: &[u8; L1_SIZE], output_bucket: usize) -> [i32; L2_SIZE * 2];

    unsafe fn propagate_l2(input: &[i32; L2_SIZE * 2], output_bucket: usize) -> [i32; L3_SIZE];

    unsafe fn propagate_l3(input: &[i32; L3_SIZE], output_bucket: usize) -> i32;
}

#[cfg(any(target_feature = "avx2", target_feature = "neon"))]
mod vectorised;
#[cfg(any(target_feature = "avx2", target_feature = "neon"))]
pub use vectorised::Vectorised as inference;

#[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]
mod scalar;
#[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]
pub use scalar::Scalar as inference;
