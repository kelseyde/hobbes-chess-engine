use crate::Network;

/// Returns the permutation order needed to counteract the cross-lane behaviour
/// of the `packus` SIMD instruction.
///
/// - AVX-512: `packus` operates on 128-bit lanes within a 512-bit register,
///   producing output in order `[0, 2, 4, 6, 1, 3, 5, 7]`.
/// - AVX-2: `packus` operates on 128-bit lanes within a 256-bit register,
///   producing output in order `[0, 2, 1, 3]`.
/// - NEON: 128-bit only, no cross-lane permutation needed.
/// - Scalar: no permutation needed.

pub struct PermuteConfig {
    pub needs_permuting: bool,
    pub order: &'static [u8],
}

#[cfg(target_feature = "avx512f")]
static ORDER: &[u8] = &[0, 2, 4, 6, 1, 3, 5, 7];

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
static ORDER: &[u8] = &[0, 2, 1, 3];

#[cfg(not(any(target_feature = "avx512f", target_feature = "avx2")))]
static ORDER: &[u8] = &[];

pub const fn permute_config() -> PermuteConfig {
    let needs_permuting = cfg!(target_feature = "avx512f") || cfg!(target_feature = "avx2");
    PermuteConfig { needs_permuting, order: ORDER }
}

/// Pre-permute L0 weights and biases so that the cross-lane interleaving
/// caused by `packus` produces output in the correct order at runtime,
/// eliminating the need for a runtime `permute` fixup.
///
/// `packus` packs two SIMD registers of i16 → u8 values, but interleaves
/// 128-bit lanes. By storing the accumulator (L0 biases) and weights in the
/// permuted order, the scrambled `packus` output ends up in natural order.
///
/// The permutation reorders 64-bit (4×i16) chunks within each pair of SIMD
/// registers according to `ORDER`.
pub fn permute_network(network: &mut Network) {
    let config = permute_config();
    if !config.needs_permuting {
        return;
    }

    let order = config.order;
    let num_chunks = order.len();

    // Each chunk is 128 bits = 16 bytes = 8 i16 values (one 128-bit lane)
    let chunk_size: usize = 8;
    let block_size = num_chunks * chunk_size;

    // Permute L0 weights per bucket
    for bucket in network.l0_weights.iter_mut() {
        permute_i16s(bucket, order, chunk_size, block_size);
    }

    // Permute L0 biases
    permute_i16s(&mut network.l0_biases, order, chunk_size, block_size);
}

// Permute a flat slice of i16 values in-place
fn permute_i16s(data: &mut [i16], order: &[u8], chunk_size: usize, block_size: usize) {
    let num_chunks = order.len();
    let mut temp = vec![0i16; block_size];
    for block_start in (0..data.len()).step_by(block_size) {
        // Copy original block into temp
        temp.copy_from_slice(&data[block_start..block_start + block_size]);
        // Write chunks back in permuted order
        for dst_chunk in 0..num_chunks {
            let src_chunk = order[dst_chunk] as usize;
            let dst_offset = block_start + dst_chunk * chunk_size;
            let src_offset = src_chunk * chunk_size;
            data[dst_offset..dst_offset + chunk_size]
                .copy_from_slice(&temp[src_offset..src_offset + chunk_size]);
        }
    }
}