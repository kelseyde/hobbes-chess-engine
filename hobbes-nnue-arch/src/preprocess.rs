use crate::{Network, UntransposedNetwork, L1_SIZE, L2_SIZE, L3_SIZE, OUTPUT_BUCKET_COUNT};

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

/// Convert an `UntransposedNetwork` (the output format from Bullet) into a `Network` (the optimal
/// format for inference).
///
/// This performs the following transformations:
/// 1. Repermutes L0 weights and biases so the most-activated neurons come first
/// 2. Permutes the L0 weights and biases to cancel out the cross-lane behaviour of packus.
/// 3. Transposes L1 weights: src[input][bucket][output] -> dst[bucket][output][input]
/// 4. Reorders L2 weights: src[input][bucket][output] -> dst[bucket][input][output]
/// 5. Reorders L3 weights: src[input][bucket] -> dst[bucket][input]
pub fn process_network(src: &UntransposedNetwork, dst: &mut Network) {
    let repermute = compute_repermute_indices();

    repermute_l0_biases(&mut dst.l0_biases, &src.l0_biases, &repermute);

    for (dst_bucket, src_bucket) in dst.l0_psq_weights.iter_mut().zip(src.l0_psq_weights.iter()) {
        repermute_l0_weights(dst_bucket, src_bucket, &repermute);
    }

    repermute_l0_weights(&mut dst.l0_threat_weights, &src.l0_threat_weights, &repermute);

    let config = permute_config();
    if config.needs_permuting {
        let order = config.order;

        let chunk_size: usize = 8;
        let block_size = order.len() * chunk_size;

        for bucket in dst.l0_psq_weights.iter_mut() {
            permute_l0_weights(bucket, order, chunk_size, block_size);
        }
        permute_l0_weights(&mut dst.l0_biases, order, chunk_size, block_size);
        permute_l0_weights(&mut dst.l0_threat_weights, order, chunk_size, block_size);
    }

    for bucket in 0..OUTPUT_BUCKET_COUNT {
        for (tgt_input_idx, &src_input_idx) in repermute.iter().enumerate() {
            for half in 0..2 {
                let tgt_idx = tgt_input_idx + half * (L1_SIZE / 2);
                let src_idx = src_input_idx + half * (L1_SIZE / 2);
                let in_block = tgt_idx / 4;
                let k = tgt_idx % 4;
                for output_idx in 0..L2_SIZE {
                    dst.l1_weights[bucket][in_block][output_idx * 4 + k] =
                        src.l1_weights[src_idx][bucket][output_idx];
                }
            }
        }
    }

    for input_idx in 0..(L2_SIZE * 2) {
        for bucket in 0..OUTPUT_BUCKET_COUNT {
            for output_idx in 0..L3_SIZE {
                dst.l2_weights[bucket][input_idx][output_idx] =
                    src.l2_weights[input_idx][bucket][output_idx];
            }
        }
    }

    for input_idx in 0..L3_SIZE {
        for bucket in 0..OUTPUT_BUCKET_COUNT {
            dst.l3_weights[bucket][input_idx] = src.l3_weights[input_idx][bucket][0];
        }
    }

    unsafe {
        std::ptr::copy_nonoverlapping(&src.l1_biases, &mut dst.l1_biases, 1);
        std::ptr::copy_nonoverlapping(&src.l2_biases, &mut dst.l2_biases, 1);
        std::ptr::copy_nonoverlapping(&src.l3_biases, &mut dst.l3_biases, 1);
    }
}

/// Compute the repermutation indices for sparsity optimisation.
///
/// Currently the identity permutation: the previous activation-frequency ordering was gathered
/// on the old 1536-wide PSQ-only net and is meaningless for the new architecture. Re-gather
/// stats (feature = "track_l0_activations") on a trained threat net before re-enabling.
fn compute_repermute_indices() -> [usize; L1_SIZE / 2] {
    std::array::from_fn(|i| i)
}

/// Re-permute the L0 biases so that the most-activated neurons come first.
fn repermute_l0_biases(
    dst: &mut [i16; L1_SIZE],
    src: &[i16; L1_SIZE],
    indices: &[usize; L1_SIZE / 2],
) {
    for (tgt, &src_idx) in indices.iter().enumerate() {
        dst[tgt] = src[src_idx];
        dst[tgt + L1_SIZE / 2] = src[src_idx + L1_SIZE / 2];
    }
}

/// Re-permute a flat `[feature][neuron]` L0 weight block so the most-activated neurons come
/// first. Generic over the element type, since the permutation is over neuron indices, not
/// storage width: this handles both i16 PSQ weights and i8 threat weights.
fn repermute_l0_weights<T: Copy>(dst: &mut [T], src: &[T], indices: &[usize; L1_SIZE / 2]) {
    debug_assert_eq!(dst.len(), src.len());
    debug_assert_eq!(src.len() % L1_SIZE, 0);
    let input_features = src.len() / L1_SIZE;
    for feature in 0..input_features {
        let base = feature * L1_SIZE;
        for (tgt, &src_idx) in indices.iter().enumerate() {
            dst[base + tgt] = src[base + src_idx];
            dst[base + tgt + L1_SIZE / 2] = src[base + src_idx + L1_SIZE / 2];
        }
    }
}

/// Permute neuron chunks in-place to cancel out the cross-lane behaviour of packus. Generic
/// over the element type for the same reason as `repermute_l0_weights`.
fn permute_l0_weights<T: Copy + Default>(
    data: &mut [T],
    order: &[u8],
    chunk_size: usize,
    block_size: usize,
) {
    debug_assert_eq!(data.len() % block_size, 0);
    let mut temp = vec![T::default(); block_size];
    for block_start in (0..data.len()).step_by(block_size) {
        temp.copy_from_slice(&data[block_start..block_start + block_size]);
        for (dst_chunk, &src_chunk) in order.iter().enumerate() {
            let dst_offset = block_start + dst_chunk * chunk_size;
            let src_offset = src_chunk as usize * chunk_size;
            data[dst_offset..dst_offset + chunk_size]
                .copy_from_slice(&temp[src_offset..src_offset + chunk_size]);
        }
    }
}