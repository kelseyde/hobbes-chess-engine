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
/// 1. Permutes the L0 weights and biases to cancel out the cross-lane behaviour of packus.
/// 2. Transposes L1 weights: src[input][bucket][output] -> dst[bucket][output][input]
/// 3. Reorders L2 weights: src[input][bucket][output] -> dst[bucket][input][output]
/// 4. Reorders L3 weights: src[input][bucket] -> dst[bucket][input]
pub fn process_network(src: &UntransposedNetwork, dst: &mut Network) {
    unsafe {
        std::ptr::copy_nonoverlapping(&src.l0_weights, &mut dst.l0_weights, 1);
        std::ptr::copy_nonoverlapping(&src.l0_biases, &mut dst.l0_biases, 1);
    }

    let config = permute_config();
    if config.needs_permuting {
        let order = config.order;
        let num_chunks = order.len();

        let chunk_size: usize = 8; // 128 bits = 8 i16 values
        let block_size = num_chunks * chunk_size;

        // Permute L0 weights per bucket.
        for bucket in dst.l0_weights.iter_mut() {
            permute_i16s(bucket, order, chunk_size, block_size);
        }
        // Permute L0 biases.
        permute_i16s(&mut dst.l0_biases, order, chunk_size, block_size);
    }

    for input_idx in 0..L1_SIZE {
        for bucket in 0..OUTPUT_BUCKET_COUNT {
            for output_idx in 0..L2_SIZE {
                dst.l1_weights[bucket][output_idx][input_idx] =
                    src.l1_weights[input_idx][bucket][output_idx];
            }
        }
    }

    for input_idx in 0..L2_SIZE {
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

// Permute a flat slice of i16 values in-place
fn permute_i16s(data: &mut [i16], order: &[u8], chunk_size: usize, block_size: usize) {
    let num_chunks = order.len();
    let mut temp = vec![0i16; block_size];
    for block_start in (0..data.len()).step_by(block_size) {
        temp.copy_from_slice(&data[block_start..block_start + block_size]);
        for dst_chunk in 0..num_chunks {
            let src_chunk = order[dst_chunk] as usize;
            let dst_offset = block_start + dst_chunk * chunk_size;
            let src_offset = src_chunk * chunk_size;
            data[dst_offset..dst_offset + chunk_size]
                .copy_from_slice(&temp[src_offset..src_offset + chunk_size]);
        }
    }
}
