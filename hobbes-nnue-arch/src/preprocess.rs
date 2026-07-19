use crate::{Network, UntransposedNetwork, L1_SIZE, L2_SIZE, L3_SIZE, OUTPUT_BUCKET_COUNT};

pub struct PermuteConfig {
    pub needs_permuting: bool,
    pub order: &'static [u8],
}

const L0_ACTIVATIONS: [usize; L1_SIZE / 2] = [
    25688, 1485014, 688414, 808204, 62542, 1858572, 300764, 287891, 726158, 88207, 134305, 400189,
    349777, 212903, 648735, 1038855, 484549, 650436, 981034, 1677349, 133668, 238434, 283692,
    1211807, 530911, 594391, 168294, 295412, 702047, 698590, 1172497, 415057, 250340, 22762, 184764,
    173870, 289911, 564309, 627778, 271119, 237260, 1144191, 394853, 89754, 553689, 658525, 190456,
    334705, 142751, 91875, 268020, 1345545, 175232, 394135, 312692, 135333, 672407, 532545, 490911,
    872870, 789166, 64290, 21296, 1184838, 335818, 164233, 80212, 214583, 929313, 290409, 6656,
    222936, 328473, 488794, 149568, 1100298, 431168, 235723, 237099, 454098, 524631, 1255513,
    416561, 71380, 303368, 142404, 236174, 500688, 42031, 541154, 848210, 246738, 155078, 205080,
    413253, 358659, 1099380, 236859, 330017, 512911, 635344, 646822, 1787848, 301281, 228753,
    388052, 896188, 145931, 529056, 361287, 263290, 984593, 538808, 624958, 378129, 229821, 130149,
    284244, 137799, 49456, 403392, 647561, 118942, 137200, 1046402, 257827, 53226, 304471, 814526,
    899694, 378831, 56090, 846168, 679503, 290079, 316733, 33433, 122458, 273310, 261393, 101000,
    135191, 18434, 401065, 209544, 1180907, 457734, 692001, 135077, 308626, 257221, 248504, 595437,
    1444683, 399847, 1041554, 306522, 255382, 61766, 634395, 119372, 1085367, 1102623, 99440,
    136878, 181654, 118499, 249378, 1804314, 314800, 348885, 636449, 861809, 125560, 369259,
    1625328, 71197, 343473, 36174, 81951, 264483, 1090507, 516217, 1783055, 153075, 849614, 1666288,
    86864, 310841, 1168251, 892929, 458601, 648967, 486512, 1026561, 572559, 162095, 227875, 34406,
    1265661, 311109, 1307996, 169412, 331077, 959703, 149612, 322077, 162301, 468210, 60985, 170029,
    409872, 325240, 24440, 739339, 1217507, 887698, 92524, 268934, 409362, 298128, 83959, 229961,
    854685, 445907, 643945, 277379, 560659, 511212, 105621, 1325794, 256052, 496484, 326873, 615885,
    240269, 1228686, 174925, 400269, 1735725, 20485, 593908, 442844, 11930, 157705, 44058, 376478,
    819288, 138054, 614036, 64593, 399017, 178693, 226482, 73400, 377601
];

#[cfg(target_feature = "avx512f")]
static ORDER: &[u8] = &[0, 2, 4, 6, 1, 3, 5, 7];

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
static ORDER: &[u8] = &[0, 2, 1, 3];

#[cfg(not(any(target_feature = "avx512f", target_feature = "avx2")))]
static ORDER: &[u8] = &[];

pub const fn permute_config() -> PermuteConfig {
    let needs_permuting = cfg!(target_feature = "avx512f") || cfg!(target_feature = "avx2");
    PermuteConfig {
        needs_permuting,
        order: ORDER,
    }
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
        let num_chunks = order.len();

        let chunk_size: usize = 8; // 128 bits = 8 i16 values
        let block_size = num_chunks * chunk_size;

        // Permute L0 piece-square weights per bucket.
        for bucket in dst.l0_psq_weights.iter_mut() {
            permute(bucket, order, chunk_size, block_size);
        }
        // Permute L0 threat weights.
        permute(&mut dst.l0_threat_weights, order, chunk_size, block_size);

        // Permute L0 biases.
        permute(&mut dst.l0_biases, order, chunk_size, block_size);
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
fn compute_repermute_indices() -> [usize; L1_SIZE / 2] {
    let mut indices: [usize; L1_SIZE / 2] = std::array::from_fn(|i| i);
    indices.sort_by(|&a, &b| L0_ACTIVATIONS[b].cmp(&L0_ACTIVATIONS[a]));
    indices
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

/// Re-permute a single L0 weight bucket for sparsity.
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

// Permute a flat slice of values in-place
fn permute<T: Copy + Default>(
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