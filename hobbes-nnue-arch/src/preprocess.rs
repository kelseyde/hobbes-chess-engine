use crate::{Network, UntransposedNetwork, L1_SIZE, L2_SIZE, L3_SIZE, OUTPUT_BUCKET_COUNT};

pub struct PermuteConfig {
    pub needs_permuting: bool,
    pub order: &'static [u8],
}

const L0_ACTIVATIONS: [usize; L1_SIZE / 2] = [
    2052869, 1991725, 1961990, 1973240, 1933722, 1836887, 1873310, 1785707, 1588737, 1562015,
    1504102, 1521088, 1486633, 1290849, 1388170, 1361186, 1389722, 1320506, 1346390, 1244296,
    1275605, 1312499, 1258306, 1195912, 1197406, 1164678, 1210702, 1220764, 1148847, 1169334,
    1124958, 1186095, 1037498, 1100342, 1103687, 1041234, 943621, 1035563, 1023832, 990043, 981771,
    1011224, 869790, 905738, 953769, 1059104, 875447, 907992, 855076, 745855, 788812, 793795,
    681327, 773093, 695884, 770573, 718727, 773015, 742949, 709193, 689820, 693448, 791388, 675255,
    654606, 659247, 733987, 782729, 759327, 723527, 665415, 639948, 688152, 623284, 661516, 660771,
    555564, 617621, 545541, 519256, 554780, 582490, 623620, 572052, 560178, 536166, 578759, 525906,
    519655, 504079, 499031, 472549, 554675, 501157, 491239, 506130, 441945, 502793, 430679, 547266,
    403642, 581074, 555713, 475528, 441829, 449461, 413841, 530496, 417511, 487374, 462934, 416700,
    450565, 403751, 469593, 407969, 403462, 432386, 450324, 376788, 436463, 381082, 397761, 418948,
    378139, 423097, 423524, 359443, 371216, 353433, 355524, 356751, 342588, 402318, 338790, 357855,
    302021, 341621, 302790, 284164, 315155, 339017, 327110, 303351, 288047, 322111, 357852, 298939,
    378224, 306223, 331898, 299658, 289895, 275042, 305237, 284169, 270638, 266550, 303361, 294679,
    265165, 269756, 291465, 257072, 325568, 283690, 308893, 223338, 260401, 244488, 237134, 275908,
    285871, 237960, 256551, 255893, 238924, 240774, 253212, 237988, 232050, 250113, 238721, 226631,
    270045, 237871, 153552, 203449, 182686, 210193, 173437, 168418, 193939, 166002, 172853, 207654,
    170716, 156916, 170943, 149927, 196362, 158254, 180609, 166433, 143785, 159995, 137683, 146165,
    133451, 176869, 132011, 141271, 145901, 159425, 151574, 128413, 158880, 105849, 121745, 133637,
    161820, 108549, 80108, 101696, 88074, 105397, 94084, 99443, 101198, 96906, 127495, 78591,
    113766, 80975, 75621, 78860, 64337, 63735, 61819, 83173, 67682, 51124, 51119, 42330, 45268,
    51967, 32976, 38199, 29878, 25196, 32788, 26409, 23246, 24630, 12931, 6644
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
