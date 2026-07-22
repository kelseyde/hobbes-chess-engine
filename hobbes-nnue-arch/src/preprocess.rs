use crate::{Network, UntransposedNetwork, L1_SIZE, L2_SIZE, L3_SIZE, OUTPUT_BUCKET_COUNT};

pub struct PermuteConfig {
    pub needs_permuting: bool,
    pub order: &'static [u8],
}

const L0_ACTIVATIONS: [usize; L1_SIZE / 2] = [
    60704, 118198, 86608, 602965, 334108, 426000, 324719, 400546, 126198, 342789, 81443, 691180,
    562228, 595090, 195875, 1005756, 47125, 339120, 716917, 749892, 508054, 387602, 401046, 762891,
    484907, 249546, 479713, 1894748, 112501, 226137, 22924, 148211, 118435, 1675835, 365247, 231514,
    360180, 680071, 349656, 141426, 443956, 177705, 189533, 115914, 342598, 10566, 32492, 555506,
    311958, 19379, 169879, 102144, 250259, 573697, 85180, 759422, 673064, 175729, 79233, 1855706,
    202706, 683755, 90126, 514270, 245164, 178113, 204371, 252192, 31854, 414833, 127061, 550136,
    346061, 42213, 1033071, 193293, 250396, 163090, 754281, 1093946, 219304, 60133, 270657, 420234,
    163184, 401652, 567296, 147032, 613298, 25243, 766160, 176864, 246647, 36823, 419413, 466316,
    216918, 407047, 83909, 336925, 240272, 217883, 85196, 81798, 318871, 137089, 180235, 431676,
    906556, 488713, 165713, 390236, 407330, 131643, 358347, 286706, 17432, 343016, 299465, 75241,
    379783, 326384, 876834, 327133, 198743, 1854796, 648113, 123162, 1211244, 102724, 511258,
    314952, 896549, 789642, 570374, 1361844, 116792, 992529, 192599, 91190, 1484612, 297038, 31476,
    366962, 617145, 63998, 84541, 357851, 722537, 1167088, 176648, 372538, 172218, 474232, 442390,
    178513, 288843, 169182, 61283, 257621, 956631, 334954, 290551, 404279, 209436, 417636, 80087,
    183457, 351935, 444762, 138707, 1178963, 490987, 160644, 286211, 323303, 262810, 204439, 117792,
    126482, 141560, 342317, 474201, 77482, 341920, 669977, 649522, 471123, 644761, 93451, 213714,
    867813, 75948, 91110, 270379, 13680, 230904, 359597, 132822, 488605, 78063, 353278, 422336,
    893058, 104006, 52994, 161018, 63793, 1042401, 108009, 334616, 1548434, 3860, 1095183, 808444,
    326411, 554964, 83555, 38375, 282852, 561391, 1793986, 124036, 47357, 598546, 205007, 142744,
    280100, 223696, 283553, 825174, 293905, 142648, 318550, 640119, 126055, 367458, 1108332,
    1208506, 630093, 261903, 568005, 295637, 143757, 320705, 149732, 274198, 120970, 427287, 34422,
    312881, 156693, 191317, 223974, 278120, 53530, 47027, 330638, 124022, 1266010, 220831, 169947,
    25016, 411635, 265150, 423213, 120272, 227392, 596048, 341538, 186494, 575077, 87005, 514988,
    136596, 42424, 906898, 62116, 85574, 129034, 268261, 103164, 412015, 85791, 208457, 1196705,
    323320, 1034216, 528878, 342050, 198512, 89026, 123380, 198211, 876412, 593711, 636733, 111178,
    463691, 632297, 118229, 92916, 1750171, 760671, 18229, 85204, 130948, 1220672, 1286058, 1300092,
    15188, 447040, 890742, 259164, 666036, 778250, 211511, 197132, 67985, 299269, 106482, 134514,
    389020, 607122, 1046408, 359952, 194899, 350040, 30936, 501872, 1083178, 1751505, 388608,
    502994, 24366, 332391, 564442, 127659, 37932, 48557, 95497, 64597, 562391, 486789, 201256,
    1203644, 274602, 1646526, 1119497, 98684, 120863, 232759, 26627, 110632, 183662, 350917, 370810,
    235845, 288739, 570184, 484815, 19664, 1362688, 171107, 969488, 72355, 69637, 76852, 253369,
    441351, 81883, 276014, 715071, 572925, 389857, 179696, 1909568, 584877, 1533089, 46583, 251905,
    1354057, 1188859, 523002

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
