use crate::{Network, UntransposedNetwork, L1_SIZE, L2_SIZE, L3_SIZE, OUTPUT_BUCKET_COUNT};

pub struct PermuteConfig {
    pub needs_permuting: bool,
    pub order: &'static [u8],
}

const L0_ACTIVATIONS: [usize; L1_SIZE / 2] = [
    173567, 271717, 318701, 271827, 355163, 437736, 746326, 2415, 159190, 11048, 34365, 42885, 414493, 53645, 118997, 198191, 171036, 117482, 190657, 89030, 758667, 28046, 112383, 372996, 34713, 151913, 131734, 251140, 379645, 37763, 47233, 303389, 214100, 239739, 133096, 477997, 225962, 111127, 32857, 108107, 249853, 155614, 25017, 558670, 493823, 432699, 71510, 174561, 284360, 175167, 7610, 267481, 11738, 189852, 101845, 128301, 274161, 74237, 154598, 315603, 26865, 183669, 390244, 101802, 15572, 22539, 196055, 175808, 97116, 257871, 74611, 12087, 213327, 192967, 35554, 294633, 100243, 290723, 35672, 207088, 102468, 300445, 39067, 41327, 6806, 270827, 346046, 273867, 46721, 91454, 172003, 206868, 40236, 352387, 187871, 37298, 47542, 290530, 45581, 210597, 245265, 235094, 133360, 301396, 510360, 53434, 37831, 128652, 208270, 217024, 210043, 18057, 304682, 60208, 145919, 281706, 142242, 265612, 180038, 376380, 265060, 236225, 367468, 128216, 39217, 140682, 131791, 336180, 314154, 214490, 82101, 185807, 81416, 134390, 268773, 51167, 55424, 470762, 117434, 127617, 991453, 319933, 313890, 125509, 109792, 145492, 160018, 102066, 253130, 182958, 319709, 619521, 73706, 93048, 140500, 6275, 7953, 141517, 84337, 215610, 93581, 439056, 272798, 135171, 116532, 484634, 9147, 177805, 267638, 272275, 507331, 13089, 1123282, 773074, 252980, 21331, 82484, 81449, 155005, 168207, 176742, 562484, 78606, 265124, 12061, 37154, 167861, 135895, 18462, 300859, 312827, 45895, 323233, 127184, 91339, 234424, 158502, 289938, 364017, 259694, 249417, 454624, 363262, 543520, 13020, 162270, 380533, 16416, 65847, 5062, 206428, 120627, 309764, 89143, 144460, 29020, 352265, 33315, 65035, 79831, 606696, 129039, 21487, 97788, 389254, 183527, 297713, 113175, 296516, 96060, 368229, 211983, 37675, 24024, 112833, 350553, 191239, 174805, 262326, 163158, 127469, 163272, 59229, 172508, 136618, 432964, 701437, 28678, 38998, 79917, 87135, 97149, 102965, 73552, 38624, 233789, 110304, 351275, 189574, 188067, 342133, 59982, 107794, 200514, 329410, 110170, 42587, 311910, 128512, 83055, 176430, 249047, 116616, 141174, 65667, 217430, 30386, 369796, 208344, 148016, 137038, 182376, 195589, 55142, 301598, 41587, 176805, 222003, 123073, 123765, 123550, 100364, 8653, 39610, 119524, 17067, 84842, 28523, 578409, 14296, 206852, 40628, 172504, 534552, 412485, 149748, 149062, 178328, 185409, 146823, 19893, 289874, 1361400, 93873, 17923, 55588, 60821, 307702, 244296, 111731, 288821, 173855, 515408, 172888, 388358, 137271, 95495, 164995, 151301, 55680, 19706, 53179, 234708, 878489, 29503, 145627, 522795, 56837, 209544, 246734, 497312, 257931, 213184, 75199, 132690, 85320, 310011, 130523, 108746, 251039, 186902, 140433, 354841, 88034, 410299, 87833, 150761, 227932, 229691, 14278, 28087, 97227, 182334, 268873, 60363, 77394, 142229, 339171, 19325, 125716, 100511, 169241, 156540, 295626, 34501, 191239, 666266, 169144, 231161, 1041530, 120976, 15235, 613045, 122285, 126315, 88284, 236956, 131440, 1204069, 183812, 87289, 55064, 196122, 502682, 310169, 19415, 2213, 165229, 160136, 210466, 253996, 179099, 223342, 395955, 212139, 30648, 371066, 35357, 232365, 219963, 230767, 22220, 37270, 137011, 253080, 28049, 95112, 53003, 3825, 164692, 205673, 29917, 46307, 207175, 15546, 278451, 122563, 213047, 106744, 128780, 17406, 125260, 244754, 206708, 324793, 301099, 385692, 116411, 22574, 60936, 32232, 231075, 236954, 34291, 115913, 254131, 13788, 297161, 138468, 252076, 2004, 384738, 123667, 66356, 228943, 473840, 189677, 367297, 200463, 215101, 51072, 527518, 60834, 208874, 39958, 8604, 111036, 3321, 118431, 229401, 282986, 231444, 18254, 238097, 283036, 124604, 118530, 370440, 228320, 107225, 37695, 336281, 56953, 76931, 44307, 248516, 139484, 65230, 194671, 24094, 39128, 147737, 332363, 207326, 276454, 1167472, 158859, 160354, 177046, 16749, 92359, 349071, 250824, 144815, 274254, 126897, 40254, 401927, 436073, 127887, 145443, 130811
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

    repermute_l0_weights(&mut dst.l0_threat_pp_weights, &src.l0_threat_pp_weights, &repermute);

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
        permute(&mut dst.l0_threat_pp_weights, order, chunk_size, block_size);

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
