use crate::{Network, UntransposedNetwork, L1_SIZE, L2_SIZE, L3_SIZE, OUTPUT_BUCKET_COUNT};

pub struct PermuteConfig {
    pub needs_permuting: bool,
    pub order: &'static [u8],
}

const L0_ACTIVATIONS: [usize; L1_SIZE / 2] = [
    233595, 407313, 681879, 91888, 174245, 405001, 145839, 301232, 231240, 439589, 447750, 601585,
    46263, 349784, 95259, 61730, 230601, 10308, 143158, 11231, 320625, 38806, 593177, 159709, 14893,
    42484, 635993, 310707, 364081, 177131, 198403, 830752, 343355, 103657, 133468, 243114, 311054,
    441899, 90892, 19798, 87826, 203493, 577786, 254953, 176019, 180953, 133015, 371989, 169639,
    27228, 208370, 273123, 305254, 65745, 137080, 57030, 291765, 67473, 169607, 888524, 142186,
    75472, 149548, 487641, 140686, 326086, 155524, 408929, 467867, 236511, 260704, 222874, 215900,
    346004, 1126938, 20990, 49114, 419138, 91065, 249478, 264740, 231205, 265259, 169978, 172391,
    237578, 238631, 71731, 249109, 254720, 188174, 217534, 122745, 67819, 542123, 99252, 87751,
    24275, 99116, 459798, 337426, 292719, 59299, 185358, 118269, 233402, 257246, 263957, 36395,
    316813, 295332, 463641, 138947, 47208, 415839, 251780, 272570, 150058, 328570, 55078, 373446,
    68811, 236863, 340377, 294667, 72811, 1277088, 79471, 666493, 190677, 102401, 33281, 188763,
    130734, 220696, 80079, 129638, 274652, 69975, 176475, 234588, 321261, 223967, 243275, 294436,
    287434, 324534, 54093, 61419, 3299, 320824, 401151, 140748, 434363, 138218, 324648, 187083,
    460108, 10901, 136660, 161844, 184180, 237317, 8521, 102179, 324670, 343584, 139085, 156221,
    309652, 204223, 203895, 317415, 127062, 604584, 248017, 278577, 105507, 424947, 106371, 206307,
    156541, 358377, 22647, 48694, 167792, 219214, 144606, 303820, 365379, 48958, 319353, 183002,
    114302, 518768, 72082, 93302, 257632, 1235924, 527895, 157880, 15152, 480376, 168692, 29424,
    15750, 322870, 163027, 549501, 190525, 201178, 66897, 17158, 44305, 255674, 133437, 327005,
    117422, 54215, 241723, 66935, 80711, 489367, 116160, 84457, 300481, 56340, 43979, 210087,
    832366, 25139, 332670, 26492, 20084, 312133, 56478, 90407, 35180, 152948, 118341, 458436,
    62387, 310545, 118393, 590415, 122627, 152204, 108514, 253478, 196120, 283670, 116394, 180736,
    239733, 233414, 29640, 827330, 156020, 75117, 103383, 461702, 702142, 90311, 661930, 227977,
    170642, 169987, 350432, 823980, 237332, 26745, 1295894, 358940, 253591, 562088, 131528, 42516,
    391047, 16468, 353860, 85590, 346611, 621298, 272663, 185123, 510553, 56359, 82304, 298750,
    86278, 532701, 432665, 191471, 153676, 16461, 59011, 375197, 243889, 440394, 229999, 195890,
    302611, 75772, 246241, 78502, 27838, 169714, 87624, 284357, 287599, 484212, 273214, 40913,
    29136, 69235, 514871, 437727, 129162, 215895, 161926, 271636, 214451, 59662, 209121, 168318,
    357204, 214603, 352497, 285352, 197178, 13007, 175998, 39604, 147127, 340714, 121225, 177316,
    332350, 170479, 285086, 125807, 439862, 76074, 226900, 284419, 108121, 181211, 214383, 37582,
    22618, 81543, 41791, 122000, 1344694, 154345, 263234, 418357, 224292, 16696, 98048, 112829,
    257631, 171706, 169111, 101024, 206188, 142205, 453167, 245419, 27172, 375742, 208278, 141066,
    304580, 157209, 242597, 311525, 34582, 181390, 251263, 146687, 50876, 176660, 88418
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
