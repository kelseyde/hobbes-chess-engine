use crate::evaluation::NETWORK;
use hobbes_nnue_arch::{L0_QUANT, L0_SHIFT, L1_SHIFT, L1_SIZE, L2_SIZE, L3_SIZE, Q, Q_BITS};

/// L0 ('feature transformer') activation with NNZ tracking.
/// Produces u8 activations and a list of non-zero i32-block indices.
pub fn activate_l0(
    us: &[i16; L1_SIZE],
    them: &[i16; L1_SIZE],
) -> ([u8; L1_SIZE], [u16; L1_SIZE / 4], usize) {
    let mut output = [0; L1_SIZE];

    for (side, feats) in [us, them].into_iter().enumerate() {
        let base = side * (L1_SIZE / 2);
        for i in 0..(L1_SIZE / 2) {
            let left: i16 = feats[i];
            let right: i16 = feats[i + (L1_SIZE / 2)];

            let l_clamped: u8 = left.clamp(0, L0_QUANT as i16) as u8;
            let r_clamped: u8 = right.clamp(0, L0_QUANT as i16) as u8;

            let multiplied: i32 = l_clamped as i32 * r_clamped as i32;
            let result: u8 = (multiplied >> L0_SHIFT).clamp(0, 255) as u8;
            output[base + i] = result;
        }
    }

    // Build NNZ indices by scanning the output as i32 blocks
    let mut nnz = [0u16; L1_SIZE / 4];
    let mut nnz_count = 0usize;
    let output32 = unsafe { std::slice::from_raw_parts(output.as_ptr() as *const i32, L1_SIZE / 4) };
    for (i, &block) in output32.iter().enumerate() {
        if block != 0 {
            nnz[nnz_count] = i as u16;
            nnz_count += 1;
        }
    }

    (output, nnz, nnz_count)
}

/// L1 propagation (scalar, dense — ignores NNZ for simplicity)
/// Weight layout is sparse-friendly: flat [input_block][output * 4 + byte_within_block]
pub fn propagate_l1(input: &[u8; L1_SIZE], _nnz: &[u16], output_bucket: usize) -> [i32; L2_SIZE * 2] {
    let weights = &NETWORK.l1_weights[output_bucket];
    let biases = &NETWORK.l1_biases[output_bucket];

    let mut intermediate: [i32; L2_SIZE] = [0; L2_SIZE];

    // The weights are in sparse-friendly layout:
    //   weights[input_block * L2_SIZE * 4 + output_idx * 4 + byte_within_block]
    // For the scalar path, we reconstruct: weight for (output_idx, input_idx) is at
    //   input_block = input_idx / 4, byte = input_idx % 4
    //   offset = input_block * (L2_SIZE * 4) + output_idx * 4 + byte
    for output_idx in 0..L2_SIZE {
        for input_idx in 0..L1_SIZE {
            let input_block = input_idx / 4;
            let byte = input_idx % 4;
            let offset = input_block * (L2_SIZE * 4) + output_idx * 4 + byte;
            let weight: i32 = weights[offset] as i32;
            intermediate[output_idx] += input[input_idx] as i32 * weight;
        }
    }

    let mut output: [i32; L2_SIZE * 2] = [0; L2_SIZE * 2];
    for i in 0..L2_SIZE {
        let bias: i32 = biases[i];
        let mut out: i32 = intermediate[i];

        out >>= L1_SHIFT;
        out += bias;

        let crelu: i32 = out.clamp(0, Q as i32) << Q_BITS;
        let csrelu: i32 = (out * out).clamp(0, (Q * Q) as i32);

        output[i] = crelu;
        output[i + L2_SIZE] = csrelu;
    }

    output
}

/// L2 propagation
pub fn propagate_l2(input: &[i32; L2_SIZE * 2], output_bucket: usize) -> [i32; L3_SIZE] {
    let weights = &NETWORK.l2_weights[output_bucket];

    let mut out = NETWORK.l2_biases[output_bucket];
    for input_idx in 0..(L2_SIZE * 2) {
        let input_val = input[input_idx];
        for output_idx in 0..L3_SIZE {
            let weight = weights[input_idx][output_idx];
            // This multiplication moves us into [0, Q^3] space
            out[output_idx] += input_val * weight;
        }
    }
    out
}

/// L3 propagation
pub fn propagate_l3(input: &[i32; L3_SIZE], output_bucket: usize) -> i32 {
    let weights = &NETWORK.l3_weights[output_bucket];
    let bias = NETWORK.l3_biases[output_bucket];

    let mut output: i32 = bias;
    for (&input, &weight) in input.iter().zip(weights.iter()) {
        let clamped = input.clamp(0, (Q * Q * Q) as i32);
        // This multiplication moves us into [0, Q^4] space
        output += clamped * weight;
    }
    output
}
