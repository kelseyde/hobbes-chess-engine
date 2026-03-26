use crate::evaluation::NETWORK;
use hobbes_nnue_arch::{L0_QUANT, L0_SHIFT, L1_SHIFT, L1_SIZE, L2_SIZE, L3_SIZE, Q, Q_BITS};
use crate::evaluation::forward::propagate_l1;

/// L0 ('feature transformer') activation
/// We are in [0, 255] space, we want to end up in [0, 127] space for the next layer.
pub fn activate_l0_and_propagate_l1(
    us: &[i16; L1_SIZE], 
    them: &[i16; L1_SIZE], 
    output_bucket: usize
) -> [i32; L2_SIZE * 2] {
    
    let mut output = [0; L1_SIZE];

    for (side, feats) in [us, them].into_iter().enumerate() {
        let base = side * (L1_SIZE / 2);
        for i in 0..(L1_SIZE / 2) {
            // Load the pair of inputs to be multiplied.
            let left: i16 = feats[i];
            let right: i16 = feats[i + (L1_SIZE / 2)];

            // Clipped ReLU activation, in [0, 255] space.
            let l_clamped: u8 = left.clamp(0, L0_QUANT as i16) as u8;
            let r_clamped: u8 = right.clamp(0, L0_QUANT as i16) as u8;

            // Pairwise multiplication of left and right input.
            let multiplied: i32 = l_clamped as i32 * r_clamped as i32;

            // Downshift back into [0, 127] space.
            // Note: this is equivalent to the << 7 >> 16 that mulhi does.
            let result: u8 = (multiplied >> L0_SHIFT).clamp(0, 255) as u8;
            output[base + i] = result;
        }
    }
    propagate_l1(&output, output_bucket)
}

/// L1 propagation
pub fn propagate_l1(input: &[u8; L1_SIZE], output_bucket: usize) -> [i32; L2_SIZE * 2] {
    let weights = &NETWORK.l1_weights[output_bucket];
    let biases = &NETWORK.l1_biases[output_bucket];

    // Unactivated L1 outputs in the quantized space L0_QUANT * L1_QUANT
    let mut intermediate: [i32; L2_SIZE] = [0; L2_SIZE];

    // L1 matrix multiplication
    for output_idx in 0..L2_SIZE {
        for input_idx in 0..L1_SIZE {
            let weight: i32 = weights[output_idx][input_idx] as i32;
            intermediate[output_idx] += input[input_idx] as i32 * weight;
        }
    }

    // Re-quantise, add biases and activate L1 outputs
    let mut output: [i32; L2_SIZE * 2] = [0; L2_SIZE * 2];
    for i in 0..L2_SIZE {
        let bias: i32 = biases[i];
        let mut out: i32 = intermediate[i];

        // Down-shift into L1 Q space
        out >>= L1_SHIFT;

        // Add the bias
        out += bias;

        // crelu activation: clamp(x, 0, Q)
        let crelu: i32 = out.clamp(0, Q as i32) << Q_BITS;

        // csrelu activation: clamp(x^2, 0, Q^2)
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
