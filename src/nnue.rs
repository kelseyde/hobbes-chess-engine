pub const HIDDEN_SIZE: usize = 32;
pub const SCALE: i32 = 400;
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const QAB: i32 = QA * QB;

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    vals: [i16; HIDDEN_SIZE],
}

impl Accumulator {

    pub fn new(net: &Network) -> Self {
        net.feature_bias
    }

    pub fn add_feature(&mut self, feature_idx: usize, net: &Network) {
        for (i, d) in self.vals.iter_mut().zip(&net.feature_weights[feature_idx].vals) {
            *i += *d
        }
    }

    pub fn remove_feature(&mut self, feature_idx: usize, net: &Network) {
        for (i, d) in self.vals.iter_mut().zip(&net.feature_weights[feature_idx].vals) {
            *i -= *d
        }
    }

}

#[repr(C)]
pub struct Network {
    feature_weights: [Accumulator; 768],
    feature_bias: Accumulator,
    output_weights: [i16; 2 * HIDDEN_SIZE],
    output_bias: i16,
}

impl Network {

    pub fn evaluate(&self, us: &Accumulator, them: &Accumulator) -> i32 {
        let mut output = i32::from(self.output_bias);

        for (&input, &weight) in us.vals.iter().zip(&self.output_weights[..HIDDEN_SIZE]) {
            output += crelu(input) * i32::from(weight);
        }

        for (&input, &weight) in them.vals.iter().zip(&self.output_weights[HIDDEN_SIZE..]) {
            output += crelu(input) * i32::from(weight);
        }

        output *= SCALE;
        output /= QAB;
        output
    }

}

fn crelu(x: i16) -> i32 {
    0.max(x).min(QA as i16) as i32
}
//
// static NNUE: Network =
//     unsafe { std::mem::transmute(*include_bytes!("path/to/nnue")) };