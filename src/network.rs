use Side::{Black, White};

use crate::board::Board;
use crate::consts::{Piece, Side, PIECES};
use crate::types::square::Square;

pub const FEATURES: usize = 768;
pub const HIDDEN: usize = 256;
pub const SCALE: i32 = 400;
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const QAB: i32 = QA * QB;

pub const PIECE_OFFSET: usize = 64;
pub const SIDE_OFFSET: usize = 64 * 6;

static NETWORK: Network =
    unsafe { std::mem::transmute(*include_bytes!("../resources/woodpusher.nnue")) };

#[repr(C, align(64))]
pub struct Network {
    feature_weights: [i16; FEATURES * HIDDEN],
    feature_bias: [i16; HIDDEN],
    output_weights: [i16; 2 * HIDDEN],
    output_bias: i16,
}

#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Accumulator {
    pub white_features: [i16; HIDDEN],
    pub black_features: [i16; HIDDEN],
}

impl Accumulator {

    pub fn new() -> Self {
        Accumulator {
            white_features: NETWORK.feature_bias,
            black_features: NETWORK.feature_bias,
        }
    }

    pub fn add(&mut self, wx1: usize, bx1: usize) {
        for i in 0..self.white_features.len() {
            self.white_features[i] += NETWORK.feature_weights[i + wx1 * HIDDEN];
        }
        for i in 0..self.black_features.len() {
            self.black_features[i] += NETWORK.feature_weights[i + bx1 * HIDDEN];
        }
    }

}

pub struct NNUE {
    pub acc: Accumulator
}

impl NNUE {

    pub fn new() -> Self {
        NNUE {
            acc: Accumulator::new()
        }
    }

    pub fn evaluate(&mut self, board: &Board) -> i32 {
        self.activate(board);
        let us = match board.stm { White => &self.acc.white_features, Black => &self.acc.black_features };
        let them = match board.stm { White => &self.acc.black_features, Black => &self.acc.white_features };
        let mut output = i32::from(NETWORK.output_bias);

        for (&input, &weight) in us.iter().zip(NETWORK.output_weights[..HIDDEN].iter()) {
            output += crelu(input) * i32::from(weight);
        }

        for (&input, &weight) in them.iter().zip(NETWORK.output_weights[HIDDEN..].iter()) {
            output += crelu(input) * i32::from(weight);
        }

        output *= SCALE;
        output /= QAB;
        output
    }

    pub fn activate(&mut self, board: &Board) {
        self.acc = Accumulator::new();
        for &pc in PIECES.iter() {
            for sq in board.pcs(pc) {
                let side = board.side_at(sq).unwrap();
                let w_idx = self.ft_idx(sq, pc, side, White);
                let b_idx = self.ft_idx(sq, pc, side, Black);
                self.acc.add(w_idx, b_idx);
            }
        }
    }

    fn ft_idx(&self, sq: Square, pc: Piece, side: Side, perspective: Side) -> usize {
        let sq = if perspective == White { sq } else { sq.flip_rank() };
        let pc_offset = pc as usize * PIECE_OFFSET;
        let side_offset = if side == perspective { 0 } else { SIDE_OFFSET };
        side_offset + pc_offset + sq.0 as usize
    }

}

fn crelu(x: i16) -> i32 {
    0.max(x).min(QA as i16) as i32
}

#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::fen;

    use super::NNUE;

    #[test]
    fn test_startpos() {
        let board = Board::from_fen(fen::STARTPOS);
        let mut eval = NNUE::new();
        let score = eval.evaluate(&board);
        assert_eq!(score, 26);
    }

}