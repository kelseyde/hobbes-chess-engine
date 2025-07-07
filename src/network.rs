use core::panic;

use Side::{Black, White};

use crate::board::Board;
use crate::consts::{Piece, Side, PIECES};
use crate::moves::Move;
use crate::types::square::Square;

pub const FEATURES: usize = 768;
pub const HIDDEN: usize = 256;
pub const SCALE: i32 = 400;
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const QAB: i32 = QA * QB;

pub const PIECE_OFFSET: usize = 64;
pub const SIDE_OFFSET: usize = 64 * 6;
pub const MAX_ACCUMULATORS: usize = 255;

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

impl Default for Accumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Accumulator {
    pub fn new() -> Self {
        Accumulator {
            white_features: NETWORK.feature_bias,
            black_features: NETWORK.feature_bias,
        }
    }

    pub fn add(&mut self, w: usize, b: usize) {
        for i in 0..self.white_features.len() {
            self.white_features[i] += NETWORK.feature_weights[i + w * HIDDEN];
        }
        for i in 0..self.black_features.len() {
            self.black_features[i] += NETWORK.feature_weights[i + b * HIDDEN];
        }
    }

    pub fn sub(&mut self, w: usize, b: usize) {
        for i in 0..self.white_features.len() {
            self.white_features[i] -= NETWORK.feature_weights[i + w * HIDDEN];
        }
        for i in 0..self.black_features.len() {
            self.black_features[i] -= NETWORK.feature_weights[i + b * HIDDEN];
        }
    }

    pub fn add_sub(&mut self, w: [usize; 2], b: [usize; 2]) {
        for i in 0..self.white_features.len() {
            self.white_features[i] += NETWORK.feature_weights[i + w[0] * HIDDEN]
                .wrapping_sub(NETWORK.feature_weights[i + w[1] * HIDDEN]);
        }
        for i in 0..self.black_features.len() {
            self.black_features[i] += NETWORK.feature_weights[i + b[0] * HIDDEN]
                .wrapping_sub(NETWORK.feature_weights[i + b[1] * HIDDEN]);
        }
    }

    pub fn add_sub_sub(&mut self, w: [usize; 3], b: [usize; 3]) {
        for i in 0..self.white_features.len() {
            self.white_features[i] += NETWORK.feature_weights[i + w[0] * HIDDEN]
                .wrapping_sub(NETWORK.feature_weights[i + w[1] * HIDDEN])
                .wrapping_sub(NETWORK.feature_weights[i + w[2] * HIDDEN]);
        }
        for i in 0..self.black_features.len() {
            self.black_features[i] += NETWORK.feature_weights[i + b[0] * HIDDEN]
                .wrapping_sub(NETWORK.feature_weights[i + b[1] * HIDDEN])
                .wrapping_sub(NETWORK.feature_weights[i + b[2] * HIDDEN]);
        }
    }

    pub fn add_add_sub_sub(&mut self, w: [usize; 4], b: [usize; 4]) {
        for i in 0..self.white_features.len() {
            self.white_features[i] += NETWORK.feature_weights[i + w[0] * HIDDEN]
                .wrapping_add(NETWORK.feature_weights[i + w[1] * HIDDEN])
                .wrapping_sub(NETWORK.feature_weights[i + w[2] * HIDDEN])
                .wrapping_sub(NETWORK.feature_weights[i + w[3] * HIDDEN]);
        }
        for i in 0..self.black_features.len() {
            self.black_features[i] += NETWORK.feature_weights[i + b[0] * HIDDEN]
                .wrapping_add(NETWORK.feature_weights[i + b[1] * HIDDEN])
                .wrapping_sub(NETWORK.feature_weights[i + b[2] * HIDDEN])
                .wrapping_sub(NETWORK.feature_weights[i + b[3] * HIDDEN]);
        }
    }
}

pub struct NNUE {
    accumulator_stack: [Accumulator; MAX_ACCUMULATORS],
    current_accumulator: usize,
}

impl Default for NNUE {
    fn default() -> Self {
        Self::new()
    }
}

impl NNUE {
    pub fn new() -> Self {
        NNUE {
            current_accumulator: 0,
            accumulator_stack: [Accumulator::new(); MAX_ACCUMULATORS],
        }
    }

    pub fn update(
        &mut self,
        played_move: &Move,
        moving_piece: Piece,
        captured_piece: Option<Piece>,
        board: &Board,
    ) {
        if self.current_accumulator == MAX_ACCUMULATORS {
            panic!("reached maximum accumulator count");
        }
        self.current_accumulator += 1;
        // TODO: This can be optimized. No need to have an extra copy
        self.accumulator_stack[self.current_accumulator] =
            self.accumulator_stack[self.current_accumulator - 1];
        let us = board.stm;
        let them = us.flip();

        if !played_move.is_castle() {
            let new_piece_idx_w = {
                if !played_move.is_promo() {
                    self.ft_idx(played_move.to(), moving_piece, us, White)
                } else {
                    self.ft_idx(
                        played_move.to(),
                        played_move
                            .promo_piece()
                            .expect("expecting promotion piece to exist"),
                        us,
                        White,
                    )
                }
            };
            let new_piece_idx_b = {
                if !played_move.is_promo() {
                    self.ft_idx(played_move.to(), moving_piece, us, Black)
                } else {
                    self.ft_idx(
                        played_move.to(),
                        played_move
                            .promo_piece()
                            .expect("expecting promotion piece to exist"),
                        us,
                        Black,
                    )
                }
            };
            let moving_piece_index_w = self.ft_idx(played_move.from(), moving_piece, us, White);
            let moving_piece_index_b = self.ft_idx(played_move.from(), moving_piece, us, Black);

            if !board.is_noisy(played_move)
                | (board.captured(played_move).is_none() & played_move.is_promo())
            {
                // Quiet moves and non-capture promotions
                self.accumulator_stack[self.current_accumulator].add_sub(
                    [new_piece_idx_w, moving_piece_index_w],
                    [new_piece_idx_b, moving_piece_index_b],
                );
            } else {
                // All captures, including en passant
                let target_piece_w = {
                    if played_move.is_ep() {
                        self.ft_idx(Square(played_move.to().0 ^ 8), Piece::Pawn, them, White)
                    } else {
                        self.ft_idx(
                            played_move.to(),
                            captured_piece.expect("expecting captured piece to exist"),
                            them,
                            White,
                        )
                    }
                };
                let target_piece_b = {
                    if played_move.is_ep() {
                        self.ft_idx(Square(played_move.to().0 ^ 8), Piece::Pawn, them, Black)
                    } else {
                        self.ft_idx(
                            played_move.to(),
                            captured_piece.expect("expecting captured piece to exist"),
                            them,
                            Black,
                        )
                    }
                };
                self.accumulator_stack[self.current_accumulator].add_sub_sub(
                    [new_piece_idx_w, moving_piece_index_w, target_piece_w],
                    [new_piece_idx_b, moving_piece_index_b, target_piece_b],
                );
            }
        } else {
            // Castling
            let kingside = played_move.to().0 > played_move.from().0;
            let white = us == White;
            let king_castling_target = played_move.to();
            let rook_castling_target = Move::rook_to(kingside, white);
            let castleable_rook = Move::rook_from(kingside, white);
            let king_castling_w = self.ft_idx(played_move.from(), Piece::King, us, White);
            let king_castling_b = self.ft_idx(played_move.from(), Piece::King, us, Black);
            let rook_castling_w = self.ft_idx(castleable_rook, Piece::Rook, us, White);
            let rook_castling_b = self.ft_idx(castleable_rook, Piece::Rook, us, Black);
            let king_castling_target_w = self.ft_idx(king_castling_target, Piece::King, us, White);
            let king_castling_target_b = self.ft_idx(king_castling_target, Piece::King, us, Black);
            let rook_castling_target_w = self.ft_idx(rook_castling_target, Piece::Rook, us, White);
            let rook_castling_target_b = self.ft_idx(rook_castling_target, Piece::Rook, us, Black);

            self.accumulator_stack[self.current_accumulator].add_add_sub_sub(
                [
                    king_castling_target_w,
                    rook_castling_target_w,
                    king_castling_w,
                    rook_castling_w,
                ],
                [
                    king_castling_target_b,
                    rook_castling_target_b,
                    king_castling_b,
                    rook_castling_b,
                ],
            );
        }
    }

    pub fn undo(&mut self) {
        if self.current_accumulator > 0 {
            self.current_accumulator -= 1;
        } else {
            panic!("attempted to undo past first accumulator");
        }
    }

    pub fn refresh(&mut self, board: &Board) {
        self.activate(board);
    }

    pub fn init(&mut self, board: Board) {
        self.current_accumulator = 0;
        self.refresh(&board);
    }

    pub fn evaluate(&mut self, board: &Board) -> i32 {
        let us = match board.stm {
            White => &self.accumulator_stack[self.current_accumulator].white_features,
            Black => &self.accumulator_stack[self.current_accumulator].black_features,
        };
        let them = match board.stm {
            White => &self.accumulator_stack[self.current_accumulator].black_features,
            Black => &self.accumulator_stack[self.current_accumulator].white_features,
        };
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
        self.accumulator_stack[self.current_accumulator] = Accumulator::new();
        for &pc in PIECES.iter() {
            for sq in board.pcs(pc) {
                let side = board.side_at(sq).unwrap();
                let w_idx = self.ft_idx(sq, pc, side, White);
                let b_idx = self.ft_idx(sq, pc, side, Black);
                self.accumulator_stack[self.current_accumulator].add(w_idx, b_idx);
            }
        }
    }

    fn ft_idx(&self, sq: Square, pc: Piece, side: Side, perspective: Side) -> usize {
        let sq = if perspective == White {
            sq
        } else {
            sq.flip_rank()
        };
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
