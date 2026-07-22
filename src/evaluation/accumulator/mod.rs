use crate::board::observer::BoardObserver;
use crate::board::piece::Piece;
use crate::board::side::Side;
use crate::board::square::Square;
use crate::board::Board;
use crate::evaluation::accumulator::psq::PieceSquareAccumulator;
use crate::evaluation::accumulator::threat::ThreatAccumulator;
use crate::evaluation::feature::psq::PieceSquareFeature;

pub mod psq;
pub mod threat;

#[derive(Default)]
pub struct Accumulator {
    pub psq: PieceSquareAccumulator,
    pub threat: ThreatAccumulator,
}

/// The Accumulator is passed in as a `BoardObserver` to `Board::make_move`, and the relevant updates
/// to both the `PieceSquareAccumulator` and `ThreatAccumulator` are recorded to later be applied
/// lazily during evaluation.
impl BoardObserver for Accumulator {
    fn on_piece_create(&mut self, board: &Board, pc: Piece, side: Side, sq: Square) {
        self.psq.adds.push(PieceSquareFeature::new(pc, sq, side));
        self.threat.push_piece_create(board, pc, side, sq);
    }

    fn on_piece_destroy(&mut self, board: &Board, pc: Piece, side: Side, sq: Square) {
        self.psq.subs.push(PieceSquareFeature::new(pc, sq, side));
        self.threat.push_piece_destroy(board, pc, side, sq);
    }

    fn on_piece_teleport(
        &mut self,
        board: &Board,
        pc: Piece,
        side: Side,
        from: Square,
        to: Square,
    ) {
        self.psq.subs.push(PieceSquareFeature::new(pc, from, side));
        self.psq.adds.push(PieceSquareFeature::new(pc, to, side));
        self.threat.push_piece_teleport(board, pc, side, from, to);
    }

    fn on_piece_transform(
        &mut self,
        board: &Board,
        old_pc: Piece,
        old_side: Side,
        new_pc: Piece,
        new_side: Side,
        sq: Square,
    ) {
        self.psq
            .subs
            .push(PieceSquareFeature::new(old_pc, sq, old_side));
        self.psq
            .adds
            .push(PieceSquareFeature::new(new_pc, sq, new_side));
        self.threat
            .push_piece_transform(board, old_pc, old_side, new_pc, new_side, sq);
    }
}
