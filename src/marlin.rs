use viriformat::chess::piece::PieceType;
use viriformat::chess::squareset::SquareSet;
use crate::board::Board;

use viriformat::dataformat
use crate::types::piece::Piece;
use crate::types::side::Side;

impl Into<viriformat::chess::board::Board> for Board {
    fn into(self) -> viriformat::chess::board::Board {

        let pieces = [
            SquareSet::from_inner(self.pcs(Piece::Pawn).0),
            SquareSet::from_inner(self.pcs(Piece::Knight).0),
            SquareSet::from_inner(self.pcs(Piece::Bishop).0),
            SquareSet::from_inner(self.pcs(Piece::Rook).0),
            SquareSet::from_inner(self.pcs(Piece::Queen).0),
            SquareSet::from_inner(self.pcs(Piece::King).0)
        ];

        let colours = [
            SquareSet::from_inner(self.side(Side::White).0),
            SquareSet::from_inner(self.side(Side::Black).0)
        ];

        let mut piece_array: [Option<viriformat::chess::piece::Piece>; 64] = [None; 64];

        for sq in self.occ() {
            let pc = self.piece_at(sq).unwrap();
            let side = self.side_at(sq).unwrap();
            piece_array[sq] = Some(to_viri_piece(pc, side));
        };

    }
}

fn to_viri_piece(pc: Piece, side: Side) -> viriformat::chess::piece::Piece {
    let idx = pc as u8 + if side == Side::White { 0 } else { 6 };
    viriformat::chess::piece::Piece::from_index(idx).unwrap()
}