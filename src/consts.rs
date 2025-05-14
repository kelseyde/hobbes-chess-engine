
pub const MAX_DEPTH: i32 = 255;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Score {
    #[default]
    Draw = 0,
    Max = 32767,
    Min = -32767,
    Mate = 32766
}

impl Score {
    pub fn is_mate(score: i32) -> bool {
        score.abs() >= Score::Mate as i32 - MAX_DEPTH
    }
}


#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Piece {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5
}

pub const PIECES: [Piece; 6] = [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King];

impl Piece {
    pub fn iter() -> impl Iterator<Item = Piece> {
        PIECES.iter().copied()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum Side {
    #[default]
    White,
    Black
}

impl Side {

    pub fn flip(&self) -> Side {
        match self {
            Side::White => Side::Black,
            Side::Black => Side::White
        }
    }

    pub const fn idx(&self) -> usize {
        *self as usize + 6
    }

}