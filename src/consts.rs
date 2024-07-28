
pub const MAX_DEPTH: u8 = 255;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Score {
    #[default]
    Draw = 0,
    Min = -30000,
    Max = 30000,
    Mate = 30000 - MAX_DEPTH as isize,
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

    pub fn idx(&self) -> usize {
        *self as usize + 6
    }

}