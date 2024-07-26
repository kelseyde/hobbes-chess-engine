
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Piece {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5
}

#[derive(Clone, Copy, Default, PartialEq)]
pub enum Side {
    #[default]
    WHITE,
    BLACK
}

impl Side {

    pub fn flip(&self) -> Side {
        match self {
            Side::WHITE => Side::BLACK,
            Side::BLACK => Side::WHITE
        }
    }

    pub fn idx(&self) -> usize {
        *self as usize + 6
    }

}