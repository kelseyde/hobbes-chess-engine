use std::ops::{Index, IndexMut};

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

impl<T, const N: usize> Index<Side> for [T; N] {
    type Output = T;

    fn index(&self, stm: Side) -> &Self::Output {
        &self[stm as usize]
    }
}

impl<T, const N: usize> IndexMut<Side> for [T; N] {
    fn index_mut(&mut self, stm: Side) -> &mut Self::Output {
        &mut self[stm as usize]
    }
}