use std::ops::{Index, IndexMut};
use crate::board::bitboard::Bitboard;
use crate::board::moves::Move;
use crate::board::piece::Piece;
use crate::search::MAX_PLY;
use crate::search::score::Score;

pub trait NodeType {
    const PV: bool;
    const ROOT: bool;
}

pub struct Root;
impl NodeType for Root {
    const PV: bool = true;
    const ROOT: bool = true;
}

pub struct PV;
impl NodeType for PV {
    const PV: bool = true;
    const ROOT: bool = false;
}

pub struct NonPV;
impl NodeType for NonPV {
    const PV: bool = false;
    const ROOT: bool = false;
}

/// Represents the variation in the search tree currently being searched. Is updated every time the
/// node currently being searched changes.
pub struct NodeStack {
    data: [Node; MAX_PLY + 8],
}

/// Container for all the information related to a single node in the search tree, that is used during
/// search. Enables us to e.g. fetch information about parent nodes easily, or to re-use computationally
/// expensive information (e.g. static eval) if searching the same node twice.
#[derive(Copy, Clone)]
pub struct Node {
    pub mv: Option<Move>,
    pub pc: Option<Piece>,
    pub captured: Option<Piece>,
    pub killer: Option<Move>,
    pub singular: Option<Move>,
    pub threats: Bitboard,
    pub raw_eval: i32,
    pub static_eval: i32,
    pub reduction: i32,
}

impl Default for NodeStack {
    fn default() -> Self {
        NodeStack {
            data: [Node {
                mv: None,
                pc: None,
                captured: None,
                killer: None,
                singular: None,
                threats: Bitboard::empty(),
                raw_eval: Score::MIN,
                static_eval: Score::MIN,
                reduction: 0,
            }; MAX_PLY + 8],
        }
    }
}

impl Index<usize> for NodeStack {
    type Output = Node;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe { self.data.get_unchecked(index) }
    }
}

impl IndexMut<usize> for NodeStack {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { self.data.get_unchecked_mut(index) }
    }
}