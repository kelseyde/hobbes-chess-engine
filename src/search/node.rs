use crate::board::bitboard::Bitboard;
use crate::board::moves::Move;
use crate::board::piece::Piece;
use crate::search::{score, MAX_PLY};
use std::ops::{Index, IndexMut};

pub const KILLERS_PER_PLY: usize = 2;

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
    pub killers: KillerTable,
    pub singular: Option<Move>,
    pub threats: Bitboard,
    pub raw_eval: i32,
    pub static_eval: i32,
    pub reduction: i32,
    pub num_fail_highs: i32,
}

#[derive(Copy, Clone, Default)]
pub struct KillerTable {
    killers: [Option<Move>; KILLERS_PER_PLY],
}

impl Default for NodeStack {
    fn default() -> Self {
        NodeStack {
            data: [Node {
                mv: None,
                pc: None,
                captured: None,
                killers: KillerTable::default(),
                singular: None,
                threats: Bitboard::empty(),
                raw_eval: score::MIN,
                static_eval: score::MIN,
                reduction: 0,
                num_fail_highs: 0,
            }; MAX_PLY + 8],
        }
    }
}

impl NodeStack {
    /// Returns true if the given move is any killer at the given ply.
    #[inline(always)]
    pub fn is_killer(&self, ply: usize, mv: Move) -> bool {
        self[ply].killers.contains(mv)
    }

    /// Returns the index (0 = primary, 1 = secondary) of the killer slot for the given move,
    /// or None if it is not a killer.
    #[inline(always)]
    pub fn killer_index(&self, ply: usize, mv: Move) -> Option<usize> {
        self[ply].killers.index_of(mv)
    }

    /// Shift the current primary killer to secondary and store the new move as primary.
    #[inline(always)]
    pub fn update_killer(&mut self, ply: usize, mv: Move) {
        self[ply].killers.push(mv);
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


impl KillerTable {
    #[inline(always)]
    pub fn index_of(&self, mv: Move) -> Option<usize> {
        self.killers.iter().position(|&k| k == Some(mv))
    }

    #[inline(always)]
    pub fn contains(&self, mv: Move) -> bool {
        self.killers.iter().any(|&k| k == Some(mv))
    }

    #[inline(always)]
    pub fn push(&mut self, mv: Move) {
        // Shift second <- first, then set first
        self.killers[1] = self.killers[0];
        self.killers[0] = Some(mv);
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.killers = [None; KILLERS_PER_PLY];
    }
}