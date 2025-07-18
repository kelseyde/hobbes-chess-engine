use crate::moves::Move;
use crate::search::Score;
use std::mem::size_of;

pub struct TranspositionTable {
    table: Vec<TTEntry>,
    size_mb: usize,
    size: usize,
}

#[derive(Clone)]
pub struct TTEntry {
    key: u16,           // 2 bytes
    best_move: u16,     // 2 bytes
    score: i16,         // 2 bytes
    depth: u8,          // 1 byte
    flags: Flags,       // 1 byte
}

#[derive(Copy, Clone)]
pub struct Flags {
    data: u8,
}

impl Flags {
    pub const fn new(flag: TTFlag, pv: bool, age: u8) -> Self {
        Self { data: (flag as u8) | ((pv as u8) << 2) | (age << 3) }
    }

    pub const fn bound(self) -> TTFlag {
        unsafe { std::mem::transmute(self.data & 0b11) }
    }

    pub const fn pv(self) -> bool {
        (self.data & 0b100) != 0
    }

    pub const fn age(self) -> u8 {
        self.data >> 3
    }
}

#[derive(Eq, PartialEq, Debug, Clone, Copy)]
pub enum TTFlag {
    None = 0,
    Exact = 1,
    Lower = 2,
    Upper = 3,
}

impl Default for TTEntry {
    fn default() -> TTEntry {
        TTEntry {
            key: 0,
            depth: 0,
            best_move: 0,
            score: Score::MIN as i16,
            flags: Flags::new(TTFlag::None, false, 0)
        }
    }
}

impl TTEntry {

    pub fn best_move(&self) -> Move {
        Move(self.best_move)
    }

    pub fn score(&self, ply: usize) -> i16 {
        to_search(self.score as i32, ply)
    }

    pub fn depth(&self) -> u8 {
        self.depth
    }

    pub fn flag(&self) -> TTFlag {
        self.flags.bound()
    }

    pub fn pv(&self) -> bool {
        self.flags.pv()
    }

    pub fn validate_key(&self, key: u64) -> bool {
        self.key == (key & 0xFFFF) as u16
    }

}


impl Default for TranspositionTable {
    fn default() -> TranspositionTable {
        TranspositionTable::new(TranspositionTable::DEFAULT_SIZE)
    }
}

impl TranspositionTable {

    pub const DEFAULT_SIZE: usize = 16;

    pub fn new(size_mb: usize) -> TranspositionTable {
        let size = size_mb * 1024 * 1024 / size_of::<TTEntry>();
        let table = vec![TTEntry::default(); size];
        TranspositionTable { table, size_mb, size }
    }

    pub fn resize(&mut self, size_mb: usize) {
        let size = size_mb * 1024 * 1024 / size_of::<TTEntry>();
        self.table = vec![TTEntry::default(); size];
        self.size_mb = size_mb;
        self.size = size;
    }

    pub fn clear(&mut self) {
        self.table
            .iter_mut()
            .for_each(|entry| *entry = TTEntry::default());
    }

    pub fn probe(&self, hash: u64) -> Option<&TTEntry> {
        let idx = self.idx(hash);
        let entry = &self.table[idx];
        if entry.validate_key(hash) {
            Some(entry)
        } else {
            None
        }
    }

    pub fn insert(&mut self,
                  hash: u64,
                  mut best_move: Move,
                  score: i32,
                  depth: u8,
                  ply: usize,
                  flag: TTFlag,
                  pv: bool) {
        let idx = self.idx(hash);
        let entry = &mut self.table[idx];

        let key_part = hash as u16;
        let key_match = key_part == entry.key;

        if !best_move.exists() && key_match {
            best_move = entry.best_move();
        }

        entry.key = key_part;
        entry.best_move = best_move.0;
        entry.score = to_tt(score, ply);
        entry.depth = depth;
        entry.flags = Flags::new(flag, pv, 0);
    }

    fn idx(&self, hash: u64) -> usize {
        let key = hash as u128;
        let len = self.table.len() as u128;
        ((key * len) >> 64) as usize
    }

    pub fn size_mb(&self) -> usize {
        self.size_mb
    }

    pub fn fill(&self) -> usize {
        self.table.iter().take(1000)
            .filter(|entry| entry.flags.bound() != TTFlag::None)
            .count()
    }

    pub fn prefetch(&self, hash: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let index = self.idx(hash);
            let ptr = self.table.as_ptr().add(index);
            _mm_prefetch::<_MM_HINT_T0>(ptr as *const _);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = hash;
    }

}

fn to_tt(score: i32, ply: usize) -> i16 {
    if !Score::is_mate(score) {
        return score as i16 ;
    }
    if score > 0 { (score - ply as i32) as i16 } else { (score + ply as i32) as i16 }
}

fn to_search(score: i32, ply: usize) -> i16 {
    if !Score::is_mate(score) {
        return score as i16
    }
    if score > 0 { (score + ply as i32) as i16 } else { (score - ply as i32) as i16 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::MoveFlag;
    use crate::types::square::Square;

    #[test]
    fn test_tt() {
        let mut tt = TranspositionTable::new(16);
        let hash = 0x1234567890ABCDEF;
        let best_move = Move::new(Square(0), Square(1), MoveFlag::Standard);
        let score = 100;
        let depth = 5;
        let flag = TTFlag::Exact;

        tt.insert(hash, best_move, score, depth, 0, flag, true);

        assert!(tt.probe(0x987654321FEDCBA).is_none());

        let entry = tt.probe(hash).unwrap();
        assert_eq!(entry.best_move(), best_move);
        assert_eq!(entry.score(0), score as i16);
        assert_eq!(entry.depth(), depth);
        assert_eq!(entry.flag(), flag);
        assert!(entry.pv());
    }

}
