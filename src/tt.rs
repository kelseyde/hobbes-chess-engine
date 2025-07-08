use crate::moves::Move;
use crate::search::Score;
use std::mem::size_of;

pub struct TranspositionTable {
    table: Vec<TTEntry>,
    size: usize,
}

#[derive(Clone)]
#[derive(Default)]
pub struct TTEntry {
    key: u16,           // 2 bytes
    best_move: u16,     // 2 bytes
    score: i16,         // 2 bytes
    depth: u8,          // 1 byte
    flag: u8,           // 1 byte
}

#[derive(Eq, PartialEq, Debug, Clone, Copy)]
pub enum TTFlag {
    Exact = 0,
    Lower = 1,
    Upper = 2,
}

impl TTFlag {

    pub fn from_u8(val: u8) -> TTFlag {
        match val {
            0 => TTFlag::Exact,
            1 => TTFlag::Lower,
            2 => TTFlag::Upper,
            _ => panic!("Invalid hash flag value"),
        }
    }

    pub fn to_u8(&self) -> u8 {
        match self {
            TTFlag::Exact => 0,
            TTFlag::Lower => 1,
            TTFlag::Upper => 2,
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
        TTFlag::from_u8(self.flag)
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
        TranspositionTable { table, size }
    }

    pub fn resize(&mut self, size_mb: usize) {
        let size = size_mb * 1024 * 1024 / size_of::<TTEntry>();
        self.table = vec![TTEntry::default(); size];
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

    pub fn insert(&mut self, hash: u64, mut best_move: Move, score: i32, depth: u8, ply: usize, flag: TTFlag) {
        let idx = self.idx(hash);
        let entry = &mut self.table[idx];

        let key_part = (hash & 0xFFFF) as u16;
        let key_match = key_part == entry.key;

        if !best_move.exists() && key_match {
            best_move = entry.best_move();
        }

        entry.key = key_part;
        entry.best_move = best_move.0;
        entry.score = to_tt(score, ply);
        entry.depth = depth;
        entry.flag = flag.to_u8();
    }

    fn idx(&self, hash: u64) -> usize {
        let key = (hash >> 48) as u16;
        (key as usize) & (self.table.len() - 1)
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

        tt.insert(hash, best_move, score, depth, 0, flag);

        assert!(tt.probe(0x987654321FEDCBA).is_none());

        let entry = tt.probe(hash).unwrap();
        assert_eq!(entry.best_move(), best_move);
        assert_eq!(entry.score(0), score as i16);
        assert_eq!(entry.depth(), depth);
        assert_eq!(entry.flag(), flag);
    }

}
