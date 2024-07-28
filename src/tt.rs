use crate::moves::Move;

pub struct TT {
    table: Vec<TTEntry>,
    size: usize,
}

#[derive(Clone)]
pub struct TTEntry {
    key: u16,           // 2 bytes
    best_move: u16,     // 2 bytes
    score: i16,         // 2 bytes
    depth: u8,          // 1 byte
    flag: u8,           // 1 byte
}

#[derive(Eq, PartialEq)]
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

    pub fn score(&self) -> i16 {
        self.score
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

impl Default for TTEntry {
    fn default() -> TTEntry {
        TTEntry {
            key: 0, best_move: 0, score: 0, depth: 0, flag: 0,
        }
    }
}

impl Default for TT {
    fn default() -> TT {
        TT::new(TT::DEFAULT_SIZE)
    }
}

impl TT {

    pub const DEFAULT_SIZE: usize = 16;

    pub fn new(size_mb: usize) -> TT {
        let size = size_mb * 1024 * 1024 / size_of::<TTEntry>();
        let table = vec![TTEntry::default(); size];
        TT { table, size }
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

    pub fn insert(&mut self, hash: u64, best_move: Move, score: i32, depth: u8, flag: TTFlag) {
        let idx = self.idx(hash);
        let entry = &mut self.table[idx];
        entry.key = (hash & 0xFFFF) as u16;
        entry.best_move = best_move.0;
        entry.score = score as i16;
        entry.depth = depth;
        entry.flag = flag.to_u8();
    }

    fn idx(&self, hash: u64) -> usize {
        let key = (hash >> 48) as u16;
        (key as usize) & (self.table.len() - 1)
    }

}
