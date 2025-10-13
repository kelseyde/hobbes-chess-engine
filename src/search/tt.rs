use crate::board::moves::Move;
use crate::search::Score;
use std::mem::size_of;

const DEFAULT_TT_SIZE: usize = 16;
const ENTRIES_PER_BUCKET: usize = 3;
const BUCKET_SIZE: usize = size_of::<Bucket>();
const AGE_CYCLE: u8 = 1 << 5;
const AGE_MASK: u8 = AGE_CYCLE - 1;

pub struct TranspositionTable {
    table: Vec<Bucket>,
    size_mb: usize,
    size: usize,
    age: u8,
}

#[derive(Clone, Default)]
#[repr(align(32))]
struct Bucket {
    entries: [Entry; ENTRIES_PER_BUCKET],
}

#[derive(Clone)]
#[repr(C)]
pub struct Entry {
    key: u16,           // 2 bytes
    best_move: u16,     // 2 bytes
    score: i16,         // 2 bytes
    static_eval: i16,   // 2 bytes
    depth: u8,          // 1 byte
    flags: Flags,       // 1 byte
}

#[derive(Eq, PartialEq, Debug, Clone, Copy)]
pub enum TTFlag {
    None = 0,
    Exact = 1,
    Lower = 2,
    Upper = 3,
}

#[derive(Copy, Clone)]
pub struct Flags {
    data: u8,
}

impl Default for Entry {
    fn default() -> Entry {
        Entry {
            key: 0,
            depth: 0,
            best_move: 0,
            score: Score::MIN as i16,
            static_eval: Score::MIN as i16,
            flags: Flags::new(TTFlag::None, false, 0)
        }
    }
}

impl Entry {

    pub const fn best_move(&self) -> Move {
        Move(self.best_move)
    }

    pub fn score(&self, ply: usize) -> i16 {
        to_search(self.score as i32, ply)
    }

    pub fn static_eval(&self) -> i16 {
        self.static_eval
    }

    pub const fn depth(&self) -> u8 {
        self.depth
    }

    pub const fn flag(&self) -> TTFlag {
        self.flags.bound()
    }

    pub const fn pv(&self) -> bool {
        self.flags.pv()
    }

    pub const fn validate_key(&self, key: u64) -> bool {
        self.key == (key & 0xFFFF) as u16
    }

    pub const fn relative_age(&self, tt_age: u8) -> i32 {
        ((AGE_CYCLE + tt_age - self.flags.age()) & AGE_MASK) as i32
    }

}

impl Default for TranspositionTable {
    fn default() -> TranspositionTable {
        TranspositionTable::new(DEFAULT_TT_SIZE)
    }
}

impl TranspositionTable {

    pub fn new(size_mb: usize) -> TranspositionTable {
        let size = size_mb * 1024 * 1024 / BUCKET_SIZE;
        let table = vec![Bucket::default(); size];
        let age = 0;
        TranspositionTable { table, size_mb, size, age }
    }

    pub fn resize(&mut self, size_mb: usize) {
        let size = size_mb * 1024 * 1024 / BUCKET_SIZE;
        self.table = vec![Bucket::default(); size];
        self.size_mb = size_mb;
        self.size = size;
    }

    pub fn clear(&mut self) {
        self.table
            .iter_mut()
            .for_each(|entry| *entry = Bucket::default());
        self.age = 0;
    }

    pub const fn birthday(&mut self) {
        self.age = (self.age + 1) & AGE_MASK;
    }

    pub fn probe(&self, hash: u64) -> Option<&Entry> {
        let idx = self.idx(hash);
        let bucket = &self.table[idx];
        for entry in &bucket.entries {
            if entry.validate_key(hash) {
                return Some(entry);
            }
        }
        None
    }

    pub fn insert(&mut self,
                  hash: u64,
                  best_move: Move,
                  score: i32,
                  static_eval: i32,
                  depth: i32,
                  ply: usize,
                  flag: TTFlag,
                  pv: bool) {

        let idx = self.idx(hash);
        let tt_age = self.age;
        let key_part = hash as u16;
        let cluster = &mut self.table[idx];

        let mut index = 0;
        let mut minimum = i32::MAX;

        for (i, entry) in cluster.entries.iter_mut().enumerate() {
            if entry.key == key_part || entry.flag() == TTFlag::None {
                index = i;
                break;
            }

            let quality = entry.depth as i32 - 4 * entry.relative_age(tt_age);
            if quality < minimum {
                index = i;
                minimum = quality;
            }
        }

        let entry = &mut cluster.entries[index];

        let key_match = key_part == entry.key;
        let mv = if !best_move.exists() && key_match { entry.best_move() } else { best_move };

        if !(key_part != entry.key
            || flag == TTFlag::Exact
            || depth + 4 > entry.depth as i32
            || entry.flags.age() != tt_age) {
            return;
        }

        entry.key = key_part;
        entry.best_move = mv.0;
        entry.score = to_tt(score, ply);
        entry.static_eval = static_eval as i16;
        entry.depth = depth as u8;
        entry.flags = Flags::new(flag, pv, tt_age);
    }

    const fn idx(&self, hash: u64) -> usize {
        let key = hash as u128;
        let len = self.size as u128;
        ((key * len) >> 64) as usize
    }

    pub const fn size_mb(&self) -> usize {
        self.size_mb
    }

    pub fn fill(&self) -> usize {
        let mut fill = 0;
        for bucket in self.table.iter().take(1000 / ENTRIES_PER_BUCKET) {
            for entry in &bucket.entries {
                if entry.flags.bound() != TTFlag::None {
                    fill += 1;
                }
            }
        }
        fill
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

const fn to_tt(score: i32, ply: usize) -> i16 {
    if !Score::is_mate(score) {
        return score as i16 ;
    }
    if score > 0 { (score - ply as i32) as i16 } else { (score + ply as i32) as i16 }
}

const fn to_search(score: i32, ply: usize) -> i16 {
    if !Score::is_mate(score) {
        return score as i16
    }
    if score > 0 { (score + ply as i32) as i16 } else { (score - ply as i32) as i16 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::moves::MoveFlag;
    use crate::board::square::Square;

    #[test]
    fn test_tt() {
        let mut tt = TranspositionTable::new(16);
        let hash = 0x1234567890ABCDEF;
        let best_move = Move::new(Square(0), Square(1), MoveFlag::Standard);
        let score = 100;
        let static_eval = -10000;
        let depth = 5;
        let flag = TTFlag::Exact;

        tt.insert(hash, best_move, score, depth, 0, static_eval, flag, true);

        assert!(tt.probe(0x987654321FEDCBA).is_none());

        let entry = tt.probe(hash).unwrap();
        assert_eq!(entry.best_move(), best_move);
        assert_eq!(entry.score(0), score as i16);
        assert_eq!(entry.static_eval(), static_eval as i16);
        assert_eq!(entry.depth() as i32, depth);
        assert_eq!(entry.flag(), flag);
        assert!(entry.pv());
    }

}
