use crate::board::moves::Move;
use crate::search::score;
use crate::search::score::{to_search, to_tt};
use std::mem::size_of;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering::Relaxed};

/// The transposition table is a lookup table that stores the results of previously searched
/// positions, including the search depth, the score, the best move found, and other relevant
/// information. Since positions are often encountered via different move orders (via
/// 'transposition'), the transposition table therefore greatly reduces the size of the search tree,
/// since on subsequent visits we can re-use the results of previous searches.
pub const DEFAULT_TT_SIZE: usize = 16;
const ENTRIES_PER_BUCKET: usize = 3;
const BUCKET_SIZE: usize = size_of::<Bucket>();
const AGE_CYCLE: u8 = 1 << 5;
const AGE_MASK: u8 = AGE_CYCLE - 1;

const _: () = assert!(BUCKET_SIZE == 32);

const SCORE_SHIFT: u32 = 16;
const EVAL_SHIFT: u32 = 32;
const DEPTH_SHIFT: u32 = 48;
const FLAGS_SHIFT: u32 = 56;

pub struct TranspositionTable {
    table: Vec<Bucket>,
    size_mb: usize,
    size: usize,
    age: AtomicU8,
}

/// Pack the default score and eval values into their appropriate positions in the 64-bit data word.
const DEFAULT_ENTRY_RAW: u64 = {
    let score = (score::MIN as i16) as u16 as u64;
    let eval = (score::MIN as i16) as u16 as u64;
    (score << SCORE_SHIFT) | (eval << EVAL_SHIFT)
};

#[repr(align(32))]
struct Bucket {
    data: [AtomicU64; ENTRIES_PER_BUCKET],
    keys: AtomicU64,
}

impl Default for Bucket {
    fn default() -> Bucket {
        Bucket {
            data: [
                AtomicU64::new(DEFAULT_ENTRY_RAW),
                AtomicU64::new(DEFAULT_ENTRY_RAW),
                AtomicU64::new(DEFAULT_ENTRY_RAW),
            ],
            keys: AtomicU64::new(0),
        }
    }
}

impl Bucket {
    /// Load and unpack the three 16-bit keys from the packed keys word.
    #[inline]
    fn load_keys(&self) -> [u16; ENTRIES_PER_BUCKET] {
        let k = self.keys.load(Relaxed);
        [k as u16, (k >> 16) as u16, (k >> 32) as u16]
    }

    /// Pack the three 16-bit keys into a single word, leaving the top 16 bits unused.
    #[inline]
    fn pack_keys(keys: [u16; ENTRIES_PER_BUCKET]) -> u64 {
        (keys[0] as u64) | ((keys[1] as u64) << 16) | ((keys[2] as u64) << 32)
    }
}

#[derive(Clone)]
#[repr(C)]
pub struct Entry {
    key: u16,       // 2 bytes
    best_move: u16, // 2 bytes
    score: i16,     // 2 bytes
    eval: i16,      // 2 bytes
    depth: u8,      // 1 byte
    flags: Flags,   // 1 byte
}

#[derive(Eq, PartialEq, Debug, Clone, Copy)]
pub enum TTFlag {
    None = 0,
    Exact = 1,
    Lower = 2,
    Upper = 3,
}

impl TTFlag {
    pub const fn bounds_match(&self, score: i32, lower: i32, upper: i32) -> bool {
        match self {
            TTFlag::None => false,
            TTFlag::Exact => true,
            TTFlag::Lower => score >= upper,
            TTFlag::Upper => score <= lower,
        }
    }

    pub const fn from_score(score: i32, alpha: i32, beta: i32) -> Self {
        match score {
            s if s <= alpha => TTFlag::Upper,
            s if s >= beta => TTFlag::Lower,
            _ => TTFlag::Exact,
        }
    }
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
            score: score::MIN as i16,
            eval: score::MIN as i16,
            flags: Flags::new(TTFlag::None, false, 0),
        }
    }
}

impl Entry {
    /// Decode an entry from its 16-bit key and packed 64-bit data word.
    #[inline]
    fn from_parts(key: u16, data: u64) -> Entry {
        Entry {
            key,
            best_move: data as u16,
            score: (data >> SCORE_SHIFT) as i16,
            eval: (data >> EVAL_SHIFT) as i16,
            depth: (data >> DEPTH_SHIFT) as u8,
            flags: Flags {
                data: (data >> FLAGS_SHIFT) as u8,
            },
        }
    }

    /// Pack this entry's non-key payload into a single 64-bit data word.
    #[inline]
    fn to_data(&self) -> u64 {
        (self.best_move as u64)
            | ((self.score as u16 as u64) << SCORE_SHIFT)
            | ((self.eval as u16 as u64) << EVAL_SHIFT)
            | ((self.depth as u64) << DEPTH_SHIFT)
            | ((self.flags.data as u64) << FLAGS_SHIFT)
    }

    pub const fn best_move(&self) -> Move {
        Move(self.best_move)
    }

    pub fn score(&self, ply: usize) -> i16 {
        to_search(self.score as i32, ply) as i16
    }

    pub fn static_eval(&self) -> i16 {
        self.eval
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
    /// Create a new transposition table with the given size in megabytes. The number of entries in
    /// the table is calculated based on the size of each bucket and the number of entries per bucket.
    pub fn new(size_mb: usize) -> TranspositionTable {
        let size = size_mb * 1024 * 1024 / BUCKET_SIZE;
        let table = (0..size).map(|_| Bucket::default()).collect();
        let age = AtomicU8::new(0);
        TranspositionTable {
            table,
            size_mb,
            size,
            age,
        }
    }

    /// Resize the transposition table to the given size in megabytes. This will reallocate the table
    /// and reset all entries to their default values.
    pub fn resize(&mut self, size_mb: usize) {
        let size = size_mb * 1024 * 1024 / BUCKET_SIZE;
        self.table = (0..size).map(|_| Bucket::default()).collect();
        self.size_mb = size_mb;
        self.size = size;
    }

    /// Clear all entries in the transposition table by resetting them to their default values.
    pub fn clear(&self) {
        for bucket in self.table.iter() {
            for word in &bucket.data {
                word.store(DEFAULT_ENTRY_RAW, Relaxed);
            }
            bucket.keys.store(0, Relaxed);
        }
        self.age.store(0, Relaxed);
    }

    /// Increment the age of the transposition table. This is used to track the relative age of the
    /// entries in the table, which is a factor in the entry replacement scheme.
    pub fn birthday(&self) {
        let next = (self.age.load(Relaxed) + 1) & AGE_MASK;
        self.age.store(next, Relaxed);
    }

    /// Probe the transposition table for an entry with the given hash. The hash is used as an index
    /// into a bucket, and the entries in the bucket are searched for a matching key.
    pub fn probe(&self, hash: u64) -> Option<Entry> {
        let idx = self.idx(hash);
        let bucket = &self.table[idx];
        let key_part = hash as u16;
        let keys = bucket.load_keys();
        for (i, &key) in keys.iter().enumerate() {
            if key == key_part {
                return Some(Entry::from_parts(key, bucket.data[i].load(Relaxed)));
            }
        }
        None
    }

    /// Insert a new entry into the transposition table. We iterate through the entries in the bucket
    /// to find either an empty slot or the least valuable entry to replace, based on a quality metric
    /// that considers both search depth and entry age.
    #[allow(clippy::too_many_arguments)]
    pub fn insert(
        &self,
        hash: u64,
        best_move: Move,
        score: i32,
        static_eval: i32,
        depth: i32,
        ply: usize,
        flag: TTFlag,
        pv: bool,
    ) {
        let idx = self.idx(hash);
        let tt_age = self.age.load(Relaxed);
        let key_part = hash as u16;
        let cluster = &self.table[idx];

        let mut keys = cluster.load_keys();
        let entries = [
            Entry::from_parts(keys[0], cluster.data[0].load(Relaxed)),
            Entry::from_parts(keys[1], cluster.data[1].load(Relaxed)),
            Entry::from_parts(keys[2], cluster.data[2].load(Relaxed)),
        ];

        let mut index = 0;
        let mut minimum = i32::MAX;

        for (i, entry) in entries.iter().enumerate() {
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

        let entry = &entries[index];

        let key_match = key_part == entry.key;
        let mv = if !best_move.exists() && key_match {
            entry.best_move()
        } else {
            best_move
        };

        if !(key_part != entry.key
            || flag == TTFlag::Exact
            || depth + 4 > entry.depth as i32
            || entry.flags.age() != tt_age)
        {
            return;
        }

        let new_entry = Entry {
            key: key_part,
            best_move: mv.0,
            score: to_tt(score, ply) as i16,
            eval: static_eval as i16,
            depth: depth as u8,
            flags: Flags::new(flag, pv, tt_age),
        };

        cluster.data[index].store(new_entry.to_data(), Relaxed);
        keys[index] = key_part;
        cluster.keys.store(Bucket::pack_keys(keys), Relaxed);
    }

    /// For efficient indexing into an arbitrarily-sized hash table, we use Lemire's multiplication
    /// trick (see: https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/).
    const fn idx(&self, hash: u64) -> usize {
        let key = hash as u128;
        let len = self.size as u128;
        ((key * len) >> 64) as usize
    }

    pub const fn size_mb(&self) -> usize {
        self.size_mb
    }

    /// Calculate the fill level of the transposition table by counting the number of occupied
    /// entries out of the first 1000. This information is printed to UCI.
    pub fn fill(&self) -> usize {
        let mut fill = 0;
        for bucket in self.table.iter().take(1000 / ENTRIES_PER_BUCKET) {
            for word in &bucket.data {
                let flags = Flags {
                    data: (word.load(Relaxed) >> FLAGS_SHIFT) as u8,
                };
                if flags.bound() != TTFlag::None {
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
        Self {
            data: (flag as u8) | ((pv as u8) << 2) | (age << 3),
        }
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
