use crate::board::moves::Move;
use crate::search::score::{to_search, to_tt};
use std::mem::{size_of, transmute};
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::Relaxed};

/// The transposition table is a lookup table that stores the results of previously searched
/// positions, including the search depth, the score, the best move found, and other relevant
/// information. Since positions are often encountered via different move orders (via
/// 'transposition'), the transposition table therefore greatly reduces the size of the search tree,
/// since on subsequent visits we can re-use the results of previous searches.
const DEFAULT_TT_SIZE: usize = 16;
const ENTRIES_PER_BUCKET: usize = 3;
const AGE_CYCLE: u8 = 1 << 5;
const AGE_MASK: u8 = AGE_CYCLE - 1;

/// A single transposition table entry, packed into 8 bytes.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct TTEntry {
    pub eval: i16,
    pub score: i16,
    best_move: u16,
    pub depth: u8,
    pub flags: Flags,
}

const _: () = assert!(size_of::<TTEntry>() == 8);

#[repr(C, align(32))]
struct TTClusterMemory {
    data: [AtomicU64; 4],
}

/// A cluster of 3 entries plus packed keys
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct TTCluster {
    entries: [TTEntry; ENTRIES_PER_BUCKET],
    keys: u64,
}

const _: () = assert!(size_of::<TTCluster>() == 32);
const _: () = assert!(size_of::<TTClusterMemory>() == 32);

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

impl TTEntry {
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
}

impl TTClusterMemory {
    fn load(&self) -> TTCluster {
        let a = self.data[0].load(Relaxed);
        let b = self.data[1].load(Relaxed);
        let c = self.data[2].load(Relaxed);
        let d = self.data[3].load(Relaxed);
        unsafe { transmute([a, b, c, d]) }
    }

    fn store(&self, cluster: TTCluster) {
        let [a, b, c, d]: [u64; 4] = unsafe { transmute(cluster) };
        self.data[0].store(a, Relaxed);
        self.data[1].store(b, Relaxed);
        self.data[2].store(c, Relaxed);
        self.data[3].store(d, Relaxed);
    }

    fn clear(&self) {
        self.data[0].store(0, Relaxed);
        self.data[1].store(0, Relaxed);
        self.data[2].store(0, Relaxed);
        self.data[3].store(0, Relaxed);
    }

    fn empty() -> Self {
        Self {
            data: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
        }
    }
}

impl TTCluster {
    /// Returns the index of the entry whose key matches, or None if no matching entry is found
    fn key_idx(&self, key: u16) -> Option<usize> {
        let low_bits: u64 = 0x0001_0001_0001_0001;
        let high_bits = low_bits << 15;
        let splat = (key as u64).wrapping_mul(low_bits);
        let diff = splat ^ self.keys;
        let i = (!diff & diff.wrapping_sub(low_bits) & high_bits).trailing_zeros() / 16;
        if i < 3 { Some(i as usize) } else { None }
    }

    #[allow(clippy::identity_op)]
    fn keys(&self) -> [u16; ENTRIES_PER_BUCKET] {
        [
            (self.keys >> (0 * 16)) as u16,
            (self.keys >> (1 * 16)) as u16,
            (self.keys >> (2 * 16)) as u16,
        ]
    }

    #[allow(clippy::identity_op)]
    fn set_keys(&mut self, keys: [u16; ENTRIES_PER_BUCKET]) {
        self.keys = ((keys[0] as u64) << (0 * 16))
            | ((keys[1] as u64) << (1 * 16))
            | ((keys[2] as u64) << (2 * 16));
    }
}

pub struct TranspositionTable {
    table: Vec<TTClusterMemory>,
    size_mb: usize,
    size: usize,
    age: AtomicU8,
}

impl Default for TranspositionTable {
    fn default() -> TranspositionTable {
        TranspositionTable::new(DEFAULT_TT_SIZE)
    }
}

impl TranspositionTable {
    /// Create a new transposition table with the given size in megabytes.
    pub fn new(size_mb: usize) -> TranspositionTable {
        let size = size_mb * 1024 * 1024 / size_of::<TTClusterMemory>();
        let table: Vec<TTClusterMemory> =
            std::iter::repeat_with(TTClusterMemory::empty).take(size).collect();
        let age = AtomicU8::new(0);
        TranspositionTable {
            table,
            size_mb,
            size,
            age
        }
    }

    /// Resize the transposition table to the given size in megabytes.
    pub fn resize(&mut self, size_mb: usize) {
        let size = size_mb * 1024 * 1024 / size_of::<TTClusterMemory>();
        self.table = std::iter::repeat_with(TTClusterMemory::empty).take(size).collect();
        self.size_mb = size_mb;
        self.size = size;
    }

    /// Clear all entries in the transposition table.
    pub fn clear(&self) {
        self.table.iter().for_each(|entry| entry.clear());
        self.age.store(0, Relaxed);
    }

    /// Increment the age of the transposition table. This is used to track the relative age of the
    /// entries in the table, which is a factor in the entry replacement scheme.
    pub fn birthday(&self) {
        self.age.fetch_update(Relaxed, Relaxed, |age| Some((age + 1) & AGE_MASK)).ok();
    }

    /// Probe the transposition table for an entry with the given hash. The hash is used as an index
    /// into a bucket, and the entries in the bucket are searched for a matching key.
    pub fn probe(&self, hash: u64) -> Option<TTEntry> {
        let idx = self.idx(hash);
        let key_part = hash as u16;
        let cluster = self.table[idx].load();
        if let Some(entry_idx) = cluster.key_idx(key_part) {
            Some(cluster.entries[entry_idx])
        } else {
            None
        }
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
        let index = self.idx(hash);
        let key_part = hash as u16;
        let age = self.age.load(Relaxed);

        let mut cluster = self.table[index].load();
        let mut keys = cluster.keys();

        let mut cluster_idx = 0;
        let mut min_value = i32::MAX;
        let mut old = None;

        for i in 0..ENTRIES_PER_BUCKET {
            let entry = cluster.entries[i];

            if keys[i] == key_part {
                old = Some(entry);
            }

            if keys[i] == key_part || entry.flags.bound() == TTFlag::None {
                cluster_idx = i;
                break;
            }

            let relative_age = ((AGE_CYCLE + age - entry.flags.age()) & AGE_MASK) as i32;
            let entry_value = entry.depth as i32 - 4 * relative_age;
            if entry_value < min_value {
                cluster_idx = i;
                min_value = entry_value;
            }
        }

        if flag == TTFlag::Exact
            || old.is_none_or(|old| {
                depth + 4 > old.depth as i32
                    || old.flags.bound() == TTFlag::None
                    || age != old.flags.age()
            })
        {
            let mv = if !best_move.exists() && old.is_some() {
                Move(old.unwrap().best_move)
            } else {
                best_move
            };

            keys[cluster_idx] = key_part;
            cluster.set_keys(keys);
            cluster.entries[cluster_idx] = TTEntry {
                eval: static_eval as i16,
                score: to_tt(score, ply) as i16,
                best_move: mv.0,
                depth: depth as u8,
                flags: Flags::new(flag, pv, age),
            };

            self.table[index].store(cluster);
        }
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
        for cluster_mem in self.table.iter().take(1000 / ENTRIES_PER_BUCKET) {
            let cluster = cluster_mem.load();
            for entry in &cluster.entries {
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
        Self {
            data: (flag as u8) | ((pv as u8) << 2) | (age << 3),
        }
    }

    pub const fn bound(self) -> TTFlag {
        unsafe { transmute(self.data & 0b11) }
    }

    pub const fn pv(self) -> bool {
        (self.data & 0b100) != 0
    }

    pub const fn age(self) -> u8 {
        self.data >> 3
    }
}
