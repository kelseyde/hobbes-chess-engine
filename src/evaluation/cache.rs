use crate::board::bitboard::Bitboard;
use crate::board::side::Side;
use crate::evaluation::arch::{L1_SIZE, NETWORK, NUM_BUCKETS};

/// Whenever the king changes bucket, a costly full refresh of the accumulator is required. This
/// service implements a technique to improve the performance of this refresh known as 'Finny tables'.
///
/// We keep a cache of the last accumulator and board state used for each bucket. When refreshing,
/// instead of starting from an empty board, we start from the last board state used for the bucket.
/// We therefore only need to apply the diff between the last board state and the current board state
/// to the accumulator.
#[derive(Clone, Default)]
#[repr(align(64))]
pub struct InputBucketCache {
    entries: Box<[[[CacheEntry; NUM_BUCKETS]; 2]; 2]>,
}

#[derive(Clone)]
pub struct CacheEntry {
    pub features: [i16; L1_SIZE],
    pub bitboards: [Bitboard; 8],
}

impl Default for CacheEntry {
    fn default() -> Self {
        CacheEntry {
            features: NETWORK.l0_biases,
            bitboards: [Bitboard::empty(); 8],
        }
    }
}

impl InputBucketCache {
    pub fn get(&mut self, perspective: Side, mirror: bool, bucket: usize) -> &mut CacheEntry {
        &mut self.entries[perspective][mirror as usize][bucket]
    }
}
