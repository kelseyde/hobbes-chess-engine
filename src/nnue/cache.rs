use crate::network::{HIDDEN, NETWORK, NUM_BUCKETS};
use crate::types::bitboard::Bitboard;
use crate::types::side::Side;
use crate::utils::boxed_and_zeroed;

#[derive(Clone)]
#[repr(align(64))]
pub struct InputBucketCache {
    entries: Box<[[[CacheEntry; NUM_BUCKETS]; 2]; 2]>,
}

#[derive(Clone)]
pub struct CacheEntry {
    pub features: [i16; HIDDEN],
    pub bitboards: [Bitboard; 8],
}

impl Default for InputBucketCache {
    fn default() -> Self {
        Self {
            entries: unsafe { boxed_and_zeroed() }
        }
    }
}

impl Default for CacheEntry {
    fn default() -> Self {
        CacheEntry {
            features: NETWORK.feature_bias.clone(),
            bitboards: [Bitboard::empty(); 8],
        }
    }
}

impl InputBucketCache {

    pub fn get(&mut self, perspective: Side, mirror: bool, bucket: usize) -> &mut CacheEntry {
        &mut self.entries[perspective][mirror as usize][bucket]
    }

}