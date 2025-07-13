use crate::network::{HIDDEN, NETWORK, NUM_BUCKETS};
use crate::types::bitboard::Bitboard;
use crate::types::piece::Piece;
use crate::types::side::Side;

#[derive(Clone, Default)]
#[repr(align(64))]
pub struct InputBucketCache {
    entries: Box<[[[CacheEntry; NUM_BUCKETS]; 2]; 2]>,
}

#[derive(Clone)]
pub struct CacheEntry {
    pub features: [i16; HIDDEN],
    pub pieces: [Bitboard; Piece::COUNT],
    pub sides: [Bitboard; 2],
}

impl Default for CacheEntry {

    fn default() -> Self {
        CacheEntry {
            features: NETWORK.feature_bias.clone(),
            pieces: [Bitboard::empty(); Piece::COUNT],
            sides: [Bitboard::empty(); 2]
        }
    }

}

impl InputBucketCache {

    pub fn get(&mut self, perspective: Side, mirror: bool, bucket: usize) -> &mut CacheEntry {
        &mut self.entries[perspective][mirror as usize][bucket]
    }

}