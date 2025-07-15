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
    pub bitboards: [Bitboard; 8],
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

    pub fn print_bbs(&self) {
        for (perspective, entries) in self.entries.iter().enumerate() {
            for (mirror, buckets) in entries.iter().enumerate() {
                for (bucket, entry) in buckets.iter().enumerate() {
                    println!("Perspective: {}, Mirror: {}, Bucket: {}", perspective, mirror, bucket);
                    for pc in [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King] {
                        println!("Piece: {:?}", pc);
                        let bb = entry.bitboards[pc as usize];
                        if !bb.is_empty() {
                            bb.print();
                        }
                    }
                    for side in [Side::White, Side::Black] {
                        println!("Side: {:?}", side);
                        let bb = entry.bitboards[side.idx()];
                        if !bb.is_empty() {
                            bb.print();
                        }
                    }
                }
            }
        }
    }

}