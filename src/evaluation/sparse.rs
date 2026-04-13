use crate::evaluation::simd;
use hobbes_nnue_arch::L1_SIZE;
use std::mem::MaybeUninit;

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct NNZEntry {
    indices: [u16; 8],
}

pub struct NNZTable {
    pub table: [NNZEntry; 256],
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub const NNZ_TABLE: NNZTable = {
    let mut table = [NNZEntry { indices: [0; 8] }; 256];
    let mut i = 0;
    while i < 256 {
        let mut j = i;
        let mut k = 0;
        while j != 0 {
            table[i].indices[k] = j.trailing_zeros() as u16;
            j &= j - 1;
            k += 1;
        }
        i += 1;
    }
    NNZTable { table }
};

#[inline(always)]
pub unsafe fn find_nonzero_indices(
    input: &[u8; L1_SIZE],
) -> ([MaybeUninit<u16>; L1_SIZE / 4], usize) {
    const UNROLL: usize = if 8 / simd::I32_LANES > 1 { 8 / simd::I32_LANES } else { 1 };
    const NUM_CHUNKS: usize = UNROLL * simd::I32_LANES / 8;

    let mut indices: [MaybeUninit<u16>; L1_SIZE / 4] = MaybeUninit::uninit().assume_init();
    let mut count = 0usize;
    let mut base = simd::splat_u16(0);
    let step = simd::splat_u16(8);

    let mut i = 0;
    while i < L1_SIZE {
        let mut mask: u64 = 0;
        for j in 0..UNROLL {
            let nonzero_mask = simd::nonzero_mask_u8(input.as_ptr().add(i)) as u64;
            mask |= nonzero_mask << (j * simd::I32_LANES);
            i += simd::U8_LANES;
        }

        for chunk in 0..NUM_CHUNKS {
            let byte = (mask >> (chunk * 8)) as u8;
            let entry = &NNZ_TABLE.table[byte as usize];
            let actual_indices = simd::add_u16(simd::load_u16(entry.indices.as_ptr()), base);
            simd::store_u16(indices[count..].as_mut_ptr() as *mut u16, actual_indices);
            count += byte.count_ones() as usize;
            base = simd::add_u16(base, step);
        }
    }

    (indices, count)
}
