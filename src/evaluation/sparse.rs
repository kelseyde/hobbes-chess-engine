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

#[cfg(test)]
mod tests {
    use super::*;
    use hobbes_nnue_arch::L1_SIZE;

    /// Brute-force reference: collect the indices of every i32-block (group of 4 bytes)
    /// that contains at least one non-zero byte.
    fn reference_nonzero_i32_blocks(input: &[u8; L1_SIZE]) -> Vec<u16> {
        (0..L1_SIZE / 4)
            .filter(|&i| input[i * 4..i * 4 + 4].iter().any(|&b| b != 0))
            .map(|i| i as u16)
            .collect()
    }

    fn check(input: &[u8; L1_SIZE]) {
        let expected = reference_nonzero_i32_blocks(input);
        let (raw, count) = unsafe { find_nonzero_indices(input) };
        let got: Vec<u16> = raw[..count]
            .iter()
            .map(|x| unsafe { x.assume_init() })
            .collect();
        assert_eq!(
            got, expected,
            "mismatch: got {got:?}, expected {expected:?}"
        );
    }

    #[test]
    fn all_zeros() {
        check(&[0u8; L1_SIZE]);
    }

    #[test]
    fn all_nonzero() {
        check(&[1u8; L1_SIZE]);
    }

    #[test]
    fn first_block_only() {
        let mut input = [0u8; L1_SIZE];
        input[0] = 42; // block 0
        check(&input);
    }

    #[test]
    fn last_block_only() {
        let mut input = [0u8; L1_SIZE];
        input[L1_SIZE - 1] = 7; // last block
        check(&input);
    }

    #[test]
    fn every_other_block() {
        let mut input = [0u8; L1_SIZE];
        // set byte 0 of every even-numbered i32-block
        for i in (0..L1_SIZE / 4).step_by(2) {
            input[i * 4] = 1;
        }
        check(&input);
    }

    #[test]
    fn non_zero_in_second_byte_of_block() {
        // non-zero value is in byte index 1 of block 3
        let mut input = [0u8; L1_SIZE];
        input[3 * 4 + 1] = 99;
        check(&input);
    }

    #[test]
    fn sparse_pattern() {
        let mut input = [0u8; L1_SIZE];
        // non-zero blocks at positions 0, 5, 17, 100, L1_SIZE/4 - 1
        for &block in &[0usize, 5, 17, 100, L1_SIZE / 4 - 1] {
            input[block * 4] = 1;
        }
        check(&input);
    }
}

