use crate::evaluation::simd;
use hobbes_nnue_arch::L1_SIZE;

// Credit to Pawnocchio and Viridithas authors from whom this implementation was shamelessly yoinked.

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
pub unsafe fn find_nonzero_indices(input: &[u8; L1_SIZE]) -> ([u16; L1_SIZE / 4], usize) {
    const UNROLL: usize = if 8 / simd::I32_LANES > 1 {
        8 / simd::I32_LANES
    } else {
        1
    };
    const NUM_CHUNKS: usize = UNROLL * simd::I32_LANES / 8;

    let mut indices = [0u16; L1_SIZE / 4];
    let indices_ptr = indices.as_mut_ptr();
    let mut count = 0usize;
    let mut base = simd::u16::splat(0);
    let step = simd::u16::splat(8);

    let mut i = 0;
    while i < L1_SIZE {
        let mut mask: u64 = 0;
        for j in 0..UNROLL {
            let nonzero_mask = simd::u8::nonzero_mask(input.as_ptr().add(i)) as u64;
            mask |= nonzero_mask << (j * simd::I32_LANES);
            i += simd::u8::LANES;
        }

        for chunk in 0..NUM_CHUNKS {
            let byte = (mask >> (chunk * 8)) as u8;
            let entry = &NNZ_TABLE.table[byte as usize];
            let actual_indices = simd::u16::add(simd::u16::load(entry.indices.as_ptr()), base);
            simd::u16::store(indices_ptr.add(count), actual_indices);
            count += byte.count_ones() as usize;
            base = simd::u16::add(base, step);
        }
    }

    (indices, count)
}

#[cfg(feature = "track_activations")]
mod bench {
    use super::L1_SIZE;
    use std::fs::File;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    pub static ACTIVATION_COUNTS: [AtomicU64; L1_SIZE / 2] =
        [const { AtomicU64::new(0) }; L1_SIZE / 2];

    pub const FILE_NAME: &str = "activations.txt";

    /// Tracks the activations of the neurons in the input layer of the NNUE neural network. Used
    /// to permute the network for sparsity.
    #[inline(always)]
    pub fn track_activations(l0_outputs: &[u8; L1_SIZE]) {
        for (i, &val) in l0_outputs.iter().enumerate() {
            if val != 0 {
                ACTIVATION_COUNTS[i % (L1_SIZE / 2)].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Writes the tracked activations to a file, to be injected into preprocess.rs.
    pub fn dump_activation_counts() {
        let mut file =
            File::create(FILE_NAME).expect(format!("Failed to create {FILE_NAME}").as_str());
        let counts: Vec<String> = ACTIVATION_COUNTS
            .iter()
            .map(|c| c.load(Ordering::Relaxed).to_string())
            .collect();
        writeln!(file, "{}", counts.join(", "))
            .expect(format!("Failed to write {FILE_NAME}").as_str());
        println!("Wrote l0 activation counts to {FILE_NAME}");
    }
}

#[cfg(feature = "track_activations")]
pub use bench::{dump_activation_counts, track_activations};

#[cfg(not(feature = "track_activations"))]
#[inline(always)]
pub fn track_activations(_l0_outputs: &[u8; L1_SIZE]) {}
