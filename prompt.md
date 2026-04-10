I want to add sparse matrix multiplication to the L1 (feature transformer → first hidden layer) propagation in my multilayer NNUE chess engine, Hobbes. This is a Rust chess engine. The technique is used in Viridithas (https://github.com/cosmobobak/viridithas) and described at https://cosmo.tardis.ac/files/2024-08-17-multilayer.html.

## My network architecture

My network is fully integer-quantised — no floats anywhere in inference:

```rust
pub struct Network {
    pub l0_weights: [FeatureWeights; INPUT_BUCKET_COUNT],  // feature transformer
    pub l0_biases:  [i16; L1_SIZE],
    pub l1_weights: [[[i8; L1_SIZE]; L2_SIZE]; OUTPUT_BUCKET_COUNT],  // <-- sparse matmul target
    pub l1_biases:  [[i32; L2_SIZE]; OUTPUT_BUCKET_COUNT],
    pub l2_weights: [[[i32; L3_SIZE]; L2_SIZE * 2]; OUTPUT_BUCKET_COUNT],
    pub l2_biases:  [[i32; L3_SIZE]; OUTPUT_BUCKET_COUNT],
    pub l3_weights: [[i32; L3_SIZE]; OUTPUT_BUCKET_COUNT],
    pub l3_biases:  [i32; OUTPUT_BUCKET_COUNT],
}
```

The L1 layer takes u8 activations (from CReLU-into-pairwise on the i16 feature transformer output), multiplies against i8 weights, and accumulates into i32 sums. Biases are i32. The L1 activation is integer SCReLU (clamp + square in integer domain), NOT float conversion. All subsequent layers are also integer.

## What sparse matmul is

After the feature transformer produces activations (post CReLU-into-pairwise), many values are zero (encouraged by L1 regularisation on activations during training). Instead of doing a dense matrix-vector multiply for the L1 affine transform, we skip zero activations entirely by:

1. **During pairwise activation**: building a non-zero index list (`nnz`) alongside the u8 output.
2. **During L1 propagation**: iterating only over non-zero indices, loading the corresponding i8 weight rows and accumulating into i32.

## How the NNZ tracking works

- The u8 activations are reinterpreted as i32 (groups of 4 bytes).
- After packing each chunk of pairwise output, compute a nonzero mask of those i32 groups using architecture-specific intrinsics:
    - x86 AVX2: `_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(vec, zero)))` → u16 mask
    - x86 SSE: `_mm_movemask_ps(...)` → u16 mask
    - ARM NEON: `vaddvq_u32(vandq_u32(vtstq_u32(a, a), vld1q_u32([1,2,4,8])))` → u16 mask
- Use a precomputed lookup table (NNZ_TABLE) indexed by each byte of the mask. Each table entry contains packed u16 offsets of the set bits. Store these offsets (adjusted by a running base counter) into an `nnz` buffer, and count them.

## How the sparse L1 matmul works

Given `nnz_slice` (the list of non-zero i32-block indices) and `input32` (u8 activations reinterpreted as `&[i32]`):

- Main loop processes 4 NNZ entries at a time for ILP:
    - Load 4 non-zero block indices from nnz_slice
    - Splat each i32 activation into a SIMD register, reinterpret as i8/u8
    - Compute weight offsets: `nnz_index * L2_SIZE * L1_CHUNK_PER_32` (where L1_CHUNK_PER_32 = 4, since 4 u8 activations fit in one i32)
    - Inner loop over L2_SIZE/I32_CHUNK: load i32 accumulator, load 4 i8 weight vectors, use `madd_u8_i8_to_i32` (dpbusd on AVX-VNNI/AVX512, emulated on older ISAs) to multiply-accumulate pairs
- Tail loop handles remaining 0-3 entries one at a time
- After accumulation: add i32 biases, apply integer SCReLU activation (clamp and square in i32), and pass to L2 — no float conversion needed.

## What I need you to do

Look at my current NNUE inference code. Then:

1. **Add NNZ tracking to my pairwise/activation function**: After computing each chunk of u8 activations, compute the i32 nonzero mask and populate an `nnz` buffer with the indices of non-zero 4-byte blocks. Add the NNZ lookup table as a const. The table maps each possible byte (0..256) to a packed array of the bit-indices that are set.

2. **Replace my dense L1 matmul with a sparse version**: Instead of iterating over all L1_SIZE/4 blocks, iterate only over the entries in `nnz_slice`. Process 4 entries at a time in the main loop for instruction-level parallelism, with a scalar tail. The multiply-accumulate is u8 (activations) × i8 (weights) → i32 (sums). Keep the integer activation and bias addition as-is.

3. **Add the required SIMD helpers** if I don't already have them:
    - `nonzero_mask_i32(vec) -> u16` — returns bitmask of which i32 lanes are nonzero
    - `trans_i32_i8(vec) -> VecI8` — reinterpret i32 SIMD register as i8
    - `trans_i8_i32(vec) -> VecI32` — reinterpret i8 SIMD register as i32
    - `splat_i32(val) -> VecI32`
    - `madd_u8_i8_to_i32(acc, u8_input, i8_weight) -> VecI32` — unsigned×signed byte dot product accumulated to i32 (dpbusd or emulation)
    - `madd_2x_u8_i8_to_i32(acc, in_a, w_a, in_b, w_b) -> VecI32` — two-pair version for 4-wide unrolling
    - 128-bit helpers for NNZ table: `v128_zero`, `v128_splat(u16)`, `v128_load`, `v128_store`, `v128_add`

4. **Wire it up**: The `nnz` buffer and `nnz_count` should be stack-allocated (MaybeUninit) alongside `ft_outputs`. Pass `nnz_slice = &nnz[..nnz_count]` to the sparse L1 function.

Use my existing SIMD abstraction layer and naming conventions. Implement for whichever architectures I already support (likely AVX2 and NEON). Use `unsafe` blocks consistent with my existing style. Keep the generic (non-SIMD) fallback working too.

Reference implementation: https://github.com/cosmobobak/viridithas — look at the NNUE inference code for the concrete implementation patterns.

### How you can verify it works

The change should be non-functional - only a speedup. You can verify this by running 'cargo r -r bench'. The last line outputted will be something like this: 3116400 nodes 1078000 nps. The nps is not important, but the nodes should exactly match 3116400. 