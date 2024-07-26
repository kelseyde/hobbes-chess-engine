use crate::bits;
use crate::board::Board;

#[derive(Debug, Clone, Copy)]
pub struct BishopLookup {
    pub attacks: [u64; 512],
    pub mask: u64,
    pub magic: u64,
    pub shift: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct RookLookup {
    pub attacks: [u64; 4096],
    pub mask: u64,
    pub magic: u64,
    pub shift: u64,
}

pub const ROOK_MAGICS: [u64; 64] =  [
    0x0080001020400080, 0x0040001000200040, 0x0080081000200080, 0x0080040800100080,
    0x0080020400080080, 0x0080010200040080, 0x0080008001000200, 0x0080002040800100,
    0x0000800020400080, 0x0000400020005000, 0x0000801000200080, 0x0000800800100080,
    0x0000800400080080, 0x0000800200040080, 0x0000800100020080, 0x0000800040800100,
    0x0000208000400080, 0x0000404000201000, 0x0000808010002000, 0x0000808008001000,
    0x0000808004000800, 0x0000808002000400, 0x0000010100020004, 0x0000020000408104,
    0x0000208080004000, 0x0000200040005000, 0x0000100080200080, 0x0000080080100080,
    0x0000040080080080, 0x0000020080040080, 0x0000010080800200, 0x0000800080004100,
    0x0000204000800080, 0x0000200040401000, 0x0000100080802000, 0x0000080080801000,
    0x0000040080800800, 0x0000020080800400, 0x0000020001010004, 0x0000800040800100,
    0x0000204000808000, 0x0000200040008080, 0x0000100020008080, 0x0000080010008080,
    0x0000040008008080, 0x0000020004008080, 0x0000010002008080, 0x0000004081020004,
    0x0000204000800080, 0x0000200040008080, 0x0000100020008080, 0x0000080010008080,
    0x0000040008008080, 0x0000020004008080, 0x0000800100020080, 0x0000800041000080,
    0x00FFFCDDFCED714A, 0x007FFCDDFCED714A, 0x003FFFCDFFD88096, 0x0000040810002101,
    0x0001000204080011, 0x0001000204000801, 0x0001000082000401, 0x0001FFFAABFAD1A2
];

pub const BISHOP_MAGICS: [u64; 64] =  [
    0x0002020202020200, 0x0002020202020000, 0x0004010202000000, 0x0004040080000000,
    0x0001104000000000, 0x0000821040000000, 0x0000410410400000, 0x0000104104104000,
    0x0000040404040400, 0x0000020202020200, 0x0000040102020000, 0x0000040400800000,
    0x0000011040000000, 0x0000008210400000, 0x0000004104104000, 0x0000002082082000,
    0x0004000808080800, 0x0002000404040400, 0x0001000202020200, 0x0000800802004000,
    0x0000800400A00000, 0x0000200100884000, 0x0000400082082000, 0x0000200041041000,
    0x0002080010101000, 0x0001040008080800, 0x0000208004010400, 0x0000404004010200,
    0x0000840000802000, 0x0000404002011000, 0x0000808001041000, 0x0000404000820800,
    0x0001041000202000, 0x0000820800101000, 0x0000104400080800, 0x0000020080080080,
    0x0000404040040100, 0x0000808100020100, 0x0001010100020800, 0x0000808080010400,
    0x0000820820004000, 0x0000410410002000, 0x0000082088001000, 0x0000002011000800,
    0x0000080100400400, 0x0001010101000200, 0x0002020202000400, 0x0001010101000200,
    0x0000410410400000, 0x0000208208200000, 0x0000002084100000, 0x0000000020880000,
    0x0000001002020000, 0x0000040408020000, 0x0004040404040000, 0x0002020202020000,
    0x0000104104104000, 0x0000002082082000, 0x0000000020841000, 0x0000000000208800,
    0x0000000010020200, 0x0000000404080200, 0x0000040404040400, 0x0002020202020200
];

pub const ROOK_SHIFTS: [u64; 64] = [
    52, 53, 53, 53, 53, 53, 53, 52,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 53, 53, 53, 53, 53
];

pub const BISHOP_SHIFTS: [u64; 64] = [
    58, 59, 59, 59, 59, 59, 59, 58,
    59, 59, 59, 59, 59, 59, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 59, 59, 59, 59, 59, 59,
    58, 59, 59, 59, 59, 59, 59, 58
];

pub const DIAGONAL_VECTORS: [i8; 4] = [-9, -7, 7, 9];
pub const ORTHOGONAL_VECTORS: [i8; 4] = [-8, -1, 1, 8];

pub const A_FILE_EXCEPTIONS: [i8; 3] = [-9, -1, 7];
pub const H_FILE_EXCEPTIONS: [i8; 3] = [-7, 1, 9];

pub static mut ROOK_MASKS: [u64; 64] = [0; 64];
pub static mut BISHOP_MASKS: [u64; 64] = [0; 64];

pub static mut ROOK_ATTACKS: [Vec<u64>; 64] = [const { Vec::new() }; 64];
pub static mut BISHOP_ATTACKS: [Vec<u64>; 64] = [const { Vec::new() }; 64];

pub static mut ROOK_MAGIC_LOOKUP: [RookLookup; 64] = [RookLookup { attacks: [0; 4096], mask: 0, magic: 0, shift: 0 }; 64];
pub static mut BISHOP_MAGIC_LOOKUP: [BishopLookup; 64] = [BishopLookup { attacks: [0; 512], mask: 0, magic: 0, shift: 0 }; 64];


pub fn init_magics() {
    unsafe {
        ROOK_MASKS = init_magic_masks(true);
        BISHOP_MASKS = init_magic_masks(false);
        ROOK_ATTACKS = init_magic_attacks(true, ROOK_MAGICS, ROOK_SHIFTS);
        BISHOP_ATTACKS = init_magic_attacks(false, BISHOP_MAGICS, BISHOP_SHIFTS);
        ROOK_MAGIC_LOOKUP = init_rook_magic_lookups(&ROOK_ATTACKS, &ROOK_MASKS, &ROOK_MAGICS, &ROOK_SHIFTS);
        BISHOP_MAGIC_LOOKUP = init_bishop_magic_lookups(&BISHOP_ATTACKS, &BISHOP_MASKS, &BISHOP_MAGICS, &BISHOP_SHIFTS);
    }
}

pub fn init_bishop_magic_lookups(all_attacks: &[Vec<u64>; 64], masks: &[u64; 64],
                          magics: &[u64; 64], shifts: &[u64; 64]) -> [BishopLookup; 64] {

    let mut magic_lookups = [BishopLookup { attacks: [0; 512], mask: 0, magic: 0, shift: 0 }; 64];
    for sq in 0..64 {
        let attack: [u64; 512] = vec_to_u64_array_with_padding(all_attacks[sq].clone());
        let mask = masks[sq];
        let magic = magics[sq];
        let shift = shifts[sq];
        magic_lookups[sq] = BishopLookup { attacks: attack, mask, magic, shift };
    }
    magic_lookups
}

pub fn init_rook_magic_lookups(all_attacks: &[Vec<u64>; 64], masks: &[u64; 64],
                          magics: &[u64; 64], shifts: &[u64; 64]) -> [RookLookup; 64] {

    let mut magic_lookups = [RookLookup { attacks: [0; 4096], mask: 0, magic: 0, shift: 0 }; 64];
    for sq in 0..64 {
        let attack: [u64; 4096] = vec_to_u64_array_with_padding(all_attacks[sq].clone());
        let mask = masks[sq];
        let magic = magics[sq];
        let shift = shifts[sq];
        magic_lookups[sq] = RookLookup { attacks: attack, mask, magic, shift };
    }
    magic_lookups
}

pub fn init_magic_attacks(orthogonal: bool, magics: [u64; 64], shifts: [u64; 64]) -> [Vec<u64>; 64]  {
    let mut magic_attacks: [Vec<u64>; 64] = [const { Vec::new() }; 64];
    for sq in 0..64 {
        magic_attacks[sq] = init_magic_table(sq, orthogonal, magics[sq], shifts[sq]);
    }
    magic_attacks
}

pub fn init_magic_table(sq: usize, orthogonal: bool, magic: u64, shift: u64) -> Vec<u64> {
    let num_bits = 64 - shift;
    let table_size = 1 << num_bits;
    let mut table = vec![0; table_size];
    let movement_mask = init_movement_mask(sq as u8, orthogonal);
    let blocker_masks = init_blocker_masks(movement_mask);
    for &blocker_mask in &blocker_masks {
        let index = (blocker_mask.wrapping_mul(magic)) >> shift;
        let attacks = init_attack_mask(sq as u8, blocker_mask, orthogonal);
        table[index as usize] = attacks;
    }
    table
}

pub fn init_blocker_masks(movement_mask: u64) -> Vec<u64> {
    let mut move_squares = Vec::new();
    for i in 0..64 {
        if (movement_mask & 1 << i) != 0 {
            move_squares.push(i);
        }
    }
    let patterns_count = 1 << move_squares.len();
    let mut blocker_bitboards = vec![0; patterns_count];
    for pattern_index in 0..patterns_count {
        for bit_index in 0..move_squares.len() {
            let bit = (pattern_index >> bit_index) & 1;
            blocker_bitboards[pattern_index] |= (bit as u64) << move_squares[bit_index];
        }
    }
    blocker_bitboards
}


fn init_magic_masks(orthogonal: bool) -> [u64; 64] {
    let mut masks: [u64; 64] = [0; 64];
    for sq in 0..64 {
        masks[sq] = init_movement_mask(sq as u8, orthogonal);
    }
    masks
}

fn init_movement_mask(sq: u8, orthogonal: bool) -> u64 {
    let mut movement_mask = 0;
    let vectors = if orthogonal { ORTHOGONAL_VECTORS } else { DIAGONAL_VECTORS };
    for vector in vectors {
        let mut current_sq = sq as i8;
        if !is_valid_vector_offset(current_sq, vector) {
            continue;
        }
        for _ in 1..8 {
            current_sq += vector;
            if Board::is_valid_sq(current_sq as u8) && is_valid_vector_offset(current_sq, vector) {
                movement_mask |= 1 << current_sq;
            } else {
                break;
            }
        }
    }
    movement_mask
}

pub fn init_attack_mask(start_square: u8, blockers: u64, is_orthogonal: bool) -> u64 {
    let mut attack_mask = 0;
    let vectors = if is_orthogonal { ORTHOGONAL_VECTORS } else { DIAGONAL_VECTORS };
    for vector in vectors {
        let mut current_square = start_square as i8;
        for _ in 1..8 {
            if Board::is_valid_sq((current_square + vector) as u8) && is_valid_vector_offset(current_square, vector) {
                current_square += vector;
                attack_mask |= 1 << current_square;
                if (blockers & 1 << current_square) != 0 {
                    break;
                }
            } else {
                break;
            }
        }
    }
    attack_mask
}

fn is_valid_vector_offset(sq: i8, vector_offset: i8) -> bool {
    let a_file = (bits::FILE_A & 1 << sq) != 0;
    let h_file = (bits::FILE_H & 1 << sq) != 0;
    let a_file_exception = A_FILE_EXCEPTIONS.contains(&vector_offset);
    let h_file_exception = H_FILE_EXCEPTIONS.contains(&vector_offset);
    (!a_file || !a_file_exception) && (!h_file || !h_file_exception)
}

fn vec_to_u64_array_with_padding<const N: usize>(vec: Vec<u64>) -> [u64; N] {
    let mut array: [u64; N] = [0; N];
    for (i, item) in vec.into_iter().enumerate() {
        if i < N {
            array[i] = item;
        } else {
            break;
        }
    }
    array
}

