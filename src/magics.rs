/// Magic bitboard implementation, stolen from Symbeline (thank you!):
/// https://github.com/sroelants/simbelmyne/blob/main/chess/src/movegen/magics.rs

#[derive(Debug, Copy, Clone)]
pub struct MagicLookup {
    pub mask: u64,
    pub magic: u64,
    pub shift: u8,
    pub offset: u32,
}

impl MagicLookup {
    pub const fn index(&self, blockers: u64) -> usize {
        let blockers = blockers & self.mask;
        let offset = self.offset as usize;
        offset + (self.magic.wrapping_mul(blockers) >> self.shift) as usize
    }
}

pub static BISHOP_ATTACKS: [u64; 5248] = gen_bishop_attacks_table();
pub static ROOK_ATTACKS: [u64; 102400] = gen_rook_attacks_table();

const fn gen_bishop_attacks_table() -> [u64; 5248]  {
    let mut table = [0; 5248];
    let mut sq: usize = 0;

    while sq < 64 {
        let entry = BISHOP_MAGICS[sq];
        let mut subset: u64 = 0;

        let attacks = gen_bishop_attacks(sq, subset);
        let blockers = subset;
        let idx = entry.index(blockers);
        table[idx] = attacks;
        subset = subset.wrapping_sub(entry.mask) & entry.mask;

        while subset != 0 {
            let attacks = gen_bishop_attacks(sq, subset);
            let blockers = subset;
            let idx = entry.index(blockers);
            table[idx] = attacks;

            subset = subset.wrapping_sub(entry.mask) & entry.mask;
        }

        sq += 1;
    }

    table
}


const fn gen_rook_attacks_table() -> [u64; 102400] {
    let mut table = [0; 102400];
    let mut sq: usize = 0;

    while sq < 64 {
        let entry = ROOK_MAGICS[sq];
        let mut subset: u64 = 0;

        let attacks = gen_rook_attacks(sq, subset);
        let blockers = subset;
        let idx = entry.index(blockers);
        table[idx] = attacks;
        subset = subset.wrapping_sub(entry.mask) & entry.mask;

        while subset != 0 {
            let attacks = gen_rook_attacks(sq, subset);
            let blockers = subset;
            let idx = entry.index(blockers);
            table[idx] = attacks;

            subset = subset.wrapping_sub(entry.mask) & entry.mask;
        }

        sq += 1;
    }

    table
}

pub const fn gen_bishop_attacks(square: usize, blockers: u64) -> u64 {
    let mut bb: u64 = 0;
    let mut tgt = square;
    while tgt % 8 > 0 && tgt / 8 < 7 {
        tgt += 7;
        bb |= 1 << tgt;
        if blockers & (1 << tgt) > 0 { break; }
    }
    let mut tgt = square;
    while tgt % 8 < 7 && tgt / 8 < 7 {
        tgt += 9;
        bb |= 1 << tgt;
        if blockers & (1 << tgt) > 0 { break; }
    }
    let mut tgt = square;
    while tgt % 8 > 0 && tgt / 8 >= 1 {
        tgt -= 9;
        bb |= 1 << tgt;
        if blockers & (1 << tgt) > 0 { break; }
    }
    let mut tgt = square;
    while tgt % 8 < 7 && tgt / 8 >= 1 {
        tgt -= 7;
        bb |= 1 << tgt;
        if blockers & (1 << tgt) > 0 { break; }
    }
    bb
}

pub const fn gen_rook_attacks(square: usize, blockers: u64) -> u64 {
    let mut bb: u64 = 0;
    let mut tgt = square;
    while tgt / 8 < 7 {
        tgt += 8;
        bb |= 1 << tgt;
        if blockers & (1 << tgt) > 0 { break; }
    }
    let mut tgt = square;
    while tgt % 8 < 7 {
        tgt += 1;
        bb |= 1 << tgt;
        if blockers & (1 << tgt) > 0 { break; }
    }
    let mut tgt = square;
    while tgt / 8 >= 1 {
        tgt -= 8;
        bb |= 1 << tgt;
        if blockers & (1 << tgt) > 0 { break; }
    }
    let mut tgt = square;
    while tgt % 8 > 0 {
        tgt -= 1;
        bb |= 1 << tgt;
        if blockers & (1 << tgt) > 0 { break; }
    }
    bb
}

pub const BISHOP_MAGICS: [MagicLookup; 64] = [
    MagicLookup { mask: 18049651735527936,   magic: 1143543703831040,     shift: 58, offset: 0 },
    MagicLookup { mask: 70506452091904,      magic: 4616207506731991056,  shift: 59, offset: 64 },
    MagicLookup { mask: 275415828992,        magic: 41134946502311936,    shift: 59, offset: 96 },
    MagicLookup { mask: 1075975168,          magic: 9237041792476577800,  shift: 59, offset: 128 },
    MagicLookup { mask: 38021120,            magic: 2324156749898121216,  shift: 59, offset: 160 },
    MagicLookup { mask: 8657588224,          magic: 571763293431064,      shift: 59, offset: 192 },
    MagicLookup { mask: 2216338399232,       magic: 9241675675694268936,  shift: 59, offset: 224 },
    MagicLookup { mask: 567382630219776,     magic: 5764627348637487104,  shift: 58, offset: 256 },
    MagicLookup { mask: 9024825867763712,    magic: 290490981856847112,   shift: 59, offset: 320 },
    MagicLookup { mask: 18049651735527424,   magic: 9949612811531387468,  shift: 59, offset: 352 },
    MagicLookup { mask: 70506452221952,      magic: 14411527775495137280, shift: 59, offset: 384 },
    MagicLookup { mask: 275449643008,        magic: 8968983333576,        shift: 59, offset: 416 },
    MagicLookup { mask: 9733406720,          magic: 2810248375360815104,  shift: 59, offset: 448 },
    MagicLookup { mask: 2216342585344,       magic: 4899956570242188290,  shift: 59, offset: 480 },
    MagicLookup { mask: 567382630203392,     magic: 27870460605105152,    shift: 59, offset: 512 },
    MagicLookup { mask: 1134765260406784,    magic: 7391071850088456,     shift: 59, offset: 544 },
    MagicLookup { mask: 4512412933816832,    magic: 4538825104173056,     shift: 59, offset: 576 },
    MagicLookup { mask: 9024825867633664,    magic: 9293415943536772,     shift: 59, offset: 608 },
    MagicLookup { mask: 18049651768822272,   magic: 150871699947004418,   shift: 57, offset: 640 },
    MagicLookup { mask: 70515108615168,      magic: 19140332795469824,    shift: 57, offset: 768 },
    MagicLookup { mask: 2491752130560,       magic: 9226750425928040466,  shift: 57, offset: 896 },
    MagicLookup { mask: 567383701868544,     magic: 578783101256343562,   shift: 57, offset: 1024 },
    MagicLookup { mask: 1134765256220672,    magic: 2339769367535685,     shift: 59, offset: 1152 },
    MagicLookup { mask: 2269530512441344,    magic: 634437543397376,      shift: 59, offset: 1184 },
    MagicLookup { mask: 2256206450263040,    magic: 9809970291735545856,  shift: 59, offset: 1216 },
    MagicLookup { mask: 4512412900526080,    magic: 149749085925214208,   shift: 59, offset: 1248 },
    MagicLookup { mask: 9024834391117824,    magic: 144203149157335168,   shift: 57, offset: 1280 },
    MagicLookup { mask: 18051867805491712,   magic: 1226113794711822470,  shift: 55, offset: 1408 },
    MagicLookup { mask: 637888545440768,     magic: 2319635351808786434,  shift: 55, offset: 1920 },
    MagicLookup { mask: 1135039602493440,    magic: 4504149803683858,     shift: 57, offset: 2432 },
    MagicLookup { mask: 2269529440784384,    magic: 144256510283892736,   shift: 59, offset: 2560 },
    MagicLookup { mask: 4539058881568768,    magic: 9295720039971849224,  shift: 59, offset: 2592 },
    MagicLookup { mask: 1128098963916800,    magic: 65311183963955200,    shift: 59, offset: 2624 },
    MagicLookup { mask: 2256197927833600,    magic: 11604879072873940992, shift: 59, offset: 2656 },
    MagicLookup { mask: 4514594912477184,    magic: 9241404336952575106,  shift: 57, offset: 2688 },
    MagicLookup { mask: 9592139778506752,    magic: 2306476637149233664,  shift: 55, offset: 2816 },
    MagicLookup { mask: 19184279556981248,   magic: 1161092880670848,     shift: 55, offset: 3328 },
    MagicLookup { mask: 2339762086609920,    magic: 159437776683649,      shift: 57, offset: 3840 },
    MagicLookup { mask: 4538784537380864,    magic: 2310426911845777600,  shift: 59, offset: 3968 },
    MagicLookup { mask: 9077569074761728,    magic: 4611831158265692448,  shift: 59, offset: 4000 },
    MagicLookup { mask: 562958610993152,     magic: 6548700399157312,     shift: 59, offset: 4032 },
    MagicLookup { mask: 1125917221986304,    magic: 594651708866433056,   shift: 59, offset: 4064 },
    MagicLookup { mask: 2814792987328512,    magic: 4900479464858722816,  shift: 57, offset: 4096 },
    MagicLookup { mask: 5629586008178688,    magic: 4611722053211527169,  shift: 57, offset: 4224 },
    MagicLookup { mask: 11259172008099840,   magic: 70373055922705,       shift: 57, offset: 4352 },
    MagicLookup { mask: 22518341868716544,   magic: 147105895422108544,   shift: 57, offset: 4480 },
    MagicLookup { mask: 9007336962655232,    magic: 565183353192960,      shift: 59, offset: 4608 },
    MagicLookup { mask: 18014673925310464,   magic: 4522018603270400,     shift: 59, offset: 4640 },
    MagicLookup { mask: 2216338399232,       magic: 216736352758153476,   shift: 59, offset: 4672 },
    MagicLookup { mask: 4432676798464,       magic: 4611769590043772928,  shift: 59, offset: 4704 },
    MagicLookup { mask: 11064376819712,      magic: 603492796636610688,   shift: 59, offset: 4736 },
    MagicLookup { mask: 22137335185408,      magic: 6790602613589024,     shift: 59, offset: 4768 },
    MagicLookup { mask: 44272556441600,      magic: 1261029954650243328,  shift: 59, offset: 4800 },
    MagicLookup { mask: 87995357200384,      magic: 8865368375296,        shift: 59, offset: 4832 },
    MagicLookup { mask: 35253226045952,      magic: 1459249945244139533,  shift: 59, offset: 4864 },
    MagicLookup { mask: 70506452091904,      magic: 37172325652660256,    shift: 59, offset: 4896 },
    MagicLookup { mask: 567382630219776,     magic: 4541005908936713,     shift: 58, offset: 4928 },
    MagicLookup { mask: 1134765260406784,    magic: 1874625569703068161,  shift: 59, offset: 4992 },
    MagicLookup { mask: 2832480465846272,    magic: 576466254165446914,   shift: 59, offset: 5024 },
    MagicLookup { mask: 5667157807464448,    magic: 585468021360036865,   shift: 59, offset: 5056 },
    MagicLookup { mask: 11333774449049600,   magic: 9228016941049914376,  shift: 59, offset: 5088 },
    MagicLookup { mask: 22526811443298304,   magic: 5296241975597737220,  shift: 59, offset: 5120 },
    MagicLookup { mask: 9024825867763712,    magic: 576469583033045504,   shift: 59, offset: 5152 },
    MagicLookup { mask: 18049651735527936,   magic: 333846923255742504,   shift: 58, offset: 5184 }
];

pub const ROOK_MAGICS: [MagicLookup; 64] = [
    MagicLookup { mask: 282578800148862,     magic: 396334507571101697,   shift: 52, offset: 0 },
    MagicLookup { mask: 565157600297596,     magic: 18014673924829184,    shift: 53, offset: 4096 },
    MagicLookup { mask: 1130315200595066,    magic: 72076294862422104,    shift: 53, offset: 6144 },
    MagicLookup { mask: 2260630401190006,    magic: 324267970069010048,   shift: 53, offset: 8192 },
    MagicLookup { mask: 4521260802379886,    magic: 2449962758546399249,  shift: 53, offset: 10240 },
    MagicLookup { mask: 9042521604759646,    magic: 72060072234319880,    shift: 53, offset: 12288 },
    MagicLookup { mask: 18085043209519166,   magic: 36064531364987392,    shift: 53, offset: 14336 },
    MagicLookup { mask: 36170086419038334,   magic: 252206124290277632,   shift: 52, offset: 16384 },
    MagicLookup { mask: 282578800180736,     magic: 4040432697074532352,  shift: 53, offset: 20480 },
    MagicLookup { mask: 565157600328704,     magic: 9223723949339181122,  shift: 54, offset: 22528 },
    MagicLookup { mask: 1130315200625152,    magic: 13853635543349461056, shift: 54, offset: 23552 },
    MagicLookup { mask: 2260630401218048,    magic: 2324420530558599200,  shift: 54, offset: 24576 },
    MagicLookup { mask: 4521260802403840,    magic: 141046734390272,      shift: 54, offset: 25600 },
    MagicLookup { mask: 9042521604775424,    magic: 422762254443520,      shift: 54, offset: 26624 },
    MagicLookup { mask: 18085043209518592,   magic: 2387189294696506624,  shift: 54, offset: 27648 },
    MagicLookup { mask: 36170086419037696,   magic: 4644343558193408,     shift: 53, offset: 28672 },
    MagicLookup { mask: 282578808340736,     magic: 1170430131961992,     shift: 53, offset: 30720 },
    MagicLookup { mask: 565157608292864,     magic: 2603714178464129024,  shift: 54, offset: 32768 },
    MagicLookup { mask: 1130315208328192,    magic: 3518988306898944,     shift: 54, offset: 33792 },
    MagicLookup { mask: 2260630408398848,    magic: 360297866348986624,   shift: 54, offset: 34816 },
    MagicLookup { mask: 4521260808540160,    magic: 9262217782933407776,  shift: 54, offset: 35840 },
    MagicLookup { mask: 9042521608822784,    magic: 9429961542730240,     shift: 54, offset: 36864 },
    MagicLookup { mask: 18085043209388032,   magic: 1130298087641094,     shift: 54, offset: 37888 },
    MagicLookup { mask: 36170086418907136,   magic: 10421171212289025,    shift: 53, offset: 38912 },
    MagicLookup { mask: 282580897300736,     magic: 5800636596004855808,  shift: 53, offset: 40960 },
    MagicLookup { mask: 565159647117824,     magic: 18331076111912960,    shift: 54, offset: 43008 },
    MagicLookup { mask: 1130317180306432,    magic: 2450241050951819272,  shift: 54, offset: 44032 },
    MagicLookup { mask: 2260632246683648,    magic: 2305959351820802,     shift: 54, offset: 45056 },
    MagicLookup { mask: 4521262379438080,    magic: 326511524888183810,   shift: 54, offset: 46080 },
    MagicLookup { mask: 9042522644946944,    magic: 1127000492343360,     shift: 54, offset: 47104 },
    MagicLookup { mask: 18085043175964672,   magic: 72058710730475522,    shift: 54, offset: 48128 },
    MagicLookup { mask: 36170086385483776,   magic: 1315192112148418820,  shift: 53, offset: 49152 },
    MagicLookup { mask: 283115671060736,     magic: 141562147242016,      shift: 53, offset: 51200 },
    MagicLookup { mask: 565681586307584,     magic: 54050343436165120,    shift: 54, offset: 53248 },
    MagicLookup { mask: 1130822006735872,    magic: 290517362560471040,   shift: 54, offset: 54272 },
    MagicLookup { mask: 2261102847592448,    magic: 6773010962843648,     shift: 54, offset: 55296 },
    MagicLookup { mask: 4521664529305600,    magic: 422367092279296,      shift: 54, offset: 56320 },
    MagicLookup { mask: 9042787892731904,    magic: 4920746711120353284,  shift: 54, offset: 57344 },
    MagicLookup { mask: 18085034619584512,   magic: 2814767248965912,     shift: 54, offset: 58368 },
    MagicLookup { mask: 36170077829103616,   magic: 9664725504745275969,  shift: 53, offset: 59392 },
    MagicLookup { mask: 420017753620736,     magic: 108227403458838528,   shift: 53, offset: 61440 },
    MagicLookup { mask: 699298018886144,     magic: 1161946296806948864,  shift: 54, offset: 63488 },
    MagicLookup { mask: 1260057572672512,    magic: 72080684319047744,    shift: 54, offset: 64512 },
    MagicLookup { mask: 2381576680245248,    magic: 563225169100808,      shift: 54, offset: 65536 },
    MagicLookup { mask: 4624614895390720,    magic: 9374242796992528405,  shift: 54, offset: 66560 },
    MagicLookup { mask: 9110691325681664,    magic: 9804900541598400516,  shift: 54, offset: 67584 },
    MagicLookup { mask: 18082844186263552,   magic: 36899680323633153,    shift: 54, offset: 68608 },
    MagicLookup { mask: 36167887395782656,   magic: 18160262709249,       shift: 53, offset: 69632 },
    MagicLookup { mask: 35466950888980736,   magic: 441987183975465472,   shift: 53, offset: 71680 },
    MagicLookup { mask: 34905104758997504,   magic: 170222060954452032,   shift: 54, offset: 73728 },
    MagicLookup { mask: 34344362452452352,   magic: 9150136340976128,     shift: 54, offset: 74752 },
    MagicLookup { mask: 33222877839362048,   magic: 35770907886080,       shift: 54, offset: 75776 },
    MagicLookup { mask: 30979908613181440,   magic: 1125934275330176,     shift: 54, offset: 76800 },
    MagicLookup { mask: 26493970160820224,   magic: 563019076338176,      shift: 54, offset: 77824 },
    MagicLookup { mask: 17522093256097792,   magic: 563002852508160,      shift: 54, offset: 78848 },
    MagicLookup { mask: 35607136465616896,   magic: 1162073852289769984,  shift: 53, offset: 79872 },
    MagicLookup { mask: 9079539427579068672, magic: 422230469181761,      shift: 52, offset: 81920 },
    MagicLookup { mask: 8935706818303361536, magic: 4683762314933977122,  shift: 53, offset: 86016 },
    MagicLookup { mask: 8792156787827803136, magic: 1441996340860027971,  shift: 53, offset: 88064 },
    MagicLookup { mask: 8505056726876686336, magic: 4503669488247041,     shift: 53, offset: 90112 },
    MagicLookup { mask: 7930856604974452736, magic: 649081365551645714,   shift: 53, offset: 92160 },
    MagicLookup { mask: 6782456361169985536, magic: 3659218049826819,     shift: 53, offset: 94208 },
    MagicLookup { mask: 4485655873561051136, magic: 90353538491745281,    shift: 53, offset: 96256 },
    MagicLookup { mask: 9115426935197958144, magic: 140896737722434,      shift: 52, offset: 98304 }
];