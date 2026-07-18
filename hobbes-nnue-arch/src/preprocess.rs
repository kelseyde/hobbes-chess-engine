use crate::{Network, UntransposedNetwork, L1_SIZE, L2_SIZE, L3_SIZE, OUTPUT_BUCKET_COUNT};

pub struct PermuteConfig {
    pub needs_permuting: bool,
    pub order: &'static [u8],
}

const L0_ACTIVATIONS: [usize; L1_SIZE / 2] = [
    180478, 301608, 83188, 1478109, 1150846, 1718920, 25122, 155629, 650369, 1176588, 3554369,
    131794, 236391, 471054, 4294781, 437998, 354946, 606517, 491821, 1139870, 221513, 1197180,
    286062, 781246, 626917, 1264866, 316869, 1100696, 979797, 1012716, 153129, 135284, 2170604,
    2474919, 9496385, 547136, 130344, 1132991, 1682961, 1723251, 133917, 85709, 194529, 1043445,
    1584904, 751667, 433820, 874381, 4892251, 1290368, 1208402, 631612, 1388780, 40080, 50152,
    488256, 116389, 9144, 347369, 2916685, 1244768, 1095777, 645905, 65912, 2359508, 450182, 22294,
    2433524, 1578335, 1024120, 2637340, 788030, 467598, 985162, 2386148, 1060275, 515652, 2400193,
    1655854, 8042685, 195700, 1356160, 138152, 197725, 2521807, 2219589, 1128483, 489314, 1528682,
    5261896, 1575053, 99049, 815048, 2747440, 10636927, 951387, 1552328, 1605979, 4260326, 4229180,
    3037641, 1718154, 882812, 3002672, 632242, 497590, 1352251, 1306488, 927515, 1327228, 510020,
    373746, 2538980, 1659166, 1771091, 793816, 1702702, 1969497, 1072798, 700857, 6248913, 3597177,
    463759, 687383, 5456156, 7932100, 1741396, 780793, 1523031, 614703, 1286976, 1888996, 2435022,
    597801, 330627, 364871, 184041, 1607637, 552477, 3135792, 2096358, 459307, 2883750, 7791260,
    58288, 1028103, 1671747, 2586796, 110791, 409353, 2936362, 1131555, 470845, 625491, 106337,
    215131, 2855571, 2569759, 3566454, 879467, 116191, 802489, 505506, 1458219, 2242884, 412524,
    1421461, 590588, 904224, 1295422, 139837, 3095689, 349399, 1067586, 461498, 2334388, 8620299,
    490286, 1162873, 1679601, 933587, 1321022, 1700009, 318801, 2945427, 1339942, 47183, 1661372,
    3233367, 1096263, 5014495, 5283598, 1765841, 2592439, 1060054, 512323, 696045, 3811361, 217358,
    455353, 407145, 654647, 1065888, 595948, 1139961, 434959, 90619, 281882, 632379, 675648,
    1009789, 427603, 891269, 969091, 150217, 627199, 4622877, 2008415, 961936, 944273, 357740,
    552164, 1543521, 635940, 520822, 122532, 1259984, 470360, 431945, 1274846, 403545, 277292,
    1479979, 1382647, 2610106, 3511913, 929467, 811526, 1593502, 2191047, 1349553, 913711, 125662,
    1184042, 1451602, 1181481, 84739, 2597746, 2682918, 693134, 1202968, 215193, 1467603, 861393,
    350622, 167393, 1379525, 658709, 1844989, 2608066, 174040, 21066, 7899385, 650287, 895773,
    590702, 779011, 646420, 2305652, 7328422, 885205, 231650, 2254572, 2293010, 994429, 968081,
    1646402, 737608, 1994358, 107622, 1587360, 6738772, 817921, 1598993, 318385, 579118, 433787,
    222204, 356394, 777862, 441729, 390050, 189473, 5807664, 3782467, 870442, 2092688, 824251,
    3907319, 191604, 2012508, 670535, 765118, 330947, 186811, 299697, 2100888, 84851, 2193566,
    154991, 706271, 7748458, 458473, 703261, 1020883, 1282850, 65689, 2248000, 3971064, 1305850,
    1972522, 808324, 3534232, 2507590, 1167072, 986730, 296280, 289661, 1131034, 668323, 257041,
    3622071, 46974, 1434906, 919054, 594017, 1555950, 3178877, 303352, 509417, 445429, 4307297,
    2478785, 1273571, 379850, 6911117, 340990, 2703880, 439443, 137353, 441334, 188381, 423260,
    563201, 256731, 683064, 308926, 395383, 2631928, 490577, 967940, 905082, 4116632, 170610,
    583887, 2459556, 1042221, 117351, 833023, 610826, 1125418, 380126, 1049137, 330773, 2601189,
    1564675, 2975985, 326486, 1822241, 2449269, 2002474, 451993, 422672, 2986865, 954127, 496439,
    81290, 1065638, 2478421, 39557, 3129670, 2720021, 2201660, 1496751, 1237616, 2376977, 2494637,
    1200496, 357440, 2847787, 172420, 854359, 649852, 1618725, 2811024, 693275, 795966, 132032,
    540947, 1549921, 2019547, 808460, 2114178, 846149, 673276, 380460, 1838661, 2030289, 144010,
    2521528, 227807, 2520834, 271558, 485822, 908982, 703958, 1363576, 388650, 4616747, 258420,
    688333, 1348484, 2464191, 2980952, 1220824, 538557, 1597161, 1313797, 115755, 2670283, 231023,
    1324630, 1091253, 1910005, 501481, 1730742, 253478, 183598, 1227268, 256762, 571382, 995931,
    220744, 776433, 726857, 487724, 220883, 96429, 323079, 840255, 1440222, 815832, 602065, 540361,
    638235, 60366, 3220618, 532696, 821904, 671381, 1408365, 287111, 1733154, 280032, 202871,
    698377, 2951019, 4336848, 842661, 595507, 527372, 493341, 825457, 511388, 319792, 3747622,
    893221, 1213695, 2435648, 237155, 690881, 44668, 1400599, 267732, 390052, 623130, 453666,
    362708, 127112, 828631, 2190564, 1327189, 297536, 558269, 1780345, 1840427, 273920, 2454875,
    266097, 168881, 7449201, 1977199, 508792, 3886561, 856485, 157198, 963119, 536875, 390016,
    6004149, 133273, 2208089, 640299, 2901244, 489677, 265187, 493048, 378879, 2220785, 383790,
    6218871, 7573309, 730141, 851663, 8676780, 1065766, 996388, 3395075, 289779, 728039, 411150,
    1416693, 1204951, 215857, 2669627, 1918036, 150928, 1633650, 1866905, 2495956, 257154, 441672,
    394799, 4484996, 649338, 536264, 1578528, 2683926, 274509, 348431, 1058796, 203819, 1569214,
    7067028, 495498, 453503, 1580985, 2216935, 463111, 643040, 1342321, 110111, 525270, 1135572,
    4565581, 1351957, 1743859, 1459602, 340779, 1898612, 360243, 242986, 1409341, 735788, 687050,
    383567, 645863, 851640, 234923, 1355219, 3059436, 1295014, 468659, 1491869, 233721, 471143,
    6364033, 4208496, 445788, 3104147, 510677, 228282, 2471200, 2707480, 1485440, 138593, 381664,
    747425, 10303504, 768851, 3042711, 1226730, 184891, 1466313, 1198249, 1363500, 239358, 3668839,
    1272079, 2663687, 881839, 1680258, 1498210, 706936, 498789, 441370, 9036627, 1205610, 1296769,
    1449700, 441345, 151839, 1623026, 222436, 684535, 401900, 117797, 8349618, 962173, 1803180,
    607286, 32245, 506916, 85386, 151747, 351981, 1071414, 159377, 1296290, 753806, 877442, 82120,
    73758, 65300, 2172225, 732274, 269724, 1770890, 1399790, 798962, 2022360, 680603, 2621685,
    8691631, 1071554, 510324, 45861, 5612578, 718932, 804063, 293555, 2825346, 668658, 597432,
    10039579, 1384219, 1390211, 508357, 2973239, 531519, 1015535, 193912, 198042, 727738, 516938,
    363012, 1080917, 676851, 9002435, 23025, 489403, 1168169, 6889217, 627790, 1062705, 9474221,
    657289, 946557, 155039, 468187, 1167798, 749395, 212738, 2389016, 975875, 789785, 139971,
    536092, 32998, 0, 100033, 1773040, 777554, 1490895, 1540933, 1466925, 185477, 3746684, 694536,
    1438583, 202196, 461775, 1228453, 625623, 43963, 2857183, 98765, 1042965, 1124237, 9239, 131901,
    230224, 1207004, 437404, 269884, 605284, 500050, 959497, 490595, 4459707, 167340, 842896,
    2761779, 596317, 4829992, 482045, 1168569, 497743, 432356, 321520, 784524, 308539, 4449958,
    171283, 733664, 291590, 5048899, 579935, 533633, 198161, 1126343, 484408, 348416, 683152,
    2054769, 4464183, 190780, 116641
];

#[cfg(target_feature = "avx512f")]
static ORDER: &[u8] = &[0, 2, 4, 6, 1, 3, 5, 7];

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
static ORDER: &[u8] = &[0, 2, 1, 3];

#[cfg(not(any(target_feature = "avx512f", target_feature = "avx2")))]
static ORDER: &[u8] = &[];

pub const fn permute_config() -> PermuteConfig {
    let needs_permuting = cfg!(target_feature = "avx512f") || cfg!(target_feature = "avx2");
    PermuteConfig {
        needs_permuting,
        order: ORDER,
    }
}

/// Convert an `UntransposedNetwork` (the output format from Bullet) into a `Network` (the optimal
/// format for inference).
///
/// This performs the following transformations:
/// 1. Repermutes L0 weights and biases so the most-activated neurons come first
/// 2. Permutes the L0 weights and biases to cancel out the cross-lane behaviour of packus.
/// 3. Transposes L1 weights: src[input][bucket][output] -> dst[bucket][output][input]
/// 4. Reorders L2 weights: src[input][bucket][output] -> dst[bucket][input][output]
/// 5. Reorders L3 weights: src[input][bucket] -> dst[bucket][input]
pub fn process_network(src: &UntransposedNetwork, dst: &mut Network) {
    let repermute = compute_repermute_indices();

    repermute_l0_biases(&mut dst.l0_biases, &src.l0_biases, &repermute);

    for (dst_bucket, src_bucket) in dst.l0_psq_weights.iter_mut().zip(src.l0_psq_weights.iter()) {
        repermute_l0_bucket(dst_bucket, src_bucket, &repermute);
    }

    let config = permute_config();
    if config.needs_permuting {
        let order = config.order;
        let num_chunks = order.len();

        let chunk_size: usize = 8; // 128 bits = 8 i16 values
        let block_size = num_chunks * chunk_size;

        // Permute L0 weights per bucket.
        for bucket in dst.l0_psq_weights.iter_mut() {
            permute_i16s(bucket, order, chunk_size, block_size);
        }
        // Permute L0 biases.
        permute_i16s(&mut dst.l0_biases, order, chunk_size, block_size);
    }

    for bucket in 0..OUTPUT_BUCKET_COUNT {
        for (tgt_input_idx, &src_input_idx) in repermute.iter().enumerate() {
            for half in 0..2 {
                let tgt_idx = tgt_input_idx + half * (L1_SIZE / 2);
                let src_idx = src_input_idx + half * (L1_SIZE / 2);
                let in_block = tgt_idx / 4;
                let k = tgt_idx % 4;
                for output_idx in 0..L2_SIZE {
                    dst.l1_weights[bucket][in_block][output_idx * 4 + k] =
                        src.l1_weights[src_idx][bucket][output_idx];
                }
            }
        }
    }

    for input_idx in 0..(L2_SIZE * 2) {
        for bucket in 0..OUTPUT_BUCKET_COUNT {
            for output_idx in 0..L3_SIZE {
                dst.l2_weights[bucket][input_idx][output_idx] =
                    src.l2_weights[input_idx][bucket][output_idx];
            }
        }
    }

    for input_idx in 0..L3_SIZE {
        for bucket in 0..OUTPUT_BUCKET_COUNT {
            dst.l3_weights[bucket][input_idx] = src.l3_weights[input_idx][bucket][0];
        }
    }

    unsafe {
        std::ptr::copy_nonoverlapping(&src.l1_biases, &mut dst.l1_biases, 1);
        std::ptr::copy_nonoverlapping(&src.l2_biases, &mut dst.l2_biases, 1);
        std::ptr::copy_nonoverlapping(&src.l3_biases, &mut dst.l3_biases, 1);
    }
}

/// Compute the repermutation indices for sparsity optimisation.
fn compute_repermute_indices() -> [usize; L1_SIZE / 2] {
    let mut indices: [usize; L1_SIZE / 2] = std::array::from_fn(|i| i);
    indices.sort_by(|&a, &b| L0_ACTIVATIONS[b].cmp(&L0_ACTIVATIONS[a]));
    indices
}

/// Re-permute the L0 biases so that the most-activated neurons come first.
fn repermute_l0_biases(dst: &mut [i16; L1_SIZE], src: &[i16; L1_SIZE], indices: &[usize; L1_SIZE / 2]) {
    for (tgt, &src_idx) in indices.iter().enumerate() {
        dst[tgt] = src[src_idx];
        dst[tgt + L1_SIZE / 2] = src[src_idx + L1_SIZE / 2];
    }
}

/// Re-permute a single L0 weight bucket for sparsity.
fn repermute_l0_bucket(dst: &mut [i16], src: &[i16], indices: &[usize; L1_SIZE / 2]) {
    let input_features = src.len() / L1_SIZE;
    for feature in 0..input_features {
        let base = feature * L1_SIZE;
        for (tgt, &src_idx) in indices.iter().enumerate() {
            dst[base + tgt] = src[base + src_idx];
            dst[base + tgt + L1_SIZE / 2] = src[base + src_idx + L1_SIZE / 2];
        }
    }
}

// Permute a flat slice of i16 values in-place
fn permute_i16s(data: &mut [i16], order: &[u8], chunk_size: usize, block_size: usize) {
    let num_chunks = order.len();
    let mut temp = vec![0i16; block_size];
    for block_start in (0..data.len()).step_by(block_size) {
        temp.copy_from_slice(&data[block_start..block_start + block_size]);
        for dst_chunk in 0..num_chunks {
            let src_chunk = order[dst_chunk] as usize;
            let dst_offset = block_start + dst_chunk * chunk_size;
            let src_offset = src_chunk * chunk_size;
            data[dst_offset..dst_offset + chunk_size]
                .copy_from_slice(&temp[src_offset..src_offset + chunk_size]);
        }
    }
}
