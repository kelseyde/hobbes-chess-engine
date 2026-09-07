use crate::tunable_params;
use crate::tunable_arrays;

#[rustfmt::skip]
tunable_params! {
    asp_delta                    = 10, 4..=36,             true;
    asp_min_depth                = 4, 0..=8,               false;
    asp_alpha_widening_factor    = 225, 50..=400,          true;
    asp_beta_widening_factor     = 272, 50..=400,          true;
    asp_prev_score_div           = 10360, 8000..=12000,    true;
    rfp_max_depth                = 8, 6..=12,              false;
    rfp_base                     = 11, -50..=50,           true;
    rfp_scale                    = 62, 40..=100,           true;
    rfp_improving_scale          = 56, 40..=100,           true;
    rfp_opp_worsening_scale      = 12, 0..=50,             true;
    rfp_tt_move_noisy_scale      = 4, 0..=70,              true;
    rfp_lerp_factor              = 38, 0..=100,            true;
    razor_base                   = 280, 200..=500,         true;
    razor_scale                  = 248, 100..=400,         true;
    nmp_min_depth                = 3, 0..=8,               false;
    nmp_margin                   = 37, 0..=80,             true;
    nmp_red_base                 = 5112, 3072..=8192,      true;
    nmp_red_depth_mult           = 338, 128..=512,         true;
    nmp_red_eval_mult            = 596, 256..=1024,        true;
    nmp_red_eval_max             = 1102, 512..=2048,       true;
    nmp_red_div                  = 125, 64..=256,          true;
    iir_min_depth                = 4, 1..=10,              false;
    iir_tt_depth_offset          = 4, 1..=6,               false;
    cutnode_red_min_depth        = 8, 4..=12,              false;
    cutnode_red_tt_offset        = 4, 1..=6,               false;
    pc_base                      = 140, 100..=400,         true;
    pc_scale                     = 21, 0..=40,             true;
    pc_ttpv_margin               = 61, 0..=100,            true;
    pc_cutnode_margin            = 40, 0..=100,            true;
    pc_improving_margin          = 63, 0..=100,            true;
    pc_max                       = 513, 300..=700,         true;
    pc_tt_depth_offset           = 2, 0..=4,               false;
    hindsight_ext_min_depth      = 1, 1..=5,               false;
    hindsight_ext_min_reduction  = 3, 1..=5,               false;
    hindsight_ext_eval_diff      = -17, -50..=50,          true;
    hindsight_red_min_depth      = 2, 1..=5,               false;
    hindsight_red_min_reduction  = 1, 1..=5,               false;
    hindsight_red_eval_diff      = 51, 0..=120,            true;
    hindsight_hist_mult          = 9, 0..=20,              true;
    hindsight_hist_min           = -64, -100..=0,          true;
    hindsight_hist_max           = 214, 75..=300,          true;
    fp_max_depth                 = 8, 4..=10,              false;
    fp_base                      = 164, 50..=250,          true;
    fp_scale                     = 96, 50..=200,           true;
    fp_pv_node                   = 3, -50..=100,           true;
    fp_movecount_mult            = 4, 2..=8,               false;
    fp_history_divisor           = 64, 64..=256,           true;
    fp_killer                    = 19, 0..=100,            true;
    fp_tt_upper                  = 25, 0..=100,            true;
    lmp_max_depth                = 8, 6..=10,              false;
    lmp_factor0_base             = 2497, 2000..=3000,      true;
    lmp_factor0_scale            = 132, 50..=200,          true;
    lmp_factor1_base             = 887, 500..=1500,        true;
    lmp_factor1_scale            = 81, 50..=200,           true;
    lmp_improvement_min          = -121, -300..=0,         true;
    lmp_improvement_max          = 204, 0..=500,           true;
    hp_max_depth                 = 4, 4..=8,               false;
    hp_scale                     = -2347, -3072..=-1024,   true;
    bnp_max_depth                = 6, 4..=10,              false;
    bnp_scale                    = 147, 64..=256,          true;
    see_max_depth                = 8, 6..=10,              false;
    see_quiet_mult1              = -16, -24..=-8,          true;
    see_quiet_mult2              = 53, 32..=80,            true;
    see_quiet_offset             = 27, 0..=40,             true;
    see_noisy_mult1              = -8, -16..=0,            true;
    see_noisy_mult2              = 37, 16..=48,            true;
    see_noisy_offset             = 10, 0..=30,             true;
    see_quiet_history_div        = 184, 164..=388,         true;
    see_noisy_history_div        = 232, 164..=388,         true;
    see_quiet_ttpv_scale         = 26, 0..=36,             true;
    see_noisy_ttpv_scale         = 22, 0..=36,             true;
    se_min_depth                 = 6, 6..=10,              false;
    se_tt_depth_offset           = 3, 1..=6,               false;
    se_depth_offset              = 1, 0..=3,               false;
    se_depth_divisor             = 2, 1..=4,               false;
    se_beta_quiet_base           = 58, 0..=100,            true;
    se_beta_quiet_scale          = 19, 16..=48,            true;
    se_beta_quiet_div            = 53, 40..=100,           true;
    se_beta_history_div          = 981, 512..=2048,       true;
    se_dext_quiet_margin         = 10, 0..=30,              true;
    se_text_quiet_margin         = 50, 20..=120,           true;
    se_beta_noisy_base           = 58, 0..=100,            true;
    se_beta_noisy_scale          = 17, 16..=48,            true;
    se_beta_noisy_div            = 54, 40..=100,           true;
    se_dext_noisy_margin         = 7, 0..=30,             true;
    se_text_noisy_margin         = 70, 20..=120,           true;
    se_dext_pv_margin            = 96, 0..=200,            true;
    se_text_pv_margin            = 101, 0..=200,           true;
    ldse_max_depth               = 7, 1..=12,              false;
    ldse_margin                  = 25, 0..=80,             true;
    ldse_dext_tt_depth_offset    = 3, 1..=5,               false;
    ldse_dext_margin             = 37, 20..=80,            true;
    lmr_min_depth                = 2, 1..=5,               false;
    lmr_min_moves                = 2, 1..=4,               false;
    lmr_quiet_base               = 89, 50..=150,           true;
    lmr_quiet_div                = 300, 250..=400,         true;
    lmr_noisy_base               = 96, 50..=150,           true;
    lmr_noisy_div                = 295, 250..=400,         true;
    lmr_complexity_margin        = 94, 0..=300,            true;
    lmr_complex                  = 217, 0..=2048,          true;
    lmr_good_noisy               = 909, 0..=2048,          true;
    lmr_bad_noisy                = 756, 0..=2048,          true;
    lmr_fail_highs               = 6, 0..=2048,           true;
    lmr_quiet_see                = 1284, 0..=2048,         true;
    lmr_se_mult                  = 552, 256..=1024,        true;
    lmr_se_offset                = 221, 0..=1024,          true;
    lmr_se_div                   = 99, 64..=256,          true;
    lmr_se_max                   = 2547, 0..=3072,         true;
    lmr_hist_offset              = 1222, -2048..=2048,     true;
    lmr_hist_divisor             = 15798, 8192..=32768,    true;
    lmr_mvv_divisor              = 3, 1..=5,               true;
    lmr_deeper_base              = 20, 0..=100,            true;
    lmr_deeper_scale             = 540, 350..=600,         true;
    lmr_deeper_div               = 166, 64..=256,          true;
    lmr_shallower_base           = 1034, 512..=2048,        true;
    lmr_shallower_scale          = 1035, 512..=2048,       true;
    lmr_shallower_div            = 1076, 512..=2048,       true;
    lmr_even_deeper_margin       = 514, 0..=1000,          true;
    lmr_cont_1_bonus_scale       = 225, 80..=280,          true;
    lmr_cont_1_bonus_offset      = 151, 0..=200,           true;
    lmr_cont_1_bonus_max         = 1183, 1000..=1600,      true;
    lmr_cont_1_malus_scale       = 199, 80..=280,          true;
    lmr_cont_1_malus_offset      = 92, 0..=200,            true;
    lmr_cont_1_malus_max         = 1301, 1000..=1600,      true;
    lmr_cont_2_bonus_scale       = 170, 80..=280,          true;
    lmr_cont_2_bonus_offset      = 120, 0..=200,           true;
    lmr_cont_2_bonus_max         = 1053, 1000..=1600,      true;
    lmr_cont_2_malus_scale       = 199, 80..=280,          true;
    lmr_cont_2_malus_offset      = 60, 0..=200,            true;
    lmr_cont_2_malus_max         = 1223, 1000..=1600,      true;
    alpha_raise_min_depth        = 2, 0..=6,               false;
    alpha_raise_max_depth        = 12, 8..=16,             false;
    quiet_hist_bonus_scale       = 219, 80..=280,          true;
    quiet_hist_bonus_offset      = 79, 0..=200,            true;
    quiet_hist_cutnode_offset    = 81, 0..=200,            true;
    quiet_hist_ttmove_bonus      = 72, 0..=200,            true;
    quiet_hist_capture_mult      = 6, 0..=50,              true;
    quiet_hist_bonus_max         = 1114, 800..=1600,       true;
    quiet_hist_malus_scale       = 78, 64..=280,           true;
    quiet_hist_malus_offset      = 12, 0..=200,            true;
    quiet_hist_ttmove_malus      = 73, 0..=200,            true;
    quiet_hist_malus_max         = 1410, 1000..=1600,      true;
    quiet_fact_bonus_scale       = 234, 80..=320,          true;
    quiet_fact_bonus_offset      = 31, 0..=200,            true;
    quiet_fact_cutnode_offset    = 70, 0..=200,            true;
    quiet_fact_ttmove_bonus      = 59, 0..=200,            true;
    quiet_fact_capture_mult      = 9, 0..=50,              true;
    quiet_fact_bonus_max         = 1066, 800..=1600,       true;
    quiet_fact_malus_scale       = 86, 64..=280,           true;
    quiet_fact_malus_offset      = 11, 0..=200,            true;
    quiet_fact_ttmove_malus      = 83, 0..=200,            true;
    quiet_fact_malus_max         = 1338, 1000..=1600,      true;
    quiet_hist_lerp_factor       = 41, 0..=100,            true;
    capt_hist_lerp_factor        = 27, 0..=100,            true;
    capt_hist_bonus_scale        = 226, 80..=360,          true;
    capt_hist_bonus_offset       = 45, 0..=200,            true;
    capt_hist_ttmove_bonus       = 71, 0..=200,            true;
    capt_hist_bonus_max          = 1522, 1000..=1800,      true;
    capt_hist_malus_scale        = 168, 80..=280,          true;
    capt_hist_malus_offset       = 73, 0..=200,            true;
    capt_hist_ttmove_malus       = 51, 0..=200,            true;
    capt_hist_malus_max          = 1323, 1000..=1600,      true;
    cont_hist_1_bonus_scale      = 134, 80..=280,          true;
    cont_hist_1_bonus_offset     = 195, 0..=300,           true;
    cont_hist_1_cutnode_offset   = 61, 0..=200,            true;
    cont_hist_1_ttmove_bonus     = 63, 0..=200,            true;
    cont_hist_1_capture_mult     = 20, 0..=100,            true;
    cont_hist_1_bonus_max        = 933, 800..=1600,        true;
    cont_hist_1_malus_scale      = 112, 80..=280,          true;
    cont_hist_1_malus_offset     = 99, 0..=200,            true;
    cont_hist_1_ttmove_malus     = 80, 0..=200,            true;
    cont_hist_1_malus_max        = 1064, 800..=1600,       true;
    cont_hist_2_bonus_scale      = 111, 80..=280,          true;
    cont_hist_2_bonus_offset     = 200, 0..=300,           true;
    cont_hist_2_cutnode_offset   = 57, 0..=200,            true;
    cont_hist_2_ttmove_bonus     = 51, 0..=200,            true;
    cont_hist_2_capture_mult     = 16, 0..=100,            true;
    cont_hist_2_bonus_max        = 981, 800..=1600,       true;
    cont_hist_2_malus_scale      = 89, 80..=280,           true;
    cont_hist_2_malus_offset     = 82, 0..=200,            true;
    cont_hist_2_ttmove_malus     = 95, 0..=200,            true;
    cont_hist_2_malus_max        = 1025, 800..=1600,       true;
    from_hist_bonus_scale        = 192, 80..=280,          true;
    from_hist_bonus_offset       = 94, 0..=200,            true;
    from_hist_bonus_max          = 948, 800..=1600,        true;
    from_hist_malus_scale        = 220, 80..=280,          true;
    from_hist_malus_offset       = 85, 0..=200,            true;
    from_hist_malus_max          = 1099, 800..=1600,       true;
    to_hist_bonus_scale          = 197, 80..=300,          true;
    to_hist_bonus_offset         = 86, 0..=200,            true;
    to_hist_bonus_max            = 931, 800..=1600,        true;
    to_hist_malus_scale          = 193, 80..=280,          true;
    to_hist_malus_offset         = 65, 0..=200,            true;
    to_hist_malus_max            = 1004, 800..=1600,       true;
    pcm_bonus_scale              = 219, 80..=300,          true;
    pcm_bonus_offset             = 77, 0..=200,            true;
    pcm_bonus_max                = 1383, 1000..=1600,      true;
    qs_capt_hist_bonus_scale     = 249, 80..=360,          true;
    qs_capt_hist_bonus_offset    = 3, 0..=200,             true;
    qs_capt_hist_bonus_max       = 1435, 1000..=1600,      true;
    qs_capt_hist_malus_scale     = 150, 80..=280,          true;
    qs_capt_hist_malus_offset    = 93, 0..=200,            true;
    qs_capt_hist_malus_max       = 1185, 1000..=1600,      true;
    corr_pawn_bonus_mult         = 219, 100..=300,         true;
    corr_pawn_bonus_div          = 87, 64..=256,           true;
    corr_pawn_bonus_min          = -4127, -6000..=-2000,   true;
    corr_pawn_bonus_max          = 2834, 2000..=6000,      true;
    corr_nonpawn_bonus_mult      = 217, 100..=300,         true;
    corr_nonpawn_bonus_div       = 83, 64..=256,           true;
    corr_nonpawn_bonus_min       = -4145, -6000..=-2000,   true;
    corr_nonpawn_bonus_max       = 3013, 2000..=6000,      true;
    corr_major_bonus_mult        = 226, 100..=360,         true;
    corr_major_bonus_div         = 99, 64..=256,           true;
    corr_major_bonus_min         = -3957, -6000..=-2000,   true;
    corr_major_bonus_max         = 3482, 2000..=6000,      true;
    corr_minor_bonus_mult        = 198, 100..=300,         true;
    corr_minor_bonus_div         = 99, 64..=256,           true;
    corr_minor_bonus_min         = -4452, -6000..=-2000,   true;
    corr_minor_bonus_max         = 3597, 2000..=6000,      true;
    corr_cont1_bonus_mult        = 199, 100..=300,         true;
    corr_cont1_bonus_div         = 95, 64..=256,           true;
    corr_cont1_bonus_min         = -4317, -6000..=-2000,   true;
    corr_cont1_bonus_max         = 3281, 2000..=6000,      true;
    corr_cont2_bonus_mult        = 206, 100..=300,         true;
    corr_cont2_bonus_div         = 114, 64..=256,          true;
    corr_cont2_bonus_min         = -4276, -6000..=-2000,   true;
    corr_cont2_bonus_max         = 2980, 2000..=6000,      true;
    corr_pawn_weight             = 71, 0..=200,            true;
    corr_non_pawn_weight         = 69, 0..=200,            true;
    corr_major_weight            = 99, 0..=200,            true;
    corr_minor_weight            = 64, 0..=200,            true;
    corr_cont1_weight            = 101, 0..=200,           true;
    corr_cont2_weight            = 129, 0..=200,           true;
    see_value_pawn_pruning       = 100, 50..=150,           true;
    see_value_knight_pruning     = 268, 200..=500,         true;
    see_value_bishop_pruning     = 307, 200..=500,         true;
    see_value_rook_pruning       = 540, 400..=700,         true;
    see_value_queen_pruning      = 966, 800..=1200,        true;
    see_value_pawn_ordering      = 103, 50..=150,          true;
    see_value_knight_ordering    = 302, 200..=500,         true;
    see_value_bishop_ordering    = 368, 200..=500,         true;
    see_value_rook_ordering      = 543, 400..=700,         true;
    see_value_queen_ordering     = 1003, 800..=1200,       true;
    scale_value_pawn             = 19, 0..=250,            true;
    scale_value_knight           = 478, 300..=500,         true;
    scale_value_bishop           = 429, 300..=500,         true;
    scale_value_rook             = 614, 500..=700,         true;
    scale_value_queen            = 1267, 1000..=1400,      true;
    material_scaling_base        = 13663, 10000..=40000,   true;
    qs_futility_threshold        = 181, 80..=250,          true;
    qs_see_threshold             = -89, -200..=100,        true;
    qs_stand_pat_lerp_factor     = 50, 0..=100,            true;
    qs_fail_high_lerp_factor     = 50, 0..=100,            true;
    movepick_see_divisor         = 45, 30..=60,            true;
    movepick_see_offset          = 121, 80..=200,          true;
    movepick_mvv_scale           = 2061, 1024..=4096,      true;
    score_stability_threshold    = 16, 4..=24,             true;
    tm_soft_base                 = 24, 10..=60,            true;
    tm_soft_scale                = 46, 20..=80,            true;
    tm_soft_inc_scale            = 790, 400..=1000,        true;
    tm_soft_fm_scale             = 49, 20..=80,            true;
    tm_hard_scale                = 792, 400..=1000,        true;
    tm_hard_inc_scale            = 732, 400..=1000,        true;
    tm_node_base                 = 1450, 1000..=2000,      true;
    tm_node_scale                = 1350, 800..=2000,       true;
    tm_best_move_base            = 2016, 1000..=2500,      true;
    tm_best_move_scale           = 73, 50..=300,           true;
    tm_best_move_min             = 867, 500..=1000,        true;
    tm_score_base                = 1146, 800..=1800,       true;
    tm_score_scale               = 45, 10..=100,           true;
    tm_score_min                 = 913, 500..=1000,        true;
    thread_weight_score_offset   = 13, 0..=20,             true;
}

#[rustfmt::skip]
tunable_arrays! {
    lmr_factor_1 = [
        -116, -295, -996, 1894, -923, -107, -149,  823, -955,
    ], -2048..=2048;

    lmr_factor_2 = [
        -10, -12, -38,  -1, -22, -20,  -9,  47,  47, -26,  96,  22,
        -20,   3,  50, -34,  13,  10, -23,  -7,   3,  49, -31, -40,
         43, -30, -96,  -6, -39,  45,   4,  52,  11, -30,  70, -19,
    ], -1024..=1024;

    lmr_factor_3 = [
          96,  -13,   33,  -89,  -25,  -43,   27,   50,   -2,    3,  -92,   -2,
         -12,   84,  -65,  -38,  -26,   39,   10,  -74,  -46,   17,  -33,   39,
         -37, -105,  -70,  -50,  -88,   78,   23,  -23,   23,   13,   69,    1,
          59,   67,   49,  -22,   17,  -48,  -34,  -99,  -31, -117,   -7,   38,
         -17,    8,  -49,   63,    0,  -29,   35,  -59,  -10,  -30,  -33,  -95,
         -16,  -51,   74,  -63,  -25,  -47,   33,   69,   68,  -25,  -12,  -25,
          45,   -3,    7,   65,    1,    2,   39,   18,   22,   37,   78,    7,
    ], -1024..=1024;
}

#[inline]
pub fn late_move_threshold(depth: i32, improvement: i32) -> i32 {
    let adjust = improvement.clamp(lmp_improvement_min(), lmp_improvement_max());
    let factor0 = lmp_factor0_base() + lmp_factor0_scale() * adjust / 16;
    let factor1 = lmp_factor1_base() + lmp_factor1_scale() * adjust / 16;

    (factor0 + factor1 * depth * depth) / 1024
}

#[inline]
pub fn se_config(is_quiet: bool) -> (i32, i32, i32) {
    if is_quiet {
        (
            se_beta_quiet_base(),
            se_beta_quiet_scale(),
            se_beta_quiet_div(),
        )
    } else {
        (
            se_beta_noisy_base(),
            se_beta_noisy_scale(),
            se_beta_noisy_div(),
        )
    }
}

#[inline]
pub fn se_dext_margin(is_quiet: bool) -> i32 {
    if is_quiet {
        se_dext_quiet_margin()
    } else {
        se_dext_noisy_margin()
    }
}

#[inline]
pub fn se_text_margin(is_quiet: bool) -> i32 {
    if is_quiet {
        se_text_quiet_margin()
    } else {
        se_text_noisy_margin()
    }
}