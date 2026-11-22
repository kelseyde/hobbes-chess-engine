use crate::search::history::{lmr_conthist_1_bonus, lmr_conthist_1_malus, lmr_conthist_2_bonus, lmr_conthist_2_malus, lmr_conthist_4_bonus, lmr_conthist_4_malus, se_conthist_1_bonus, se_conthist_2_bonus, se_conthist_4_bonus, ContinuationHistory};
use crate::tunable_arrays;
use crate::tunable_params;

#[rustfmt::skip]
tunable_params! {
    asp_delta                    = 10, 4..=36,             true;
    asp_min_depth                = 4, 0..=8,               false;
    asp_alpha_widening_factor    = 226, 50..=400,          true;
    asp_beta_widening_factor     = 261, 50..=400,          true;
    asp_prev_score_div           = 10271, 8000..=12000,    true;
    rfp_max_depth                = 8, 6..=12,              false;
    rfp_base                     = 12, -50..=50,           true;
    rfp_scale                    = 63, 40..=100,           true;
    rfp_improving_scale          = 56, 40..=100,           true;
    rfp_opp_worsening_scale      = 12, 0..=50,             true;
    rfp_tt_move_noisy_scale      = 5, 0..=70,              true;
    rfp_lerp_factor              = 37, 0..=100,            true;
    razor_base                   = 278, 200..=500,         true;
    razor_scale                  = 252, 100..=400,         true;
    nmp_min_depth                = 3, 0..=8,               false;
    nmp_margin_base              = 39, -128..=128,         true;
    nmp_margin_scale             = 11, 0..=1536,           true;
    nmp_margin_improving         = -1, -60..=60,           true;
    nmp_red_base                 = 5114, 3072..=8192,      true;
    nmp_red_depth_mult           = 333, 128..=512,         true;
    nmp_red_eval_mult            = 597, 256..=1024,        true;
    nmp_red_eval_max             = 1116, 512..=2048,       true;
    nmp_red_div                  = 123, 64..=256,          true;
    iir_min_depth                = 4, 1..=10,              false;
    iir_tt_depth_offset          = 4, 1..=6,               false;
    cutnode_red_min_depth        = 8, 4..=12,              false;
    cutnode_red_tt_offset        = 4, 1..=6,               false;
    pc_base                      = 136, 100..=400,         true;
    pc_scale                     = 21, 0..=40,             true;
    pc_ttpv_margin               = 61, 0..=100,            true;
    pc_cutnode_margin            = 39, 0..=100,            true;
    pc_improving_margin          = 65, 0..=100,            true;
    pc_max                       = 509, 300..=700,         true;
    pc_tt_depth_offset           = 2, 0..=4,               false;
    hindsight_ext_min_depth      = 1, 1..=5,               false;
    hindsight_ext_min_reduction  = 1014, 0..=5120,         true;
    hindsight_ext_eval_diff      = -17, -50..=50,          true;
    hindsight_red_min_depth      = 2, 1..=5,               false;
    hindsight_red_min_reduction  = 1039, 0..=5120,         true;
    hindsight_red_eval_diff      = 51, 0..=120,            true;
    hindsight_hist_mult          = 9, 0..=20,              true;
    hindsight_hist_min           = -63, -100..=0,          true;
    hindsight_hist_max           = 218, 75..=300,          true;
    fp_max_depth                 = 8, 4..=10,              false;
    fp_base                      = 161, 50..=250,          true;
    fp_scale                     = 93, 50..=200,           true;
    fp_pv_node                   = 3, -50..=100,           true;
    fp_movecount_mult            = 4, 2..=8,               false;
    fp_history_divisor           = 65, 64..=256,           true;
    fp_killer                    = 19, 0..=100,            true;
    fp_tt_upper                  = 23, 0..=100,            true;
    fp_eval_beta                 = 77, 0..=150,            true;
    lmp_max_depth                = 8, 6..=10,              false;
    lmp_factor0_base             = 2495, 2000..=3000,      true;
    lmp_factor0_scale            = 128, 50..=200,          true;
    lmp_factor1_base             = 885, 500..=1500,        true;
    lmp_factor1_scale            = 78, 50..=200,           true;
    lmp_improvement_min          = -128, -300..=0,         true;
    lmp_improvement_max          = 197, 0..=500,           true;
    hp_max_depth                 = 4, 4..=8,               false;
    hp_base                      = 87, -2000..=2000,       true;
    hp_scale                     = -2370, -3072..=-1024,   true;
    bnp_max_depth                = 6, 4..=10,              false;
    bnp_base                     = -1, -50..=50,           true;
    bnp_scale                    = 143, 64..=256,          true;
    see_max_depth                = 8, 6..=10,              false;
    see_quiet_mult1              = -2074, -3072..=-1024,   true;
    see_quiet_mult2              = 53, 32..=80,            true;
    see_quiet_offset             = 28, 0..=40,             true;
    see_noisy_mult1              = -957, -2048..=0,        true;
    see_noisy_mult2              = 37, 16..=48,            true;
    see_noisy_offset             = 10, 0..=30,             true;
    see_quiet_history_div        = 181, 164..=388,         true;
    see_noisy_history_div        = 179, 164..=388,         true;
    see_quiet_ttpv_scale         = 26, 0..=36,             true;
    see_noisy_ttpv_scale         = 22, 0..=36,             true;
    se_min_depth                 = 6, 6..=10,              false;
    se_tt_depth_offset           = 3, 1..=6,               false;
    se_depth_offset              = 1, 0..=3,               false;
    se_depth_divisor             = 2, 1..=4,               false;
    se_beta_quiet_base           = 59, 0..=100,            true;
    se_beta_quiet_scale          = 19, 16..=48,            true;
    se_beta_quiet_div            = 52, 40..=100,           true;
    se_beta_history_div          = 1019, 512..=2048,       true;
    se_dext_quiet_margin         = 10, 0..=30,             true;
    se_text_quiet_margin         = 48, 20..=120,           true;
    se_beta_noisy_base           = 59, 0..=100,            true;
    se_beta_noisy_scale          = 17, 16..=48,            true;
    se_beta_noisy_div            = 55, 40..=100,           true;
    se_dext_noisy_margin         = 7, 0..=30,              true;
    se_text_noisy_margin         = 73, 20..=120,           true;
    se_dext_pv_margin            = 96, 0..=200,            true;
    se_text_pv_margin            = 98, 0..=200,            true;
    ldse_max_depth               = 7, 1..=12,              false;
    ldse_margin                  = 26, 0..=80,             true;
    ldse_dext_tt_depth_offset    = 3, 1..=5,               false;
    ldse_dext_margin             = 37, 20..=80,            true;
    lmr_min_depth                = 2, 1..=5,               false;
    lmr_min_moves                = 2, 1..=4,               false;
    lmr_quiet_base               = 88, 50..=150,           true;
    lmr_quiet_div                = 300, 250..=400,         true;
    lmr_noisy_base               = 97, 50..=150,           true;
    lmr_noisy_div                = 298, 250..=400,         true;
    lmr_complexity_margin        = 93, 0..=300,            true;
    lmr_complex                  = 271, 0..=2048,          true;
    lmr_good_noisy               = 908, 0..=2048,          true;
    lmr_bad_noisy                = 720, 0..=2048,          true;
    lmr_fail_highs               = 4, 0..=2048,            true;
    lmr_quiet_see                = 1302, 0..=2048,         true;
    lmr_se_mult                  = 571, 256..=1024,        true;
    lmr_se_offset                = 232, 0..=1024,          true;
    lmr_se_div                   = 96, 64..=256,           true;
    lmr_se_max                   = 2516, 0..=3072,         true;
    lmr_hist_offset              = 1249, -2048..=2048,     true;
    lmr_hist_divisor             = 15628, 8192..=32768,    true;
    lmr_hist_min                 = -3077, -4096..=0,       true;
    lmr_hist_max                 = 3000, 0..=4096,         true;
    lmr_mvv_divisor              = 378, 128..=640,         true;
    lmr_deeper_base              = 19, 0..=100,            true;
    lmr_deeper_scale             = 534, 350..=768,         true;
    lmr_deeper_div               = 166, 64..=256,          true;
    lmr_shallower_base           = 1063, 512..=2048,       true;
    lmr_shallower_scale          = 1032, 512..=2048,       true;
    lmr_shallower_div            = 1098, 512..=2048,       true;
    lmr_even_deeper_margin       = 533, 0..=1000,          true;
    lmr_cont_1_bonus_scale       = 229, 80..=280,          true;
    lmr_cont_1_bonus_offset      = 152, 0..=200,           true;
    lmr_cont_1_bonus_max         = 1187, 1000..=1600,      true;
    lmr_cont_1_malus_scale       = 200, 80..=280,          true;
    lmr_cont_1_malus_offset      = 91, 0..=200,            true;
    lmr_cont_1_malus_max         = 1312, 1000..=1600,      true;
    lmr_cont_2_bonus_scale       = 172, 80..=280,          true;
    lmr_cont_2_bonus_offset      = 119, 0..=200,           true;
    lmr_cont_2_bonus_max         = 1061, 1000..=1600,      true;
    lmr_cont_2_malus_scale       = 199, 80..=280,          true;
    lmr_cont_2_malus_offset      = 62, 0..=200,            true;
    lmr_cont_2_malus_max         = 1214, 1000..=1600,      true;
    lmr_cont_4_bonus_scale       = 168, 80..=280,          true;
    lmr_cont_4_bonus_offset      = 119, 0..=200,           true;
    lmr_cont_4_bonus_max         = 1080, 1000..=1600,      true;
    lmr_cont_4_malus_scale       = 203, 80..=280,          true;
    lmr_cont_4_malus_offset      = 66, 0..=200,            true;
    lmr_cont_4_malus_max         = 1213, 1000..=1600,      true;
    se_cont_1_bonus_scale       = 229, 80..=280,          true;
    se_cont_1_bonus_offset      = 152, 0..=200,           true;
    se_cont_1_bonus_max         = 1187, 1000..=1600,      true;
    se_cont_2_bonus_scale       = 172, 80..=280,          true;
    se_cont_2_bonus_offset      = 119, 0..=200,           true;
    se_cont_2_bonus_max         = 1061, 1000..=1600,      true;
    se_cont_4_bonus_scale       = 168, 80..=280,          true;
    se_cont_4_bonus_offset      = 119, 0..=200,           true;
    se_cont_4_bonus_max         = 1080, 1000..=1600,      true;
    alpha_raise_min_depth        = 2, 0..=6,               false;
    alpha_raise_max_depth        = 12, 8..=16,             false;
    quiet_hist_bonus_scale       = 218, 80..=280,          true;
    quiet_hist_bonus_offset      = 80, 0..=200,            true;
    quiet_hist_cutnode_offset    = 80, 0..=200,            true;
    quiet_hist_ttmove_bonus      = 75, 0..=200,            true;
    quiet_hist_capture_mult      = 7, 0..=50,              true;
    quiet_hist_bonus_max         = 1114, 800..=1600,       true;
    quiet_hist_malus_scale       = 79, 12..=280,           true;
    quiet_hist_malus_offset      = 15, 0..=200,            true;
    quiet_hist_ttmove_malus      = 73, 0..=200,            true;
    quiet_hist_malus_max         = 1410, 1000..=1600,      true;
    quiet_fact_bonus_scale       = 238, 80..=320,          true;
    quiet_fact_bonus_offset      = 30, 0..=200,            true;
    quiet_fact_cutnode_offset    = 73, 0..=200,            true;
    quiet_fact_ttmove_bonus      = 62, 0..=200,            true;
    quiet_fact_capture_mult      = 9, 0..=50,              true;
    quiet_fact_bonus_max         = 1057, 800..=1600,       true;
    quiet_fact_malus_scale       = 88, 12..=280,           true;
    quiet_fact_malus_offset      = 10, 0..=200,            true;
    quiet_fact_ttmove_malus      = 86, 0..=200,            true;
    quiet_fact_malus_max         = 1348, 1000..=1600,      true;
    quiet_hist_lerp_factor       = 41, 0..=100,            true;
    capt_hist_lerp_factor        = 27, 0..=100,            true;
    capt_hist_bonus_scale        = 232, 80..=360,          true;
    capt_hist_bonus_offset       = 48, 0..=200,            true;
    capt_hist_ttmove_bonus       = 66, 0..=200,            true;
    capt_hist_bonus_max          = 1509, 1000..=1800,      true;
    capt_hist_malus_scale        = 166, 80..=280,          true;
    capt_hist_malus_offset       = 74, 0..=200,            true;
    capt_hist_ttmove_malus       = 51, 0..=200,            true;
    capt_hist_malus_max          = 1328, 1000..=1600,      true;
    cont_hist_1_bonus_scale      = 135, 80..=280,          true;
    cont_hist_1_bonus_offset     = 193, 0..=300,           true;
    cont_hist_1_cutnode_offset   = 55, 0..=200,            true;
    cont_hist_1_ttmove_bonus     = 61, 0..=200,            true;
    cont_hist_1_capture_mult     = 22, 0..=100,            true;
    cont_hist_1_bonus_max        = 935, 800..=1600,        true;
    cont_hist_1_malus_scale      = 118, 80..=280,          true;
    cont_hist_1_malus_offset     = 100, 0..=200,           true;
    cont_hist_1_ttmove_malus     = 85, 0..=200,            true;
    cont_hist_1_malus_max        = 1063, 800..=1600,       true;
    cont_hist_2_bonus_scale      = 110, 80..=280,          true;
    cont_hist_2_bonus_offset     = 199, 0..=300,           true;
    cont_hist_2_cutnode_offset   = 58, 0..=200,            true;
    cont_hist_2_ttmove_bonus     = 53, 0..=200,            true;
    cont_hist_2_capture_mult     = 16, 0..=100,            true;
    cont_hist_2_bonus_max        = 989, 800..=1600,        true;
    cont_hist_2_malus_scale      = 88, 64..=280,           true;
    cont_hist_2_malus_offset     = 85, 0..=200,            true;
    cont_hist_2_ttmove_malus     = 91, 0..=200,            true;
    cont_hist_2_malus_max        = 1019, 800..=1600,       true;
    cont_hist_4_bonus_scale      = 110, 80..=280,          true;
    cont_hist_4_bonus_offset     = 196, 0..=300,           true;
    cont_hist_4_bonus_max        = 987, 800..=1600,        true;
    cont_hist_4_malus_scale      = 95, 64..=280,           true;
    cont_hist_4_malus_offset     = 82, 0..=200,            true;
    cont_hist_4_malus_max        = 1028, 800..=1600,       true;
    from_hist_bonus_scale        = 194, 80..=280,          true;
    from_hist_bonus_offset       = 95, 0..=200,            true;
    from_hist_bonus_max          = 937, 800..=1600,        true;
    from_hist_malus_scale        = 222, 80..=280,          true;
    from_hist_malus_offset       = 83, 0..=200,            true;
    from_hist_malus_max          = 1098, 800..=1600,       true;
    to_hist_bonus_scale          = 196, 80..=300,          true;
    to_hist_bonus_offset         = 89, 0..=200,            true;
    to_hist_bonus_max            = 944, 800..=1600,        true;
    to_hist_malus_scale          = 190, 80..=280,          true;
    to_hist_malus_offset         = 65, 0..=200,            true;
    to_hist_malus_max            = 1011, 800..=1600,       true;
    pcm_bonus_scale              = 220, 80..=300,          true;
    pcm_bonus_offset             = 74, 0..=200,            true;
    pcm_bonus_max                = 1388, 1000..=1600,      true;
    corr_pawn_bonus_mult         = 219, 100..=300,         true;
    corr_pawn_bonus_div          = 84, 64..=256,           true;
    corr_pawn_bonus_min          = -4074, -6000..=-2000,   true;
    corr_pawn_bonus_max          = 2842, 2000..=6000,      true;
    corr_nonpawn_bonus_mult      = 217, 100..=300,         true;
    corr_nonpawn_bonus_div       = 79, 64..=256,           true;
    corr_nonpawn_bonus_min       = -4194, -6000..=-2000,   true;
    corr_nonpawn_bonus_max       = 3011, 2000..=6000,      true;
    corr_major_bonus_mult        = 228, 100..=360,         true;
    corr_major_bonus_div         = 99, 64..=256,           true;
    corr_major_bonus_min         = -3879, -6000..=-2000,   true;
    corr_major_bonus_max         = 3441, 2000..=6000,      true;
    corr_minor_bonus_mult        = 197, 100..=300,         true;
    corr_minor_bonus_div         = 99, 64..=256,           true;
    corr_minor_bonus_min         = -4514, -6000..=-2000,   true;
    corr_minor_bonus_max         = 3552, 2000..=6000,      true;
    corr_cont1_bonus_mult        = 202, 100..=300,         true;
    corr_cont1_bonus_div         = 95, 64..=256,           true;
    corr_cont1_bonus_min         = -4329, -6000..=-2000,   true;
    corr_cont1_bonus_max         = 3189, 2000..=6000,      true;
    corr_cont2_bonus_mult        = 207, 100..=300,         true;
    corr_cont2_bonus_div         = 117, 64..=256,          true;
    corr_cont2_bonus_min         = -4248, -6000..=-2000,   true;
    corr_cont2_bonus_max         = 3099, 2000..=6000,      true;
    corr_pawn_weight             = 70, 1..=200,            true;
    corr_non_pawn_weight         = 72, 1..=200,            true;
    corr_major_weight            = 105, 1..=200,           true;
    corr_minor_weight            = 63, 1..=200,            true;
    corr_cont1_weight            = 97, 1..=200,            true;
    corr_cont2_weight            = 132, 1..=200,           true;
    see_value_pawn_pruning       = 100, 50..=150,          true;
    see_value_knight_pruning     = 258, 200..=500,         true;
    see_value_bishop_pruning     = 309, 200..=500,         true;
    see_value_rook_pruning       = 542, 400..=700,         true;
    see_value_queen_pruning      = 968, 800..=1200,        true;
    see_value_pawn_ordering      = 103, 50..=150,          true;
    see_value_knight_ordering    = 307, 200..=500,         true;
    see_value_bishop_ordering    = 368, 200..=500,         true;
    see_value_rook_ordering      = 545, 400..=700,         true;
    see_value_queen_ordering     = 1015, 800..=1200,       true;
    scale_value_pawn             = 17, 0..=250,            true;
    scale_value_knight           = 480, 300..=500,         true;
    scale_value_bishop           = 430, 300..=500,         true;
    scale_value_rook             = 612, 500..=700,         true;
    scale_value_queen            = 1266, 1000..=1400,      true;
    material_scaling_base        = 12935, 10000..=40000,   true;
    qs_futility_threshold        = 182, 80..=250,          true;
    qs_see_threshold             = -81, -200..=100,        true;
    qs_stand_pat_lerp_factor     = 52, 0..=100,            true;
    qs_fail_high_lerp_factor     = 49, 0..=100,            true;
    movepick_see_divisor         = 45, 30..=60,            true;
    movepick_see_offset          = 119, 80..=200,          true;
    movepick_mvv_scale           = 2028, 1024..=4096,      true;
    score_stability_threshold    = 16, 4..=24,             true;
    tm_soft_base                 = 24, 10..=60,            true;
    tm_soft_scale                = 46, 20..=80,            true;
    tm_soft_inc_scale            = 788, 400..=1000,        true;
    tm_soft_fm_scale             = 50, 20..=80,            true;
    tm_hard_scale                = 798, 400..=1000,        true;
    tm_hard_inc_scale            = 739, 400..=1000,        true;
    tm_node_base                 = 1461, 1000..=2000,      true;
    tm_node_scale                = 1356, 800..=2000,       true;
    tm_best_move_base            = 2007, 1000..=2500,      true;
    tm_best_move_scale           = 70, 50..=300,           true;
    tm_best_move_min             = 862, 500..=1000,        true;
    tm_score_base                = 1131, 800..=1800,       true;
    tm_score_scale               = 46, 10..=100,           true;
    tm_score_min                 = 926, 500..=1000,        true;
    thread_weight_score_offset   = 14, 0..=20,             true;
}

#[rustfmt::skip]
tunable_arrays! {
    lmr_factor_1 = [
         -207,  -243, -1106,  1983,  -821,   -57,  -227,   741,  -964,
    ], -2048..=2048;

    lmr_factor_2 = [
         33,  16,   0, -18, -12, -53,  12,  19,  55,  40,  85,  50,
        -30,  23,  54, -43, -28,  24, -51, -15,  -1,  76, -41, -59,
         42, -13, -95, -32, -43,  21,  12,  75,  -6, -71,  85, -39,
    ], -1024..=1024;

    lmr_factor_3 = [
          61,  -73,   16,  -93,  -14,  -74,   74,   67,   35,   -5,  -72,    7,
           2,   98,  -36,  -36,  -53,   54,   -7,  -50,  -49,   43,   -7,   26,
         -55,  -51,  -59,  -44,  -86,   49,   -7,  -13,   30,   -4,   43,   41,
          73,   71,   90,  -57,   16,  -26,  -55, -109,  -43,  -98,  -16,   57,
         -28,   25,  -53,   62,   21,   31,   47,  -32,  -14,  -54,  -20,  -95,
          18,  -36,   84,  -68,   44,  -46,   30,   47,   53,   -8,  -15,  -42,
           2,   13,   16,   67,   16,   -9,   21,   46,  -31,   34,   99,   -2,
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

#[inline]
pub fn lmr_cont_bonuses(depth: i32, good: bool) -> [i16; ContinuationHistory::PLY_COUNT] {
    let bonus_1 = if good {
        lmr_conthist_1_bonus(depth)
    } else {
        lmr_conthist_1_malus(depth)
    };
    let bonus_2 = if good {
        lmr_conthist_2_bonus(depth)
    } else {
        lmr_conthist_2_malus(depth)
    };
    let bonus_4 = if good {
        lmr_conthist_4_bonus(depth)
    } else {
        lmr_conthist_4_malus(depth)
    };
    [bonus_1, bonus_2, bonus_4]
}

#[inline]
pub fn se_cont_bonuses(depth: i32) -> [i16; ContinuationHistory::PLY_COUNT] {
    [se_conthist_1_bonus(depth), se_conthist_2_bonus(depth), se_conthist_4_bonus(depth)]
}
