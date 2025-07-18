use crate::tunable_params;

#[rustfmt::skip]
tunable_params! {
    asp_delta                   = 24, 4, 36, 4;
    asp_min_depth               = 4, 0, 8, 1;
    asp_alpha_widening_factor   = 200, 50, 400, 50;
    asp_beta_widening_factor    = 200, 50, 400, 50;
    rfp_max_depth               = 8, 6, 12, 1;
    rfp_base                    = 0, -50, 50, 10;
    rfp_scale                   = 80, 40, 100, 10;
    rfp_improving_scale         = 80, 40, 100, 10;
    razor_base                  = 300, 200, 500, 25;
    razor_scale                 = 250, 100, 400, 25;
    nmp_min_depth               = 3, 0, 8, 1;
    nmp_base_reduction          = 3, 2, 5, 1;
    nmp_depth_divisor           = 3, 1, 4, 1;
    nmp_eval_divisor            = 210, 100, 300, 25;
    nmp_eval_max_reduction      = 4, 2, 6, 1;
    iir_min_depth               = 5, 1, 10, 1;
    iir_tt_depth_offset         = 4, 1, 6, 1;
    hindsight_ext_min_depth     = 1, 1, 5, 1;
    hindsight_ext_min_reduction = 3, 1, 5, 1;
    hindsight_ext_eval_diff     = 0, -50, 50, 5;
    hindsight_red_min_depth     = 2, 1, 5, 1;
    hindsight_red_min_reduction = 1, 1, 5, 1;
    hindsight_red_eval_diff     = 80, 0, 120, 20;
    fp_max_depth                = 6, 4, 10, 1;
    fp_base                     = 150, 50, 250, 25;
    fp_scale                    = 100, 50, 200, 10;
    fp_movecount_mult           = 4, 2, 8, 1;
    lmp_max_depth               = 8, 6, 10, 1;
    lmp_base                    = 1, 1, 5, 1;
    lmp_improving_base          = 3, 1, 5, 1;
    lmp_scale                   = 39, 10, 100, 20;
    lmp_improving_scale         = 87, 10, 100, 20;
    hp_max_depth                = 4, 4, 8, 1;
    hp_scale                    = -2048, -3072, -1024, 128;
    bnp_max_depth               = 6, 4, 10, 1;
    bnp_scale                   = 128, 64, 256, 16;
    pvs_see_max_depth           = 8, 6, 10, 1;
    pvs_see_quiet_scale         = -56, -72, -36, 12;
    pvs_see_noisy_scale         = -36, -48, -12, 12;
    se_min_depth                = 8, 6, 10, 1;
    se_tt_depth_offset          = 3, 1, 6, 1;
    se_beta_scale               = 32, 16, 48, 6;
    se_depth_offset             = 1, 0, 3, 1;
    se_depth_divisor            = 2, 1, 4, 1;
    se_double_ext_margin        = 20, 10, 30, 5;
    lmr_min_depth               = 3, 1, 5, 1;
    lmr_min_moves               = 3, 1, 4, 1;
    lmr_pv_node                 = 1024, 0, 2048, 256;
    lmr_cut_node                = 1024, 0, 2048, 256;
    lmr_improving               = 1024, 0, 2048, 256;
    lmr_hist_offset             = 512, -2048, 2048, 256;
    lmr_hist_divisor            = 16384, 8192, 32768, 2048;
    lmr_cont_hist_bonus_scale   = 120, 80, 280, 40;
    lmr_cont_hist_bonus_offset  = 75, 0, 200, 25;
    lmr_cont_hist_bonus_max     = 1200, 1000, 1600, 100;
    lmr_cont_hist_malus_scale   = 120, 80, 280, 40;
    lmr_cont_hist_malus_offset  = 75, 0, 200, 25;
    lmr_cont_hist_malus_max     = 1200, 1000, 1600, 100;
    alpha_raise_min_depth       = 2, 0, 6, 1;
    alpha_raise_max_depth       = 12, 8, 16, 1;
    quiet_hist_bonus_scale      = 120, 80, 280, 40;
    quiet_hist_bonus_offset     = 75, 0, 200, 25;
    quiet_hist_bonus_max        = 1200, 1000, 1600, 100;
    quiet_hist_malus_scale      = 120, 80, 280, 40;
    quiet_hist_malus_offset     = 75, 0, 200, 25;
    quiet_hist_malus_max        = 1200, 1000, 1600, 100;
    capt_hist_bonus_scale       = 120, 80, 280, 40;
    capt_hist_bonus_offset      = 75, 0, 200, 25;
    capt_hist_bonus_max         = 1200, 1000, 1600, 100;
    capt_hist_malus_scale       = 120, 80, 280, 40;
    capt_hist_malus_offset      = 75, 0, 200, 25;
    capt_hist_malus_max         = 1200, 1000, 1600, 100;
    cont_hist_bonus_scale       = 120, 80, 280, 40;
    cont_hist_bonus_offset      = 75, 0, 200, 25;
    cont_hist_bonus_max         = 1200, 1000, 1600, 100;
    cont_hist_malus_scale       = 120, 80, 280, 40;
    cont_hist_malus_offset      = 75, 0, 200, 25;
    cont_hist_malus_max         = 1200, 1000, 1600, 100;
    pcm_bonus_scale             = 120, 80, 280, 40;
    pcm_bonus_offset            = 75, 0, 200, 25;
    pcm_bonus_max               = 1200, 1000, 1600, 100;
    corr_pawn_weight            = 100, 0, 200, 10;
    corr_non_pawn_weight        = 100, 0, 200, 10;
    corr_major_weight           = 100, 0, 200, 10;
    corr_minor_weight           = 100, 0, 200, 10;
    corr_counter_weight         = 100, 0, 200, 10;
    corr_follow_up_weight       = 100, 0, 200, 10;
    see_value_pawn              = 100, 50, 150, 10;
    see_value_knight            = 300, 200, 500, 50;
    see_value_bishop            = 300, 200, 500, 50;
    see_value_rook              = 500, 400, 700, 50;
    see_value_queen             = 900, 800, 1200, 50;
    qs_futility_threshold       = 135, 80, 250, 10;
    qs_see_threshold            = 0, -200, 100, 25;
}