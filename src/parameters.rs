use std::ops::Deref;

pub const ASP_DELTA:                 Tunable = Tunable::new("AspDelta", 24, 4, 36, 4);
pub const ASP_MIN_DEPTH:             Tunable = Tunable::new("AspMinDepth", 4, 0, 8, 1);
pub const ASP_ALPHA_WIDENING_FACTOR: Tunable = Tunable::new("AspAlphaWideningFactor", 200, 50, 400, 50);
pub const ASP_BETA_WIDENING_FACTOR:  Tunable = Tunable::new("AspBetaWideningFactor", 200, 50, 400, 50);
pub const RFP_MAX_DEPTH:             Tunable = Tunable::new("RfpMaxDepth", 8, 6, 12, 1);
pub const RFP_BASE:                  Tunable = Tunable::new("RfpBase", 0, -50, 50, 10);
pub const RFP_SCALE:                 Tunable = Tunable::new("RfpScale", 80, 40, 100, 10);
pub const RFP_IMPROVING_SCALE:       Tunable = Tunable::new("RfpImprovingScale", 80, 40, 100, 10);
pub const RAZOR_BASE:                Tunable = Tunable::new("RazorBase", 300, 200, 500, 25);
pub const RAZOR_SCALE:               Tunable = Tunable::new("RazorScale", 250, 100, 400, 25);
pub const NMP_MIN_DEPTH:             Tunable = Tunable::new("NmpMinDepth", 3, 0, 8, 1);
pub const NMP_BASE_REDUCTION:        Tunable = Tunable::new("NmpBaseReduction", 3, 2, 5, 1);
pub const NMP_DEPTH_DIVISOR:         Tunable = Tunable::new("NmpDepthDivisor", 3, 1, 4, 1);
pub const NMP_EVAL_DIVISOR:          Tunable = Tunable::new("NmpEvalDivisor", 210, 100, 300, 25);
pub const NMP_EVAL_MAX_REDUCTION:    Tunable = Tunable::new("NmpEvalMaxReduction", 4, 2, 6, 1);
pub const IIR_MIN_DEPTH:             Tunable = Tunable::new("IirMinDepth", 5, 1, 10, 1);
pub const IIR_TT_DEPTH_OFFSET:       Tunable = Tunable::new("IirTtDepthOffset", 4, 1, 6, 1);
pub const HINDSIGHT_MIN_REDUCTION:   Tunable = Tunable::new("HindsightMinReduction", 3, 1, 5, 1);
pub const HINDSIGHT_EVAL_DIFF:       Tunable = Tunable::new("HindsightEvalDiff", 0, -50, 50, 5);
pub const FP_MAX_DEPTH:              Tunable = Tunable::new("FpMaxDepth", 6, 4, 10, 1);
pub const FP_BASE:                   Tunable = Tunable::new("FpBase", 150, 50, 250, 25);
pub const FP_SCALE:                  Tunable = Tunable::new("FpScale", 100, 50, 200, 10);
pub const LMP_MAX_DEPTH:             Tunable = Tunable::new("LmpMaxDepth", 8, 6, 10, 1);
pub const LMP_BASE:                  Tunable = Tunable::new("LmpBase", 1, 1, 5, 1);
pub const LMP_IMPROVING_BASE:        Tunable = Tunable::new("LmpImprovingBase", 3, 1, 5, 1);
pub const LMP_SCALE:                 Tunable = Tunable::new("LmpScale", 39, 10, 100, 20);
pub const LMP_IMPROVING_SCALE:       Tunable = Tunable::new("LmpImprovingScale", 87, 10, 100, 20);
pub const HP_MAX_DEPTH:              Tunable = Tunable::new("HpMaxDepth", 4, 4, 8, 1);
pub const HP_SCALE:                  Tunable = Tunable::new("HpScale", -2048, -3072, -1024, 128);
pub const BNP_MAX_DEPTH:             Tunable = Tunable::new("BnpMaxDepth", 6, 4, 10, 1);
pub const BNP_SCALE:                 Tunable = Tunable::new("BnpScale", 128, 64, 256, 16);
pub const PVS_SEE_MAX_DEPTH:         Tunable = Tunable::new("PvsSeeMaxDepth", 8, 6, 10, 1);
pub const PVS_SEE_QUIET_SCALE:       Tunable = Tunable::new("PvsSeeQuietScale", -56, -72, -36, 12);
pub const PVS_SEE_NOISY_SCALE:       Tunable = Tunable::new("PvsSeeNoisyScale", -36, -48, -12, 12);
pub const SE_MIN_DEPTH:              Tunable = Tunable::new("SeMinDepth", 8, 6, 10, 1);
pub const SE_TT_DEPTH_OFFSET:        Tunable = Tunable::new("SeTtDepthOffset", 3, 1, 6, 1);
pub const SE_BETA_SCALE:             Tunable = Tunable::new("SeBetaScale", 32, 16, 48, 6);
pub const SE_DEPTH_OFFSET:           Tunable = Tunable::new("SeDepthOffset", 1, 0, 3, 1);
pub const SE_DEPTH_DIVISOR:          Tunable = Tunable::new("SeDepthDivisor", 2, 1, 4, 1);
pub const SE_DOUBLE_EXT_MARGIN:      Tunable = Tunable::new("SeDoubleExtMargin", 20, 10, 30, 5);
pub const LMR_MIN_DEPTH:             Tunable = Tunable::new("LmrMinDepth", 3, 1, 5, 1);
pub const LMR_MIN_MOVES:             Tunable = Tunable::new("LmrMinMoves", 3, 1, 4, 1);
pub const LMR_PV_NODE:               Tunable = Tunable::new("LmrPvNode", 0, 0, 2048, 256);
pub const LMR_CUT_NODE:              Tunable = Tunable::new("LmrCutNode", 1024, 0, 2048, 256);
pub const LMR_IMPROVING:             Tunable = Tunable::new("LmrImproving", 1024, 0, 2048, 256);
pub const LMR_CONTHIST_BONUS_SCALE:  Tunable = Tunable::new("LmrContHistBonusScale", 120, 80, 280, 40);
pub const LMR_CONTHIST_BONUS_OFFSET: Tunable = Tunable::new("LmrContHistBonusOffset", 75, 0, 200, 25);
pub const LMR_CONTHIST_BONUS_MAX:    Tunable = Tunable::new("LmrContHistBonusMax", 1200, 1600, 1000, 100);
pub const LMR_CONTHIST_MALUS_SCALE:  Tunable = Tunable::new("LmrContHistMalusScale", 120, 80, 280, 40);
pub const LMR_CONTHIST_MALUS_OFFSET: Tunable = Tunable::new("LmrContHistMalusOffset", 75, 0, 200, 25);
pub const LMR_CONTHIST_MALUS_MAX:    Tunable = Tunable::new("LmrContHistMalusMax", 1200, 1600, 1000, 100);
pub const ALPHA_RAISE_MIN_DEPTH:     Tunable = Tunable::new("AlphaRaiseMinDepth", 2, 0, 6, 1);
pub const ALPHA_RAISE_MAX_DEPTH:     Tunable = Tunable::new("AlphaRaiseMaxDepth", 12, 16, 8, 1);
pub const QUIET_HIST_BONUS_SCALE:    Tunable = Tunable::new("QuietHistBonusScale", 120, 80, 280, 40);
pub const QUIET_HIST_BONUS_OFFSET:   Tunable = Tunable::new("QuietHistBonusOffset", 75, 0, 200, 25);
pub const QUIET_HIST_BONUS_MAX:      Tunable = Tunable::new("QuietHistBonusMax", 1200, 1600, 1000, 100);
pub const QUIET_HIST_MALUS_SCALE:    Tunable = Tunable::new("QuietHistMalusScale", 120, 80, 280, 40);
pub const QUIET_HIST_MALUS_OFFSET:   Tunable = Tunable::new("QuietHistMalusOffset", 75, 0, 200, 25);
pub const QUIET_HIST_MALUS_MAX:      Tunable = Tunable::new("QuietHistMalusMax", 1200, 1600, 1000, 100);
pub const CAPT_HIST_BONUS_SCALE:     Tunable = Tunable::new("CaptHistBonusScale", 120, 80, 280, 40);
pub const CAPT_HIST_BONUS_OFFSET:    Tunable = Tunable::new("CaptHistBonusOffset", 75, 0, 200, 25);
pub const CAPT_HIST_BONUS_MAX:       Tunable = Tunable::new("CaptHistBonusMax", 1200, 1600, 1000, 100);
pub const CAPT_HIST_MALUS_SCALE:     Tunable = Tunable::new("CaptHistMalusScale", 120, 80, 280, 40);
pub const CAPT_HIST_MALUS_OFFSET:    Tunable = Tunable::new("CaptHistMalusOffset", 75, 0, 200, 25);
pub const CAPT_HIST_MALUS_MAX:       Tunable = Tunable::new("CaptHistMalusMax", 1200, 1600, 1000, 100);
pub const CONT_HIST_BONUS_SCALE:     Tunable = Tunable::new("ContHistBonusScale", 120, 80, 280, 40);
pub const CONT_HIST_BONUS_OFFSET:    Tunable = Tunable::new("ContHistBonusOffset", 75, 0, 200, 25);
pub const CONT_HIST_BONUS_MAX:       Tunable = Tunable::new("ContHistBonusMax", 1200, 1600, 1000, 100);
pub const CONT_HIST_MALUS_SCALE:     Tunable = Tunable::new("ContHistMalusScale", 120, 80, 280, 40);
pub const CONT_HIST_MALUS_OFFSET:    Tunable = Tunable::new("ContHistMalusOffset", 75, 0, 200, 25);
pub const CONT_HIST_MALUS_MAX:       Tunable = Tunable::new("ContHistMalusMax", 1200, 1600, 1000, 100);
pub const PCM_BONUS_SCALE:           Tunable = Tunable::new("PcmBonusScale", 120, 80, 280, 40);
pub const PCM_BONUS_OFFSET:          Tunable = Tunable::new("PcmBonusOffset", 75, 0, 200, 25);
pub const PCM_BONUS_MAX:             Tunable = Tunable::new("PcmBonusMax", 1200, 1600, 1000, 100);
pub const CORR_PAWN_WEIGHT:          Tunable = Tunable::new("CorrPawnWeight", 100, 0, 200, 10);
pub const CORR_NON_PAWN_WEIGHT:      Tunable = Tunable::new("CorrNonPawnWeight", 100, 0, 200, 10);
pub const CORR_MAJOR_WEIGHT:         Tunable = Tunable::new("CorrMajorWeight", 100, 0, 200, 10);
pub const CORR_MINOR_WEIGHT:         Tunable = Tunable::new("CorrMinorWeight", 100, 0, 200, 10);
pub const CORR_COUNTER_WEIGHT:       Tunable = Tunable::new("CorrCounterWeight", 100, 0, 200, 10);
pub const CORR_FOLLOW_UP_WEIGHT:     Tunable = Tunable::new("CorrFollowUpWeight", 100, 0, 200, 10);
pub const SEE_VALUE_PAWN:            Tunable = Tunable::new("SeeValuePawn", 100, 50, 150, 10);
pub const SEE_VALUE_KNIGHT:          Tunable = Tunable::new("SeeValueKnight", 300, 200, 500, 50);
pub const SEE_VALUE_BISHOP:          Tunable = Tunable::new("SeeValueBishop", 300, 200, 500, 50);
pub const SEE_VALUE_ROOK:            Tunable = Tunable::new("SeeValueRook", 500, 400, 700, 50);
pub const SEE_VALUE_QUEEN:           Tunable = Tunable::new("SeeValueQueen", 900, 800, 1200, 50);

pub fn asp_delta() -> i32 {
    *ASP_DELTA
}

pub fn asp_min_depth() -> i32 {
    *ASP_MIN_DEPTH
}

pub fn asp_alpha_widening_factor() -> i32 {
    *ASP_ALPHA_WIDENING_FACTOR
}

pub fn asp_beta_widening_factor() -> i32 {
    *ASP_BETA_WIDENING_FACTOR
}

pub fn rfp_max_depth() -> i32 {
    *RFP_MAX_DEPTH
}

pub fn rfp_base() -> i32 {
    *RFP_BASE
}

pub fn rfp_scale() -> i32 {
    *RFP_SCALE
}

pub fn rfp_improving_scale() -> i32 {
    *RFP_IMPROVING_SCALE
}

pub fn razor_base() -> i32 {
    *RAZOR_BASE
}

pub fn razor_scale() -> i32 {
    *RAZOR_SCALE
}

pub fn nmp_min_depth() -> i32 {
    *NMP_MIN_DEPTH
}

pub fn nmp_base_reduction() -> i32 {
    *NMP_BASE_REDUCTION
}

pub fn nmp_depth_divisor() -> i32 {
    *NMP_DEPTH_DIVISOR
}

pub fn nmp_eval_divisor() -> i32 {
    *NMP_EVAL_DIVISOR
}

pub fn nmp_eval_max_reduction() -> i32 {
    *NMP_EVAL_MAX_REDUCTION
}

pub fn iir_min_depth() -> i32 {
    *IIR_MIN_DEPTH
}

pub fn iir_tt_depth_offset() -> i32 {
    *IIR_TT_DEPTH_OFFSET
}

pub fn hindsight_min_reduction() -> i32 {
    *HINDSIGHT_MIN_REDUCTION
}

pub fn hindsight_eval_diff() -> i32 {
    *HINDSIGHT_EVAL_DIFF
}

pub fn fp_max_depth() -> i32 {
    *FP_MAX_DEPTH
}

pub fn fp_base() -> i32 {
    *FP_BASE
}

pub fn fp_scale() -> i32 {
    *FP_SCALE
}

pub fn lmp_max_depth() -> i32 {
    *LMP_MAX_DEPTH
}

pub fn lmp_base() -> i32 {
    *LMP_BASE
}

pub fn lmp_improving_base() -> i32 {
    *LMP_IMPROVING_BASE
}

pub fn lmp_scale() -> i32 {
    *LMP_SCALE
}

pub fn lmp_improving_scale() -> i32 {
    *LMP_IMPROVING_SCALE
}

pub fn hp_max_depth() -> i32 {
    *HP_MAX_DEPTH
}

pub fn hp_scale() -> i32 {
    *HP_SCALE
}

pub fn bnp_max_depth() -> i32 {
    *BNP_MAX_DEPTH
}

pub fn bnp_scale() -> i32 {
    *BNP_SCALE
}

pub fn pvs_see_max_depth() -> i32 {
    *PVS_SEE_MAX_DEPTH
}

pub fn pvs_see_quiet_scale() -> i32 {
    *PVS_SEE_QUIET_SCALE
}

pub fn pvs_see_noisy_scale() -> i32 {
    *PVS_SEE_NOISY_SCALE
}

pub fn se_min_depth() -> i32 {
    *SE_MIN_DEPTH
}

pub fn se_tt_depth_offset() -> i32 {
    *SE_TT_DEPTH_OFFSET
}

pub fn se_beta_scale() -> i32 {
    *SE_BETA_SCALE
}

pub fn se_depth_offset() -> i32 {
    *SE_DEPTH_OFFSET
}

pub fn se_depth_divisor() -> i32 {
    *SE_DEPTH_DIVISOR
}

pub fn se_double_ext_margin() -> i32 {
    *SE_DOUBLE_EXT_MARGIN
}

pub fn lmr_min_depth() -> i32 {
    *LMR_MIN_DEPTH
}

pub fn lmr_min_moves() -> i32 {
    *LMR_MIN_MOVES
}

pub fn lmr_pv_node() -> i32 {
    *LMR_PV_NODE
}

pub fn lmr_cut_node() -> i32 {
    *LMR_CUT_NODE
}

pub fn lmr_improving() -> i32 {
    *LMR_IMPROVING
}

pub fn lmr_conthist_bonus_scale() -> i16 {
    *LMR_CONTHIST_BONUS_SCALE as i16
}

pub fn lmr_conthist_bonus_offset() -> i16 {
    *LMR_CONTHIST_BONUS_OFFSET as i16
}

pub fn lmr_conthist_bonus_max() -> i16 {
    *LMR_CONTHIST_BONUS_MAX as i16
}

pub fn lmr_conthist_malus_scale() -> i16 {
    *LMR_CONTHIST_MALUS_SCALE as i16
}

pub fn lmr_conthist_malus_offset() -> i16 {
    *LMR_CONTHIST_MALUS_OFFSET as i16
}

pub fn lmr_conthist_malus_max() -> i16 {
    *LMR_CONTHIST_MALUS_MAX as i16
}

pub fn alpha_raise_min_depth() -> i32 {
    *ALPHA_RAISE_MIN_DEPTH
}

pub fn alpha_raise_max_depth() -> i32 {
    *ALPHA_RAISE_MAX_DEPTH
}

pub fn quiet_hist_bonus_scale() -> i16 {
    *QUIET_HIST_BONUS_SCALE as i16
}

pub fn quiet_hist_bonus_offset() -> i16 {
    *QUIET_HIST_BONUS_OFFSET as i16
}

pub fn quiet_hist_bonus_max() -> i16 {
    *QUIET_HIST_BONUS_MAX as i16
}

pub fn quiet_hist_malus_scale() -> i16 {
    *QUIET_HIST_MALUS_SCALE as i16
}

pub fn quiet_hist_malus_offset() -> i16 {
    *QUIET_HIST_MALUS_OFFSET as i16
}

pub fn quiet_hist_malus_max() -> i16 {
    *QUIET_HIST_MALUS_MAX as i16
}

pub fn capt_hist_bonus_scale() -> i16 {
    *CAPT_HIST_BONUS_SCALE as i16
}

pub fn capt_hist_bonus_offset() -> i16 {
    *CAPT_HIST_BONUS_OFFSET as i16
}

pub fn capt_hist_bonus_max() -> i16 {
    *CAPT_HIST_BONUS_MAX as i16
}

pub fn capt_hist_malus_scale() -> i16 {
    *CAPT_HIST_MALUS_SCALE as i16
}

pub fn capt_hist_malus_offset() -> i16 {
    *CAPT_HIST_MALUS_OFFSET as i16
}

pub fn capt_hist_malus_max() -> i16 {
    *CAPT_HIST_MALUS_MAX as i16
}

pub fn cont_hist_bonus_scale() -> i16 {
    *CONT_HIST_BONUS_SCALE as i16
}

pub fn cont_hist_bonus_offset() -> i16 {
    *CONT_HIST_BONUS_OFFSET as i16
}

pub fn cont_hist_bonus_max() -> i16 {
    *CONT_HIST_BONUS_MAX as i16
}

pub fn cont_hist_malus_scale() -> i16 {
    *CONT_HIST_MALUS_SCALE as i16
}

pub fn cont_hist_malus_offset() -> i16 {
    *CONT_HIST_MALUS_OFFSET as i16
}

pub fn cont_hist_malus_max() -> i16 {
    *CONT_HIST_MALUS_MAX as i16
}

pub fn pcm_bonus_scale() -> i16 {
    *PCM_BONUS_SCALE as i16
}

pub fn pcm_bonus_offset() -> i16 {
    *PCM_BONUS_OFFSET as i16
}

pub fn pcm_bonus_max() -> i16 {
    *PCM_BONUS_MAX as i16
}

pub fn corr_pawn_weight() -> i32 {
    *CORR_PAWN_WEIGHT
}

pub fn corr_non_pawn_weight() -> i32 {
    *CORR_NON_PAWN_WEIGHT
}

pub fn corr_major_weight() -> i32 {
    *CORR_MAJOR_WEIGHT
}

pub fn corr_minor_weight() -> i32 {
    *CORR_MINOR_WEIGHT
}

pub fn corr_counter_weight() -> i32 {
    *CORR_COUNTER_WEIGHT
}

pub fn corr_follow_up_weight() -> i32 {
    *CORR_FOLLOW_UP_WEIGHT
}

pub fn see_value_pawn() -> i32 {
    *SEE_VALUE_PAWN
}

pub fn see_value_knight() -> i32 {
    *SEE_VALUE_KNIGHT
}

pub fn see_value_bishop() -> i32 {
    *SEE_VALUE_BISHOP
}

pub fn see_value_rook() -> i32 {
    *SEE_VALUE_ROOK
}

pub fn see_value_queen() -> i32 {
    *SEE_VALUE_QUEEN
}

pub struct Tunable {
    name: &'static str,
    value: i32,
    min: i32,
    max: i32,
    step: i32
}

impl Tunable {

    pub const fn new(name: &'static str, value: i32, min: i32, max: i32, step: i32) -> Self {
        Self { name, value, min, max, step, }
    }

    pub fn print(&self) {
        println!("option name {} type spin default {} min {} max {}",
            self.name, self.value, self.min, self.max);
    }

    pub fn set_value(&mut self, value: i32) {
        if value < self.min || value > self.max {
            println!("info error {} out of bounds for tunable {} (min: {}, max: {})",
                     value, self.name, self.min, self.max);
        } else {
            self.value = value;
        }
    }

}

impl Deref for Tunable {
    type Target = i32;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

#[cfg(feature = "spsa")]
pub fn print_tunables() {
    let tunables = [
        &ASP_DELTA,
        &ASP_MIN_DEPTH,
        &ASP_ALPHA_WIDENING_FACTOR,
        &ASP_BETA_WIDENING_FACTOR,
        &RFP_MAX_DEPTH,
        &RFP_BASE,
        &RFP_SCALE,
        &RFP_IMPROVING_SCALE,
        &RAZOR_BASE,
        &RAZOR_SCALE,
        &NMP_MIN_DEPTH,
        &NMP_BASE_REDUCTION,
        &NMP_DEPTH_DIVISOR,
        &NMP_EVAL_DIVISOR,
        &NMP_EVAL_MAX_REDUCTION,
        &HINDSIGHT_MIN_REDUCTION,
        &HINDSIGHT_EVAL_DIFF,
        &IIR_MIN_DEPTH,
        &IIR_TT_DEPTH_OFFSET,
        &FP_MAX_DEPTH,
        &FP_BASE,
        &FP_SCALE,
        &LMP_MAX_DEPTH,
        &LMP_BASE,
        &LMP_IMPROVING_BASE,
        &LMP_SCALE,
        &LMP_IMPROVING_SCALE,
        &HP_MAX_DEPTH,
        &HP_SCALE,
        &BNP_MAX_DEPTH,
        &BNP_SCALE,
        &PVS_SEE_MAX_DEPTH,
        &PVS_SEE_QUIET_SCALE,
        &PVS_SEE_NOISY_SCALE,
        &SE_MIN_DEPTH,
        &SE_TT_DEPTH_OFFSET,
        &SE_BETA_SCALE,
        &SE_DEPTH_OFFSET,
        &SE_DEPTH_DIVISOR,
        &SE_DOUBLE_EXT_MARGIN,
        &LMR_MIN_DEPTH,
        &LMR_MIN_MOVES,
        &LMR_PV_NODE,
        &LMR_CUT_NODE,
        &LMR_IMPROVING,
        &LMR_CONTHIST_BONUS_SCALE,
        &LMR_CONTHIST_BONUS_OFFSET,
        &LMR_CONTHIST_BONUS_MAX,
        &LMR_CONTHIST_MALUS_SCALE,
        &LMR_CONTHIST_MALUS_OFFSET,
        &LMR_CONTHIST_MALUS_MAX,
        &ALPHA_RAISE_MIN_DEPTH,
        &ALPHA_RAISE_MAX_DEPTH,
        &QUIET_HIST_BONUS_SCALE,
        &QUIET_HIST_BONUS_OFFSET,
        &QUIET_HIST_BONUS_MAX,
        &QUIET_HIST_MALUS_SCALE,
        &QUIET_HIST_MALUS_OFFSET,
        &QUIET_HIST_MALUS_MAX,
        &CAPT_HIST_BONUS_SCALE,
        &CAPT_HIST_BONUS_OFFSET,
        &CAPT_HIST_BONUS_MAX,
        &CAPT_HIST_MALUS_SCALE,
        &CAPT_HIST_MALUS_OFFSET,
        &CAPT_HIST_MALUS_MAX,
        &CONT_HIST_BONUS_SCALE,
        &CONT_HIST_BONUS_OFFSET,
        &CONT_HIST_BONUS_MAX,
        &CONT_HIST_MALUS_SCALE,
        &CONT_HIST_MALUS_OFFSET,
        &CONT_HIST_MALUS_MAX,
        &PCM_BONUS_SCALE,
        &PCM_BONUS_OFFSET,
        &PCM_BONUS_MAX,
        &CORR_PAWN_WEIGHT,
        &CORR_NON_PAWN_WEIGHT,
        &CORR_MAJOR_WEIGHT,
        &CORR_MINOR_WEIGHT,
        &CORR_COUNTER_WEIGHT,
        &CORR_FOLLOW_UP_WEIGHT,
    ];
    for tunable in tunables {
        tunable.print();
    }
}

#[cfg(not(feature = "spsa"))]
pub fn print_tunables() {

}