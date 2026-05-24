use std::time::Duration;

use crate::search::parameters::*;

pub const UCI_OVERHEAD_MS: u64 = 50;

#[rustfmt::skip]
pub struct TimeParams {
    pub soft_tm_base:        f64,
    pub soft_tm_scale:       f64,
    pub soft_tm_inc_scale:   f64,
    pub soft_tm_fm_scale:    f64,
    pub hard_tm_scale:       f64,
    pub hard_tm_inc_scale:   f64,
    pub node_tm_base:        f32,
    pub node_tm_scale:       f32,
    pub best_move_tm_base:   f32,
    pub best_move_tm_scale:  f32,
    pub best_move_tm_min:    f32,
    pub score_tm_base:       f32,
    pub score_tm_scale:      f32,
    pub score_tm_min:        f32,
}

impl TimeParams {
    #[rustfmt::skip]
    pub fn init() -> Self {
        Self {
            soft_tm_base:       tm_soft_base() as f64 / 1000.0,
            soft_tm_scale:      tm_soft_scale() as f64 / 1000.0,
            soft_tm_inc_scale:  tm_soft_inc_scale() as f64 / 1000.0,
            soft_tm_fm_scale:   tm_soft_fm_scale() as f64 / 1000.0,
            hard_tm_scale:      tm_hard_scale() as f64 / 1000.0,
            hard_tm_inc_scale:  tm_hard_inc_scale() as f64 / 1000.0,
            node_tm_base:       tm_node_base() as f32 / 1000.0,
            node_tm_scale:      tm_node_scale() as f32 / 1000.0,
            best_move_tm_base:  tm_best_move_base() as f32 / 1000.0,
            best_move_tm_scale: tm_best_move_scale() as f32 / 1000.0,
            best_move_tm_min:   tm_best_move_min() as f32 / 1000.0,
            score_tm_base:      tm_score_base() as f32 / 1000.0,
            score_tm_scale:     tm_score_scale() as f32 / 1000.0,
            score_tm_min:       tm_score_min() as f32 / 1000.0,
        }
    }
}

pub enum LimitType {
    Soft,
    Hard,
}

pub type FischerTime = (u64, u64);

/// The amount of time the engine chooses to search is split into to two limits: hard and soft. The
/// hard limit is checked regularly during search, and the search is aborted as soon as it is reached.
/// The soft limit is checked at the start of each iterative deepening loop, and the engine does not
/// bother starting a new search if it is reached.
///
/// We can then scale the soft limit up or down during search, based on how stable the search is.
/// 'How stable the search is' can be captured a few ways: how many iterations the best move has
/// remained the same, how stable the search score has been across iterations, or what portion of
/// nodes have been spent searching the current best move.
pub struct SearchLimits {
    pub hard_time: Option<Duration>,
    pub soft_time: Option<Duration>,
    pub soft_nodes: Option<u64>,
    pub hard_nodes: Option<u64>,
    pub depth: Option<u64>,
    pub time_params: TimeParams,
}

impl SearchLimits {
    pub fn new(
        fischer: Option<FischerTime>,
        movetime: Option<u64>,
        soft_nodes: Option<u64>,
        hard_nodes: Option<u64>,
        depth: Option<u64>,
        fm_clock: usize,
    ) -> SearchLimits {
        let time_params = TimeParams::init();
        let (soft_time, hard_time) = match (fischer, movetime) {
            (Some(f), _) => {
                let (soft, hard) = Self::calc_time_limits(&time_params, f, fm_clock);
                (Some(soft), Some(hard))
            }
            (None, Some(mt)) => {
                let duration = Duration::from_millis(mt);
                (Some(duration), Some(duration))
            }
            (None, None) => (None, None),
        };

        SearchLimits {
            hard_time,
            soft_time,
            soft_nodes,
            hard_nodes,
            depth,
            time_params,
        }
    }

    pub fn init(&mut self) {
        self.time_params = TimeParams::init();
    }

    /// Scale the soft limit based on how stable we think the search is, and therefore how confident
    /// we are that the current best move is likely to be correct.
    pub fn scaled_soft_limit(
        &self,
        depth: i32,
        nodes: u64,
        best_move_nodes: u64,
        best_move_stability: u64,
        score_stability: u64,
    ) -> Option<Duration> {
        let p = &self.time_params;
        self.soft_time.map(|soft_time| {
            let scaled = soft_time.as_secs_f32()
                * Self::node_tm_scale(p, depth, nodes, best_move_nodes)
                * Self::best_move_stability_scale(p, best_move_stability)
                * Self::score_stability_scale(p, score_stability);
            Duration::from_secs_f32(scaled)
        })
    }

    /// 'Node TM': scale the soft limit based on the fraction of nodes that have been spent searching
    /// the current best move.
    fn node_tm_scale(p: &TimeParams, depth: i32, nodes: u64, best_move_nodes: u64) -> f32 {
        if depth < 4 || best_move_nodes == 0 {
            return 1.0;
        }
        let fraction = best_move_nodes as f32 / nodes as f32;
        (p.node_tm_base - fraction) * p.node_tm_scale
    }

    /// 'Best move stability': scale the soft limit based on how many iterations the best move has
    /// remained the same.
    fn best_move_stability_scale(p: &TimeParams, stability: u64) -> f32 {
        (p.best_move_tm_base - p.best_move_tm_scale * (stability as f32)).max(p.best_move_tm_min)
    }

    /// 'Score stability': scale the soft limit based on how stable the search score has been across
    /// iterations.
    fn score_stability_scale(p: &TimeParams, stability: u64) -> f32 {
        (p.score_tm_base - p.score_tm_scale * stability as f32).max(p.score_tm_min)
    }

    fn calc_time_limits(
        p: &TimeParams,
        fischer: FischerTime,
        fm_clock: usize,
    ) -> (Duration, Duration) {
        let (time, inc) = (fischer.0, fischer.1 as f64);
        let max_time = time.saturating_sub(UCI_OVERHEAD_MS);
        let soft_scale = p.soft_tm_base
            + p.soft_tm_scale * (1.0 - (-p.soft_tm_fm_scale * fm_clock as f64).exp());
        let soft_bound = (soft_scale * max_time as f64 + p.soft_tm_inc_scale * inc) as u64;
        let hard_bound =
            ((p.hard_tm_scale * max_time as f64 + p.hard_tm_inc_scale * inc) as u64).min(max_time);
        (
            Duration::from_millis(soft_bound),
            Duration::from_millis(hard_bound),
        )
    }
}
