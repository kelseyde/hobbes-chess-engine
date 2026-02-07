use std::time::Duration;
use crate::search::score::Score;

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
}

pub enum LimitType {
    Soft,
    Hard,
}

pub type FischerTime = (u64, u64);

impl SearchLimits {
    pub fn new(
        fischer: Option<FischerTime>,
        movetime: Option<u64>,
        soft_nodes: Option<u64>,
        hard_nodes: Option<u64>,
        depth: Option<u64>,
        fm_clock: usize,
    ) -> SearchLimits {
        let (soft_time, hard_time) = match (fischer, movetime) {
            (Some(f), _) => {
                let (soft, hard) = Self::calc_time_limits(f, fm_clock);
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
        }
    }

    /// Scale the soft limit based on how stable we think the search is, and therefore how confident
    /// we are that the current best move is likely to be correct.
    pub fn scaled_soft_limit(
        &self,
        depth: i32,
        score: i32,
        nodes: u64,
        best_move_nodes: u64,
        best_move_stability: u64,
        score_stability: u64,
        root_qsearch_score: i32
    ) -> Option<Duration> {
        self.soft_time.map(|soft_time| {
            let scaled =
                soft_time.as_secs_f32()
                    * Self::node_tm_scale(depth, nodes, best_move_nodes)
                    * Self::best_move_stability_scale(best_move_stability)
                    * Self::score_stability_scale(score_stability)
                    * Self::root_complexity_scale(score, root_qsearch_score, depth);
            Duration::from_secs_f32(scaled)
        })
    }

    /// 'Node TM': scale the soft limit based on the fraction of nodes that have been spent searching
    /// the current best move. If we've spent a large fraction of our nodes on the current best move,
    /// then we should be more confident in it and spend less time, and vice versa.
    const fn node_tm_scale(depth: i32, nodes: u64, best_move_nodes: u64) -> f32 {
        if depth < 4 || best_move_nodes == 0 {
            return 1.0;
        }
        let fraction = best_move_nodes as f32 / nodes as f32;
        (1.5 - fraction) * 1.35
    }

    /// 'Best move stability': scale the soft limit based on how many iterations the best move has
    /// remained the same. If the best move has been stable for many iterations, then we should be
    /// more confident in it and spend less time, and vice versa.
    const fn best_move_stability_scale(stability: u64) -> f32 {
        (1.8 - 0.1 * (stability as f32)).max(0.9)
    }

    /// 'Score stability': scale the soft limit based on how stable the search score has been across
    /// iterations. If the score has been stable for many iterations, then we should be more confident
    /// in the search score and spend less time, and vice versa.
    const fn score_stability_scale(stability: u64) -> f32 {
        (1.2 - 0.04 * stability as f32).max(0.88)
    }

fn root_complexity_scale(score: i32, root_qsearch_score: i32, depth: i32) -> f32 {
    let complexity = if Score::is_mate(score) {
        0.0
    } else {
        0.87 * (root_qsearch_score - score).abs() as f64 * (depth as f64).ln()
    };

    (0.78 + complexity.clamp(0.0, 200.0) / 382.0).max(1.0) as f32
}

    fn calc_time_limits(fischer: FischerTime, fm_clock: usize) -> (Duration, Duration) {
        let (time, inc) = (fischer.0, fischer.1);
        // Credit to Reckless for this formula for calculating the hard/soft bounds
        let soft_scale = 0.024 + 0.042 * (1.0 - (-0.045 * fm_clock as f64).exp());
        let hard_scale = 0.742;
        let soft_bound = (soft_scale * time.saturating_sub(50) as f64 + 0.75 * inc as f64) as u64;
        let hard_bound = (hard_scale * time.saturating_sub(50) as f64 + 0.75 * inc as f64) as u64;
        (
            Duration::from_millis(soft_bound),
            Duration::from_millis(hard_bound),
        )
    }
}
