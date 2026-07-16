use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use std::time::Instant;

use crate::board::moves::Move;
use crate::evaluation::NNUE;
use crate::search::correction::CorrectionHistories;
use crate::search::history::Histories;
use crate::search::node::NodeStack;
use crate::search::parameters::{lmr_noisy_base, lmr_noisy_div, lmr_quiet_base, lmr_quiet_div};
use crate::search::time::{LimitType, SearchLimits};
use crate::search::tt::TranspositionTable;
use crate::search::{score, MAX_PLY};
use crate::tools::utils::boxed_and_zeroed;

/// State shared between all search threads.
pub struct SharedContext {
    pub tt: TranspositionTable,
    pub nodes: AtomicU64,
}

pub struct ThreadData {
    pub id: usize,
    pub main: bool,
    pub minimal_output: bool,
    pub use_soft_nodes: bool,
    pub shared: Arc<SharedContext>,
    pub abort: Arc<AtomicBool>,
    pub pv: PrincipalVariationTable,
    pub stack: NodeStack,
    pub nnue: NNUE,
    pub keys: Vec<u64>,
    pub root_ply: usize,
    pub history: Histories,
    pub correction_history: CorrectionHistories,
    pub lmr: LmrTable,
    pub node_table: NodeTable,
    pub limits: SearchLimits,
    pub start_time: Instant,
    pub best_move_stability: u32,
    pub score_stability: u32,
    pub local_nodes: u64,
    pub depth: i32,
    pub completed_depth: i32,
    pub seldepth: usize,
    pub nmp_min_ply: i32,
    pub best_move: Move,
    pub best_score: i32,
}

impl ThreadData {
    pub fn new(id: usize, main: bool, shared: Arc<SharedContext>, abort: Arc<AtomicBool>) -> Self {
        ThreadData {
            id,
            main,
            minimal_output: false,
            use_soft_nodes: false,
            shared,
            abort,
            pv: PrincipalVariationTable::default(),
            stack: NodeStack::default(),
            nnue: NNUE::default(),
            keys: Vec::new(),
            root_ply: 0,
            history: Histories::default(),
            correction_history: CorrectionHistories::default(),
            lmr: LmrTable::default(),
            node_table: NodeTable::default(),
            limits: SearchLimits::new(None, None, None, None, None, 0),
            start_time: Instant::now(),
            best_move_stability: 0,
            score_stability: 0,
            local_nodes: 0,
            depth: 1,
            completed_depth: 0,
            seldepth: 0,
            nmp_min_ply: 0,
            best_move: Move::NONE,
            best_score: score::MIN,
        }
    }
}

impl Default for ThreadData {
    fn default() -> Self {
        Self::new(
            0,
            true,
            Arc::new(SharedContext::default()),
            Arc::new(AtomicBool::new(false)),
        )
    }
}

impl SharedContext {
    pub fn new(tt_size_mb: usize) -> SharedContext {
        SharedContext {
            tt: TranspositionTable::new(tt_size_mb),
            nodes: AtomicU64::new(0),
        }
    }
}

impl Default for SharedContext {
    fn default() -> Self {
        SharedContext::new(64)
    }
}

impl ThreadData {
    #[inline]
    pub fn tt(&self) -> &TranspositionTable {
        &self.shared.tt
    }

    /// The total number of nodes searched across all threads.
    #[inline]
    pub fn nodes(&self) -> u64 {
        self.shared.nodes.load(Relaxed)
    }

    /// This thread's own node count.
    #[inline]
    pub fn local_nodes(&self) -> u64 {
        self.local_nodes
    }

    /// Increment both this thread's local counter and the shared global node counter.
    #[inline]
    pub fn inc_nodes(&mut self) {
        self.local_nodes += 1;
        self.shared.nodes.fetch_add(1, Relaxed);
    }

    pub fn reset(&mut self) {
        self.shared.nodes.store(0, Relaxed);
        self.abort.store(false, Relaxed);
        self.reset_local();
    }

    pub fn reset_local(&mut self) {
        self.stack = NodeStack::default();
        self.node_table.clear();
        self.shared.nodes.store(0, Relaxed);
        self.local_nodes = 0;
        self.abort.store(false, Relaxed);
        self.depth = 1;
        self.completed_depth = 0;
        self.seldepth = 0;
        self.best_move = Move::NONE;
        self.best_score = 0;
        self.best_move_stability = 0;
        self.score_stability = 0;
    }

    /// Clear the (shared) transposition table and this thread's per-thread search tables.
    pub fn clear(&mut self) {
        self.tt().clear();
        self.clear_local();
    }

    /// Clear this thread's local data, excluding the TT, which is managed by the main thread.
    pub fn clear_local(&mut self) {
        self.keys.clear();
        self.root_ply = 0;
        self.history.clear();
        self.correction_history.clear();
    }

    pub fn time(&self) -> u128 {
        self.start_time.elapsed().as_millis()
    }

    pub fn should_stop(&self, limit_type: LimitType) -> bool {
        // Always complete the first depth, to ensure at least one legal move.
        if self.depth <= 1 {
            return false;
        }
        // Only check time management if we are the main thread.
        if !self.main {
            return self.abort.load(Relaxed);
        }
        // The main thread aborts immediately on an external stop (e.g. the UCI 'stop' command).
        if self.abort.load(Relaxed) {
            return true;
        }
        let stop = match limit_type {
            LimitType::Soft => self.soft_limit_reached(),
            LimitType::Hard => self.hard_limit_reached(),
        };
        if stop {
            self.abort.store(true, Relaxed);
        }
        stop
    }

    fn soft_limit_reached(&self) -> bool {
        let best_move_nodes = self.node_table.get(&self.best_move);
        let best_move_stability = self.best_move_stability as u64;
        let score_stability = self.score_stability as u64;

        if let Some(soft_time) = self.limits.scaled_soft_limit(
            self.depth,
            self.local_nodes,
            best_move_nodes,
            best_move_stability,
            score_stability,
        ) {
            if self.start_time.elapsed() >= soft_time {
                return true;
            }
        }

        if let Some(soft_nodes) = self.limits.soft_nodes {
            if self.nodes() >= soft_nodes {
                return true;
            }
        }

        if let Some(depth_limit) = self.limits.depth {
            if self.depth >= depth_limit as i32 {
                return true;
            }
        }

        false
    }

    fn hard_limit_reached(&self) -> bool {
        if let Some(hard_nodes) = self.limits.hard_nodes {
            if self.nodes() >= hard_nodes {
                return true;
            }
        }

        // Only check hard time/depth limits every 2048 nodes to reduce overhead.
        if self.local_nodes % 2048 != 0 {
            return false;
        }

        if let Some(hard_time) = self.limits.hard_time {
            if self.start_time.elapsed() >= hard_time {
                return true;
            }
        }

        if let Some(depth_limit) = self.limits.depth {
            if self.depth >= depth_limit as i32 {
                return true;
            }
        }

        false
    }
}

pub struct NodeTable {
    table: Box<[[u64; 64]; 64]>,
}

impl Default for NodeTable {
    fn default() -> Self {
        NodeTable {
            table: unsafe { boxed_and_zeroed() },
        }
    }
}

impl NodeTable {
    pub fn add(&mut self, mv: &Move, nodes: u64) {
        self.table[mv.from()][mv.to()] += nodes;
    }

    pub fn get(&self, mv: &Move) -> u64 {
        self.table[mv.from()][mv.to()]
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

pub struct PrincipalVariationTable {
    table: Box<[[Move; MAX_PLY + 1]; MAX_PLY + 1]>,
    len: [usize; MAX_PLY + 1],
}

impl PrincipalVariationTable {
    pub const fn best_move(&self) -> Move {
        self.table[0][0]
    }

    pub fn line(&self) -> &[Move] {
        &self.table[0][..self.len[0]]
    }

    pub fn clear(&mut self, ply: usize) {
        self.len[ply] = 0;
    }

    pub fn update(&mut self, ply: usize, mv: Move) {
        self.table[ply][0] = mv;
        self.len[ply] = self.len[ply + 1] + 1;

        for i in 0..self.len[ply + 1] {
            self.table[ply][i + 1] = self.table[ply + 1][i];
        }
    }
}

impl Default for PrincipalVariationTable {
    fn default() -> Self {
        Self {
            table: unsafe { boxed_and_zeroed() },
            len: [0; MAX_PLY + 1],
        }
    }
}

pub struct LmrTable {
    table: Box<[[[i32; 2]; 64]; 256]>,
}

impl LmrTable {
    pub fn reduction(&self, depth: i32, move_count: i32, is_quiet: bool) -> i32 {
        self.table[depth.min(255) as usize][move_count.min(63) as usize][is_quiet as usize]
    }

    pub fn init(&mut self) {
        let quiet_base = lmr_quiet_base() as f32 / 100.0;
        let quiet_divisor = lmr_quiet_div() as f32 / 100.0;
        let noisy_base = lmr_noisy_base() as f32 / 100.0;
        let noisy_divisor = lmr_noisy_div() as f32 / 100.0;

        for depth in 1..256 {
            for move_count in 1..64 {
                for is_quiet in [true, false] {
                    let base = if is_quiet { quiet_base } else { noisy_base };
                    let divisor = if is_quiet {
                        quiet_divisor
                    } else {
                        noisy_divisor
                    };
                    let ln_depth = (depth as f32).ln();
                    let ln_move_count = (move_count as f32).ln();
                    let reduction = (base + (ln_depth * ln_move_count / divisor)) as i32;
                    self.table[depth as usize][move_count as usize][is_quiet as usize] = reduction;
                }
            }
        }
    }
}

impl Default for LmrTable {
    fn default() -> Self {
        unsafe {
            Self {
                table: boxed_and_zeroed(),
            }
        }
    }
}
