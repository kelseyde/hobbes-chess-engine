use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

use crate::board::moves::Move;
use crate::evaluation::NNUE;
use crate::search::correction::CorrectionHistories;
use crate::search::history::Histories;
use crate::search::stack::SearchStack;
use crate::search::time::{LimitType, SearchLimits};
use crate::search::tt::TranspositionTable;
use crate::search::{Score, MAX_PLY};
#[cfg(debug_assertions)]
use crate::tools::debug::DebugStatsMap;
use crate::tools::utils::boxed_and_zeroed;

pub struct ThreadPool {
    workers: Vec<WorkerThread>,
    shared: Arc<SharedContext>,
}

pub struct WorkerThread {
    data: Box<ThreadData>,
    handle: Option<std::thread::JoinHandle<()>>,
}

pub struct SharedContext {
    pub tt: TranspositionTable,
    pub stop: AtomicBool,
}

impl ThreadPool {

    pub fn new(tt_size_mb: usize) -> ThreadPool {
        let shared = Arc::new(SharedContext::new(tt_size_mb));
        ThreadPool {
            workers: Vec::new(),
            shared,
        }
    }

    pub fn resize(&mut self, num_threads: usize) {
        // Clear existing threads
        self.workers.clear();

        // Create new threads
        for id in 0..num_threads {
            self.workers.push(WorkerThread::new(id, Arc::clone(&self.shared)));
        }
    }

    pub fn main_thread(&mut self) -> &mut ThreadData {
        &mut self.workers[0].data
    }

    pub fn shared(&self) -> &Arc<SharedContext> {
        &self.shared
    }

    pub fn reset_all_threads(&mut self) {
        for worker in &mut self.workers {
            worker.data.reset();
        }
    }

    pub fn size(&self) -> usize {
        self.workers.len()
    }
}

impl WorkerThread {
    pub fn new(id: usize, shared: Arc<SharedContext>) -> Self {
        WorkerThread {
            data: Box::new(ThreadData::new(id, shared)),
            handle: None,
        }
    }
}

pub struct ThreadData {
    pub id: usize,
    pub shared: Arc<SharedContext>,
    pub minimal_output: bool,
    pub use_soft_nodes: bool,
    pub pv: PrincipalVariationTable,
    pub ss: SearchStack,
    pub nnue: NNUE,
    pub keys: Vec<u64>,
    pub root_ply: usize,
    pub history: Histories,
    pub correction_history: CorrectionHistories,
    pub lmr: LmrTable,
    pub node_table: NodeTable,
    #[cfg(debug_assertions)]
    pub debug_stats: DebugStatsMap,
    pub limits: SearchLimits,
    pub start_time: Instant,
    pub nodes: u64,
    pub depth: i32,
    pub seldepth: usize,
    pub nmp_min_ply: i32,
    pub best_move: Move,
    pub best_score: i32,
}

impl ThreadData {

    pub fn new(id: usize, shared: Arc<SharedContext>) -> Self {
        ThreadData {
            id,
            shared,
            minimal_output: false,
            use_soft_nodes: false,
            pv: PrincipalVariationTable::default(),
            ss: SearchStack::new(),
            nnue: NNUE::default(),
            keys: Vec::new(),
            root_ply: 0,
            history: Histories::default(),
            correction_history: CorrectionHistories::default(),
            lmr: LmrTable::default(),
            node_table: NodeTable::default(),
            #[cfg(debug_assertions)]
            debug_stats: DebugStatsMap::default(),
            limits: SearchLimits::new(None, None, None, None, None),
            start_time: Instant::now(),
            nodes: 0,
            depth: 1,
            seldepth: 0,
            nmp_min_ply: 0,
            best_move: Move::NONE,
            best_score: Score::MIN,
        }
    }

    pub fn reset(&mut self) {
        self.ss = SearchStack::new();
        self.node_table.clear();
        self.nodes = 0;
        self.depth = 1;
        self.seldepth = 0;
        self.best_move = Move::NONE;
        self.best_score = 0;
    }

    pub fn clear(&mut self) {
        self.shared.tt.clear();
        self.keys.clear();
        self.root_ply = 0;
        self.history.clear();
        self.correction_history.clear();
    }

    pub fn is_main_thread(&self) -> bool {
        self.id == 0
    }

    pub fn time(&self) -> u128 {
        self.start_time.elapsed().as_millis()
    }

    pub fn should_stop(&self, limit_type: LimitType) -> bool {
        if self.depth <= 1 {
            // Always clear the first depth, to ensure at least one legal move
            return false;
        }
        match limit_type {
            LimitType::Soft => self.soft_limit_reached(),
            LimitType::Hard => self.hard_limit_reached(),
        }
    }

    pub fn soft_limit_reached(&self) -> bool {
        let best_move_nodes = self.node_table.get(&self.best_move);

        if let Some(soft_time) =
            self.limits
                .scaled_soft_limit(self.depth, self.nodes, best_move_nodes)
        {
            if self.start_time.elapsed() >= soft_time {
                return true;
            }
        }

        if let Some(soft_nodes) = self.limits.soft_nodes {
            if self.nodes >= soft_nodes {
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

    pub fn hard_limit_reached(&self) -> bool {
        if let Some(hard_time) = self.limits.hard_time {
            if self.start_time.elapsed() >= hard_time {
                return true;
            }
        }

        if let Some(hard_nodes) = self.limits.hard_nodes {
            if self.nodes >= hard_nodes {
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

impl SharedContext {
    pub fn new(tt_size_mb: usize) -> Self {
        SharedContext {
            tt: TranspositionTable::new(tt_size_mb),
            stop: AtomicBool::new(false),
        }
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
    table: Box<[[i32; 64]; 256]>,
}

impl LmrTable {
    pub fn reduction(&self, depth: i32, move_count: i32) -> i32 {
        self.table[depth.min(255) as usize][move_count.min(63) as usize]
    }
}

impl Default for LmrTable {
    fn default() -> Self {
        let base = 0.92;
        let divisor = 3.11;

        let mut table: Box<[[i32; 64]; 256]> = unsafe { boxed_and_zeroed() };

        for depth in 1..256 {
            for move_count in 1..64 {
                let ln_depth = (depth as f32).ln();
                let ln_move_count = (move_count as f32).ln();
                let reduction = (base + (ln_depth * ln_move_count / divisor)) as i32;
                table[depth as usize][move_count as usize] = reduction;
            }
        }

        Self { table }
    }
}
