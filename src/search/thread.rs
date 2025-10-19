use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::mpsc::{self, Sender, Receiver};
use std::time::Instant;
use crate::board::Board;
use crate::board::moves::Move;
use crate::evaluation::NNUE;
use crate::search::correction::CorrectionHistories;
use crate::search::history::Histories;
use crate::search::stack::SearchStack;
use crate::search::time::{LimitType, SearchLimits};
use crate::search::tt::TranspositionTable;
use crate::search::{search, Score, MAX_PLY};
use crate::tools::utils::boxed_and_zeroed;

pub struct ThreadPool {
    workers: Vec<WorkerThread>,
    job_sender: Sender<SearchJob>,
    result_receiver: Receiver<SearchResult>,
    global_stop: Arc<AtomicBool>,
    global_nodes: Arc<AtomicU64>,
}

pub struct WorkerThread {
    pub data: ThreadData<'static>,
    handle: std::thread::JoinHandle<()>,
}

pub struct SearchJob {
    pub board: Board,
    // Add more fields as needed (limits, etc)
}

pub struct SearchResult {
    pub id: usize,
    pub best_move: Move,
    pub best_score: i32,
}

impl ThreadPool {

    pub fn new(size: usize, tt: &TranspositionTable, global_stop: Arc<AtomicBool>) -> Self {
        let global_nodes = Arc::new(AtomicU64::new(0));
        let (job_sender, job_receiver) = mpsc::channel::<SearchJob>();
        let (result_sender, result_receiver) = mpsc::channel::<SearchResult>();
        let mut workers = Vec::with_capacity(size);
        for id in 0..size {
            let tt = tt.clone();
            let global_stop = global_stop.clone();
            let global_nodes = global_nodes.clone();
            let job_receiver = job_receiver.clone();
            let result_sender = result_sender.clone();
            let data = ThreadData::new(id, global_stop.clone(), &tt);
            let handle = std::thread::spawn(move || {
                let mut thread_data = data;
                loop {
                    match job_receiver.recv() {
                        Ok(job) => {
                            // Run search
                            thread_data.reset();
                            // You may want to set up thread_data with job.board, limits, etc
                            let (best_move, best_score) = search(&job.board, &mut thread_data);
                            let _ = result_sender.send(SearchResult {
                                id: thread_data.id,
                                best_move,
                                best_score,
                            });
                        },
                        Err(_) => break, // Channel closed
                    }
                }
            });
            workers.push(WorkerThread { data, handle });
        }
        Self {
            workers,
            job_sender,
            result_receiver,
            global_stop,
            global_nodes,
        }
    }

    pub fn dispatch_search(&self, board: &Board) {
        for _ in &self.workers {
            let job = SearchJob { board: board.clone() };
            let _ = self.job_sender.send(job);
        }
    }

    pub fn collect_results(&self) -> Vec<SearchResult> {
        let mut results = Vec::with_capacity(self.workers.len());
        for _ in &self.workers {
            if let Ok(res) = self.result_receiver.recv() {
                results.push(res);
            }
        }
        results
    }
}

pub struct ThreadData<'a> {
    pub id: usize,
    pub global_stop: Arc<AtomicBool>,
    pub minimal_output: bool,
    pub use_soft_nodes: bool,
    pub tt: &'a TranspositionTable,
    pub pv: PrincipalVariationTable,
    pub ss: SearchStack,
    pub nnue: NNUE,
    pub keys: Vec<u64>,
    pub root_ply: usize,
    pub history: Histories,
    pub correction_history: CorrectionHistories,
    pub lmr: LmrTable,
    pub node_table: NodeTable,
    pub limits: SearchLimits,
    pub start_time: Instant,
    pub nodes: u64,
    pub depth: i32,
    pub seldepth: usize,
    pub nmp_min_ply: i32,
    pub best_move: Move,
    pub best_score: i32,
}

impl<'a> ThreadData<'a> {
    pub fn new(id: usize, global_stop: Arc<AtomicBool>, tt: &'a TranspositionTable) -> Self {
        ThreadData {
            id,
            tt,
            global_stop,
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
            limits: SearchLimits::new(None, None, None, None, None),
            start_time: Instant::now(),
            nodes: 0,
            depth: 0,
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
        self.tt.clear();
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
