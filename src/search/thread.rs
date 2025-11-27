use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::mpsc::{channel, Sender};
use std::time::Instant;

use crate::board::Board;
use crate::board::moves::Move;
use crate::evaluation::NNUE;
use crate::search;
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
    sender: Sender<WorkerMessage>,
    handle: Option<std::thread::JoinHandle<()>>,
}

enum WorkerMessage {
    Search(Board, fn(&Board, &mut ThreadData) -> (Move, i32)),
    Shutdown,
}

pub struct SharedContext {
    pub tt: TranspositionTable,
    pub stop: AtomicBool,
}

pub struct MainThreadData {
    pub limits: SearchLimits,
    pub start_time: Instant,
    pub minimal_output: bool,
    pub use_soft_nodes: bool,
}

pub struct ThreadData {
    pub id: usize,
    pub shared: Arc<SharedContext>,
    pub main_thread_data: Option<MainThreadData>,
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
    pub nodes: u64,
    pub depth: i32,
    pub seldepth: usize,
    pub nmp_min_ply: i32,
    pub best_move: Move,
    pub best_score: i32,
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
        // Shutdown existing threads
        self.shutdown_workers();

        // Clear workers
        self.workers.clear();

        // Create new threads
        for id in 0..num_threads {
            self.workers.push(WorkerThread::spawn(id, Arc::clone(&self.shared)));
        }
    }

    fn shutdown_workers(&mut self) {
        // Send shutdown message to all workers
        for worker in &self.workers {
            worker.sender.send(WorkerMessage::Shutdown).ok();
        }

        // Wait for all threads to finish
        for worker in &mut self.workers {
            if let Some(handle) = worker.handle.take() {
                handle.join().ok();
            }
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

    /// Start all worker threads searching the given position
    pub fn start_search(&mut self, board: &Board) {
        // Reset stop flag
        self.shared.stop.store(false, Ordering::Relaxed);

        // Copy position state from main thread to all helper threads
        let main_thread = &self.workers[0].data;
        let keys = main_thread.keys.clone();
        let root_ply = main_thread.root_ply;

        // Send search message to worker threads
        for worker in self.workers.iter_mut() {
            worker.data.keys = keys.clone();
            worker.data.root_ply = root_ply;
            worker.data.reset();
            println!("Thread {} starting search with limits present? {}", worker.data.id, worker.data.limits().is_some());
            println!("soft time: {:?}, hard time: {:?}, soft nodes: {:?}, hard nodes: {:?}, depth: {:?}",
                     worker.data.limits().and_then(|l| l.soft_time),
                     worker.data.limits().and_then(|l| l.hard_time),
                     worker.data.limits().and_then(|l| l.soft_nodes),
                     worker.data.limits().and_then(|l| l.hard_nodes),
                     worker.data.limits().and_then(|l| l.depth),
            );

            let search_fn = search::search;

            worker.sender.send(WorkerMessage::Search(*board, search_fn)).ok();
        }

    }

    /// Signal all threads to stop searching.
    pub fn stop_search(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
    }

}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shutdown_workers();
    }
}

impl WorkerThread {
    pub fn spawn(id: usize, shared: Arc<SharedContext>) -> Self {
        let (sender, receiver) = channel();
        let mut data = Box::new(ThreadData::new(id, Arc::clone(&shared)));

        let handle = std::thread::spawn(move || {
            loop {
                match receiver.recv() {
                    Ok(WorkerMessage::Search(board, search_fn)) => {
                        search_fn(&board, &mut data);
                    }
                    Ok(WorkerMessage::Shutdown) | Err(_) => {
                        // Shutdown message or channel closed
                        break;
                    }
                }
            }
        });

        WorkerThread {
            data,
            sender,
            handle: Some(handle),
        }
    }

    pub fn new(id: usize, shared: Arc<SharedContext>) -> Self {
        Self::spawn(id, shared)
    }
}

impl Default for MainThreadData {
    fn default() -> Self {
        MainThreadData {
            limits: SearchLimits::new(None, None, None, None, None),
            start_time: Instant::now(),
            minimal_output: false,
            use_soft_nodes: false,
        }
    }
}

impl ThreadData {

    pub fn new(id: usize, shared: Arc<SharedContext>) -> Self {
        let main_thread_data = if id == 0 {
            Some(MainThreadData::default())
        } else {
            None
        };

        ThreadData {
            id,
            shared,
            main_thread_data,
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

    pub fn limits(&self) -> Option<&SearchLimits> {
        self.main_thread_data.as_ref().map(|data| &data.limits)
    }

    pub fn start_time(&self) -> Instant {
        self.main_thread_data
            .as_ref()
            .map_or(Instant::now(), |data| data.start_time)
    }

    pub fn minimal_output(&self) -> bool {
        self.main_thread_data
            .as_ref()
            .map_or(false, |data| data.minimal_output)
    }

    pub fn use_soft_nodes(&self) -> bool {
        self.main_thread_data
            .as_ref()
            .map_or(false, |data| data.use_soft_nodes)
    }

    pub fn should_stop(&self, limit_type: LimitType) -> bool {

        // Check the global stop flag first
        if self.shared.stop.load(Ordering::Relaxed) {
            return true;
        }

        // Always clear the first depth, to ensure at least one legal move
        if self.depth <= 1 {
            return false;
        }

        // Only main thread checks time/node limits for stopping
        if !self.is_main_thread() {
            println!("skipping limit check on thread {}", self.id);
            return false;
        }

        // Finally, check the configured search limits
        let limit_reached = match limit_type {
            LimitType::Soft => self.soft_limit_reached(),
            LimitType::Hard => self.hard_limit_reached(),
        };
        if limit_reached {
            self.shared.stop.store(true, Ordering::Relaxed);
        }
        limit_reached
    }

    pub fn soft_limit_reached(&self) -> bool {

        println!("Checking soft limit");
        println!("{}", self.limits().and_then(|l| l.soft_time).is_some());
        println!("{}", self.limits().and_then(|l| l.hard_time).is_some());

        println!("soft time: {:?}, hard time: {:?}, soft nodes: {:?}, hard nodes: {:?}, depth: {:?}",
                 self.limits().and_then(|l| l.soft_time),
                 self.limits().and_then(|l| l.hard_time),
                 self.limits().and_then(|l| l.soft_nodes),
                 self.limits().and_then(|l| l.hard_nodes),
                 self.limits().and_then(|l| l.depth),
        );

        let main_data = match &self.main_thread_data {
            Some(data) => data,
            None => {
                println!("no main thread data in soft limit check");
                return false
            },
        };

        let best_move_nodes = self.node_table.get(&self.best_move);

        if let Some(soft_time) = main_data.limits
                .scaled_soft_limit(self.depth, self.nodes, best_move_nodes)
        {
            println!("comparing elapsed {:?} to soft time {:?}", main_data.start_time.elapsed(), soft_time);
            if main_data.start_time.elapsed() >= soft_time {
                return true;
            }
        }

        if let Some(soft_nodes) = main_data.limits.soft_nodes {
            if self.nodes >= soft_nodes {
                return true;
            }
        }

        if let Some(depth_limit) = main_data.limits.depth {
            if self.depth >= depth_limit as i32 {
                return true;
            }
        }

        false
    }

    pub fn hard_limit_reached(&self) -> bool {
        // Only main thread checks time/node limits for stopping
        if !self.is_main_thread() {
            return false;
        }

        let main_data = match &self.main_thread_data {
            Some(data) => data,
            None => {
                println!("no main thread data in hard limit check");
                return false
            },
        };

        if let Some(hard_time) = main_data.limits.hard_time {
            println!("comparing elapsed {:?} to hard time {:?}", main_data.start_time.elapsed(), hard_time);
            if main_data.start_time.elapsed() >= hard_time {
                return true;
            }
        }

        if let Some(hard_nodes) = main_data.limits.hard_nodes {
            if self.nodes >= hard_nodes {
                return true;
            }
        }

        if let Some(depth_limit) = main_data.limits.depth {
            if self.depth >= depth_limit as i32 {
                return true;
            }
        }

        false
    }

    pub fn set_limits(&mut self, limits: SearchLimits) {
        if let Some(main_data) = &mut self.main_thread_data {
            main_data.limits = limits;
        }
    }

    pub fn set_start_time(&mut self, start_time: Instant) {
        if let Some(main_data) = &mut self.main_thread_data {
            main_data.start_time = start_time;
        }
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
