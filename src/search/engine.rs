use crate::board::Board;
use crate::evaluation::NNUE;
use crate::search::search;
use crate::search::thread::{SharedContext, ThreadData};
use crate::search::time::SearchLimits;
use crate::tools::bench::bench;
use crate::tools::channel::{self, Receiver, Sender};
use std::sync::atomic::{
    AtomicBool, AtomicU32,
    Ordering::{Acquire, Relaxed, Release},
};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

pub const MAX_THREADS: usize = 256;

/// Manages the UCI thread's state and a pool of N persistent search threads.
pub struct Engine {
    /// Position history for repetition detection.
    pub keys: Vec<u64>,
    /// Number of half-moves played from the root.
    pub root_ply: usize,
    /// Whether to print minimal UCI output (only the final info line before bestmove).
    pub minimal_output: bool,
    /// Whether the `nodes` UCI parameter is treated as a soft limit rather than a hard one.
    pub use_soft_nodes: bool,
    /// Shared transposition table + global node counter.
    shared: Arc<SharedContext>,
    /// Shared atomic flag that signals all threads to stop searching.
    abort: Arc<AtomicBool>,
    /// Pool of N persistent search threads. Dropped and recreated when the thread count changes.
    pool: Option<ThreadPool>,
    /// Number of threads to use for the next search.
    num_threads: usize,
    /// Number of threads currently searching. Main thread waits for helpers to finish before printing best move.
    num_searching: Arc<AtomicU32>,
}

/// A fixed-size pool of N persistent search threads.
/// Thread 0 is the "main" thread. It performs time management, waits for helpers to stop, and then
/// prints the best move. On `Drop`, the pool broadcasts `Quit` to all threads and joins them synchronously.
struct ThreadPool {
    sender: Sender<EngineCommand>,
    handles: Vec<JoinHandle<()>>,
}

/// Commands broadcast to every thread in the pool.
#[derive(Clone)]
enum EngineCommand {
    Search(SearchParams),
    Bench,
    UpdateShared(Arc<SharedContext>),
    NewGame,
    Quit,
}

/// Parameters broadcast to every search thread at the start of a search.
#[derive(Clone)]
struct SearchParams {
    board: Board,
    keys: Vec<u64>,
    root_ply: usize,
    limits: SearchLimits,
    start_time: Instant,
    minimal: bool,
    use_soft_nodes: bool,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Self {
        let shared = Arc::new(SharedContext::default());
        let abort = Arc::new(AtomicBool::new(false));
        Engine {
            keys: Vec::new(),
            root_ply: 0,
            minimal_output: false,
            use_soft_nodes: false,
            shared,
            abort,
            pool: None,
            num_threads: 1,
            num_searching: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Resize the transposition table. Rebuilds the `SharedContext` and re-assigns it to worker threads.
    pub fn set_hash(&mut self, mb: usize) {
        let shared = Arc::new(SharedContext::new(mb));
        self.shared = Arc::clone(&shared);
        if let Some(ref mut p) = self.pool {
            p.sender.send(EngineCommand::UpdateShared(shared));
        }
    }

    /// Return the transposition table size in megabytes.
    pub fn hash_mb(&self) -> usize {
        self.shared.tt.size_mb()
    }

    /// Update the number of search threads. Takes effect on the next `go`.
    pub fn set_threads(&mut self, n: usize) {
        let n = n.clamp(1, MAX_THREADS);
        if n != self.num_threads {
            self.num_threads = n;
            self.pool = None;
        }
    }

    /// Return the number of configured search threads.
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Set whether the engine should print minimal output (only the last info line before bestmove).
    pub fn set_minimal_output(&mut self, value: bool) {
        self.minimal_output = value;
    }

    pub fn set_use_soft_nodes(&mut self, value: bool) {
        self.use_soft_nodes = value;
    }

    pub fn use_soft_nodes(&self) -> bool {
        self.use_soft_nodes
    }

    /// Get the NNUE evaluation of the current board.
    pub fn eval(&self, board: &Board) -> i32 {
        let mut nnue = NNUE::default();
        nnue.activate(board);
        nnue.evaluate(board)
    }

    pub fn new_td(&self) -> Box<ThreadData> {
        Box::new(ThreadData::new(
            0,
            false,
            Arc::clone(&self.shared),
            Arc::clone(&self.abort),
        ))
    }

    /// Start a search. Broadcasts `Search(params)` to all threads in the pool.
    pub fn go(&mut self, board: Board, limits: SearchLimits) {
        self.ensure_pool();

        let params = SearchParams {
            board,
            keys: self.keys.clone(),
            root_ply: self.root_ply,
            limits,
            start_time: Instant::now(),
            minimal: self.minimal_output,
            use_soft_nodes: self.use_soft_nodes,
        };

        self.abort.store(false, Relaxed);
        self.shared.nodes.store(0, Relaxed);
        self.shared.tt.birthday();
        self.num_searching.store(self.num_threads as u32, Relaxed);

        self.pool
            .as_mut()
            .unwrap()
            .sender
            .send(EngineCommand::Search(params));
    }

    /// Bench is executed by the main thread only.
    pub fn bench(&mut self) {
        self.ensure_pool();
        self.num_searching.store(1, Relaxed);
        self.pool
            .as_mut()
            .unwrap()
            .sender
            .send(EngineCommand::Bench);
    }

    pub fn stop(&self) {
        self.abort.store(true, Relaxed);
    }

    /// Block until the current search finishes and reclaim the thread vec.
    pub fn join(&mut self) {
        let mut n = self.num_searching.load(Acquire);
        while n > 0 {
            atomic_wait::wait(&*self.num_searching, n);
            n = self.num_searching.load(Acquire);
        }
    }

    /// Whether a search is currently running.
    pub fn searching(&self) -> bool {
        self.num_searching.load(Relaxed) > 0
    }

    /// Clear the TT and all thread-local state. Called on `ucinewgame`.
    pub fn new_game(&mut self) {
        self.shared.tt.clear();
        self.keys.clear();
        self.root_ply = 0;
        if let Some(ref mut p) = self.pool {
            p.sender.send(EngineCommand::NewGame);
        }
    }

    /// Rebuild the pool if the thread count has changed.
    fn ensure_pool(&mut self) {
        let wanted = self.num_threads;
        let actual = self.pool.as_ref().map_or(0, |p| p.num_threads());
        if actual == wanted {
            return;
        }
        self.pool = None;
        self.pool = Some(ThreadPool::new(
            Arc::clone(&self.shared),
            Arc::clone(&self.abort),
            Arc::clone(&self.num_searching),
            wanted,
        ));
    }
}

impl ThreadPool {
    fn new(
        shared: Arc<SharedContext>,
        abort: Arc<AtomicBool>,
        num_searching: Arc<AtomicU32>,
        num_threads: usize,
    ) -> Self {
        // Create a channel with `num_threads` receivers, one for each search thread.
        let (sender, receivers) = channel::channel::<EngineCommand>(num_threads as u32);
        let handles = (0..num_threads)
            .zip(receivers)
            .map(|(id, rx)| {
                let main_thread = id == 0;
                let shared = Arc::clone(&shared);
                let abort = Arc::clone(&abort);
                let td = Box::new(ThreadData::new(id, main_thread, shared, abort));
                let ns = Arc::clone(&num_searching);
                std::thread::spawn(move || run_thread(td, rx, ns))
            })
            .collect();
        ThreadPool { sender, handles }
    }

    fn num_threads(&self) -> usize {
        self.handles.len()
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.sender.send(EngineCommand::Quit);
        for handle in self.handles.drain(..) {
            handle.join().expect("search thread panicked");
        }
    }
}

/// Entry point for every search thread, both main and helpers.
fn run_thread(
    mut td: Box<ThreadData>,
    mut rx: Receiver<EngineCommand>,
    num_searching: Arc<AtomicU32>,
) {
    // Wait until we have received a new message from the sender.
    loop {
        let msg = rx.recv(|msg| msg.clone());
        match msg {
            EngineCommand::Search(params) => {
                // Prepare this thread's local state.
                td.keys = params.keys;
                td.root_ply = params.root_ply;
                td.minimal_output = params.minimal;
                td.use_soft_nodes = params.use_soft_nodes;
                td.reset_local();
                td.start_time = params.start_time;
                td.limits = params.limits;

                search(&params.board, &mut td);

                if td.main {
                    // Main thread: wait for all helpers to finish, then print the bestmove.
                    let mut n = num_searching.load(Acquire);
                    while n > 1 {
                        atomic_wait::wait(&*num_searching, n);
                        n = num_searching.load(Acquire);
                    }
                    // TODO thread voting/meritocracy
                    println!("bestmove {}", td.best_move.to_uci());
                    num_searching.store(0, Release);
                    atomic_wait::wake_all(&*num_searching);
                } else {
                    // Helper thread: decrement the num_searching counter and wake the main thread.
                    num_searching.fetch_sub(1, Release);
                    atomic_wait::wake_all(&*num_searching);
                }
            }
            EngineCommand::Bench => {
                if td.main {
                    bench(&mut td);
                    num_searching.store(0, Release);
                    atomic_wait::wake_all(&*num_searching);
                }
            }
            EngineCommand::UpdateShared(shared) => td.shared = shared,
            EngineCommand::NewGame => td.clear_local(),
            EngineCommand::Quit => break,
        }
    }
}
