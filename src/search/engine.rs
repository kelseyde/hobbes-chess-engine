use crate::board::Board;
use crate::search::search;
use crate::search::thread::{SharedContext, ThreadData};
use crate::search::time::SearchLimits;
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
    /// UCI-side thread data for holding settings, position state, and running NNUE evals.
    td: Box<ThreadData>,
    /// Pool of N persistent search threads. Dropped and recreated when the thread count changes.
    pool: Option<ThreadPool>,
    /// Number of threads to use for the next search.
    num_threads: usize,
    /// Shared atomic flag that signals all threads to stop searching.
    abort: Arc<AtomicBool>,
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
    /// Start a new search with the given parameters.
    Search(SearchParams),
    /// Clear local search state.
    NewGame,
    /// Shut the thread down cleanly.
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
        let main = Box::new(ThreadData::default());
        let abort = Arc::clone(&main.abort);
        Engine {
            td: main,
            pool: None,
            num_threads: 1,
            abort,
            num_searching: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Resize the transposition table. Rebuilds the `SharedContext` and re-assigns it to worker threads.
    pub fn set_hash(&mut self, mb: usize) {
        let shared = Arc::new(SharedContext::new(mb));
        self.td.shared = Arc::clone(&shared);
        self.pool = None;
    }

    /// Return the transposition table size in megabytes.
    pub fn hash_mb(&self) -> usize {
        self.td.tt().size_mb()
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
        self.td.minimal_output = value;
    }

    pub fn set_use_soft_nodes(&mut self, value: bool) {
        self.td.use_soft_nodes = value;
    }

    pub fn use_soft_nodes(&self) -> bool {
        self.td.use_soft_nodes
    }

    /// Immutable access to the main thread's data.
    pub fn td(&self) -> &ThreadData {
        &self.td
    }

    /// Mutable access to the main thread's data.
    pub fn td_mut(&mut self) -> &mut ThreadData {
        &mut self.td
    }

    pub fn eval(&mut self, board: Board) -> i32 {
        let td = self.td_mut();
        td.nnue.activate(&board);
        td.nnue.evaluate(&board)
    }

    /// Start a search. Broadcasts `Search(params)` to all threads in the pool.
    pub fn go(&mut self, board: Board, limits: SearchLimits) {
        self.ensure_pool();

        let start_time = Instant::now();
        let params = SearchParams {
            board,
            keys: self.td.keys.clone(),
            root_ply: self.td.root_ply,
            limits,
            start_time,
            minimal: self.td.minimal_output,
            use_soft_nodes: self.td.use_soft_nodes,
        };

        // Reset shared state once, before any thread starts searching.
        self.abort.store(false, Relaxed);
        self.td.shared.nodes.store(0, Relaxed);
        self.td.tt().birthday();
        self.num_searching.store(self.num_threads as u32, Relaxed);

        self.pool
            .as_mut()
            .unwrap()
            .sender
            .send(EngineCommand::Search(params));
    }

    /// Set the global abort flag, signalling all search threads to stop.
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
        self.td.clear();
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
        let shared = Arc::clone(&self.td.shared);
        let abort = Arc::clone(&self.td.abort);
        self.pool = Some(ThreadPool::new(
            wanted,
            shared,
            abort,
            Arc::clone(&self.num_searching),
        ));
    }
}

impl ThreadPool {
    fn new(
        num_threads: usize,
        shared: Arc<SharedContext>,
        abort: Arc<AtomicBool>,
        num_searching: Arc<AtomicU32>,
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
            EngineCommand::NewGame => td.clear_local(),
            EngineCommand::Quit => break,
        }
    }
}