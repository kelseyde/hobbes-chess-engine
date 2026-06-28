use crate::board::Board;
use crate::search::channel::{self, Receiver, Sender};
use crate::search::thread::{SharedContext, ThreadData};
use crate::search::time::SearchLimits;
use crate::search::search;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

/// Maximum number of search threads that can be configured.
pub const MAX_THREADS: usize = 256;

// -----------------------------------------------------------------------
// Channel message types
// -----------------------------------------------------------------------

/// Data broadcast to all helper threads at the start of each search.
/// All fields must be `Sync` because the channel temporarily shares a
/// pointer to this value across threads.
struct SearchParams {
    board:         Board,
    keys:          Vec<u64>,
    root_ply:      usize,
    limits:        SearchLimits,
    start_time:    Instant,
    minimal:       bool,
    use_soft_nodes: bool,
}

/// Commands broadcast to helper threads via the pool channel.
enum HelperMsg {
    /// Start a new search with the given parameters.
    Search(SearchParams),
    /// Clear local search state (called on `ucinewgame`).
    NewGame,
    /// Shut the helper thread down cleanly.
    Quit,
}

// -----------------------------------------------------------------------
// Persistent thread pool
// -----------------------------------------------------------------------

/// A fixed-size pool of persistent helper search threads.
///
/// Helpers are created once when the pool is built and loop indefinitely,
/// waiting for commands via a broadcast channel. This avoids spawning new
/// OS threads on every `go` command.
///
/// Dropping the pool broadcasts a [`HelperMsg::Quit`] to all helpers,
/// blocking until every helper has acknowledged and exited.
struct ThreadPool {
    sender:      Sender<HelperMsg>,
    num_helpers: usize,
}

impl ThreadPool {
    fn new(num_helpers: usize, shared: Arc<SharedContext>, abort: Arc<AtomicBool>) -> Self {
        let (sender, rx_iter) = channel::channel::<HelperMsg>(num_helpers as u32);
        for (id, rx) in (1..=num_helpers).zip(rx_iter) {
            let td = Box::new(ThreadData::new(
                id,
                false,
                Arc::clone(&shared),
                Arc::clone(&abort),
            ));
            std::thread::spawn(move || run_helper(td, rx));
        }
        ThreadPool { sender, num_helpers }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Wake all helpers with Quit. Blocks until every helper acknowledges.
        self.sender.send(HelperMsg::Quit);
    }
}

/// Entry point for each persistent helper thread.
///
/// Loops waiting for channel messages:
/// - `Search`: configure `td` using [`ThreadData::reset_local`] (safe: does not
///   touch shared state owned by the main thread), then run the search.
/// - `NewGame`: clear local history tables.
/// - `Quit`: exit.
fn run_helper(mut td: Box<ThreadData>, mut rx: Receiver<HelperMsg>) {
    loop {
        let keep_going = rx.recv(|msg| match msg {
            HelperMsg::Search(params) => {
                // Initialise position state for this search.
                td.keys          = params.keys.clone();
                td.root_ply      = params.root_ply;
                td.minimal_output = params.minimal;
                td.use_soft_nodes = params.use_soft_nodes;

                // reset_local() resets only per-thread fields: it does NOT touch
                // shared.nodes (reset by the main thread before sending) or abort
                // (set to false by go() before the scope; helpers must not clear it
                // after the main thread may have already raised it).
                td.reset_local();
                td.start_time = params.start_time;
                td.limits     = params.limits.clone();
                search(&params.board, &mut td);
                true
            }
            HelperMsg::NewGame => {
                td.clear_local();
                true
            }
            HelperMsg::Quit => false,
        });
        if !keep_going {
            break;
        }
    }
}

// -----------------------------------------------------------------------
// Engine
// -----------------------------------------------------------------------

/// Manages the main search thread and a pool of persistent helper threads.
///
/// `Engine` owns the main [`ThreadData`] (id 0) directly. Helper threads
/// are owned by the [`ThreadPool`] and persist across searches, retaining
/// their history tables for search diversity.
///
/// Only the main thread performs time management. When a limit is reached
/// (or a UCI `stop` arrives) it raises the shared abort flag, and every
/// helper stops at its next [`ThreadData::should_stop`] check.
pub struct Engine {
    /// Main thread data. Moved into the coordinator thread during a search.
    main_td:     Option<Box<ThreadData>>,
    /// Persistent helper pool (ids 1..num_threads-1).
    pool:        Option<ThreadPool>,
    /// Configured thread count (1 = main only, N = main + N-1 helpers).
    num_threads: usize,
    /// Clone of the abort flag for the UCI `stop` command.
    abort:       Arc<AtomicBool>,
    /// Handle to the coordinator thread, while a search is running.
    handle:      Option<JoinHandle<(Box<ThreadData>, Option<ThreadPool>)>>,
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
            main_td: Some(main),
            pool: None,
            num_threads: 1,
            abort,
            handle: None,
        }
    }

    // --- Configuration ---------------------------------------------------

    /// Resize the transposition table shared by all threads.
    ///
    /// The existing pool is dropped (helpers receive `Quit` and exit), and
    /// a new pool will be built with the fresh shared context on the next `go`.
    pub fn set_hash(&mut self, mb: usize) {
        let shared = Arc::new(SharedContext::new(mb));
        self.main_td.as_mut().unwrap().shared = Arc::clone(&shared);
        // Drop old pool; helpers will be re-created with the new shared context.
        self.pool = None;
    }

    pub fn hash_mb(&self) -> usize {
        self.main_td.as_ref().unwrap().tt().size_mb()
    }

    /// Set the number of search threads. Takes effect on the next `go`.
    ///
    /// If the count changes the existing pool is dropped so it can be
    /// rebuilt at the right size.
    pub fn set_threads(&mut self, n: usize) {
        let n = n.clamp(1, MAX_THREADS);
        if n != self.num_threads {
            self.num_threads = n;
            self.pool = None;
        }
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    pub fn set_minimal_output(&mut self, value: bool) {
        self.main_td.as_mut().unwrap().minimal_output = value;
    }

    pub fn set_use_soft_nodes(&mut self, value: bool) {
        self.main_td.as_mut().unwrap().use_soft_nodes = value;
    }

    pub fn use_soft_nodes(&self) -> bool {
        self.main_td.as_ref().unwrap().use_soft_nodes
    }

    // --- ThreadData access -----------------------------------------------

    pub fn td(&self) -> &ThreadData {
        self.main_td.as_ref().unwrap()
    }

    pub fn td_mut(&mut self) -> &mut ThreadData {
        self.main_td.as_mut().unwrap()
    }

    // --- Search lifecycle ------------------------------------------------

    /// Start a search on the given `board` with the given `limits`.
    ///
    /// Spawns a coordinator thread that:
    /// 1. Broadcasts `Search(params)` to all helpers (blocking until they finish).
    /// 2. Concurrently runs the main search on the coordinator thread.
    ///
    /// Returns immediately; the coordinator runs in the background.
    pub fn go(&mut self, board: Board, limits: SearchLimits) {
        self.ensure_pool();

        let mut main_td = self.main_td.take().unwrap();
        let mut pool    = self.pool.take();

        let start_time = Instant::now();

        // Snapshot position state for helpers *before* resetting main_td.
        let params = SearchParams {
            board,
            keys:           main_td.keys.clone(),
            root_ply:       main_td.root_ply,
            limits:         limits.clone(),
            start_time,
            minimal:        main_td.minimal_output,
            use_soft_nodes: main_td.use_soft_nodes,
        };

        // Reset shared state once, before any thread starts searching.
        main_td.reset();
        main_td.start_time = start_time;
        main_td.limits     = limits;

        self.handle = Some(std::thread::spawn(move || {
            // thread::scope lets us borrow params/pool by reference from spawned threads,
            // while running the main search concurrently on the coordinator.
            // The scope does not exit until all spawned threads have returned, which means
            // all helpers have finished their searches before we continue.
            std::thread::scope(|s| {
                if let Some(ref mut p) = pool {
                    // The send thread broadcasts the Search message to all helpers.
                    // It blocks until every helper has finished its recv handler (i.e.
                    // finished searching), so we know helpers are done when the scope exits.
                    s.spawn(|| p.sender.send(HelperMsg::Search(params)));
                }
                // Main search runs concurrently with the helpers.
                search(&board, &mut main_td);
            });

            println!("bestmove {}", main_td.best_move.to_uci());
            (main_td, pool)
        }));
    }

    /// Signal the running search to stop. The main thread will end its current
    /// iteration, helpers will stop at their next `should_stop` check, and
    /// `bestmove` will be printed.
    pub fn stop(&self) {
        self.abort.store(true, Relaxed);
    }

    /// Block until the search finishes and reclaim both the main thread data
    /// and the pool.
    pub fn join(&mut self) {
        if let Some(handle) = self.handle.take() {
            let (main_td, pool) = handle.join().expect("coordinator thread panicked");
            self.main_td = Some(main_td);
            self.pool    = pool;
        }
    }

    /// Reclaim thread data only if the search has already finished (non-blocking).
    pub fn try_reclaim(&mut self) {
        if self.handle.as_ref().is_some_and(|h| h.is_finished()) {
            self.join();
        }
    }

    /// Whether a search is currently running.
    pub fn searching(&self) -> bool {
        self.handle.is_some()
    }

    // --- Game management -------------------------------------------------

    /// Clear the transposition table and all per-thread search state.
    /// Called on `ucinewgame`.
    pub fn new_game(&mut self) {
        self.main_td.as_mut().unwrap().clear();
        // Broadcast NewGame to helpers so they also clear their local state.
        if let Some(ref mut p) = self.pool {
            p.sender.send(HelperMsg::NewGame);
        }
    }

    // --- Private helpers -------------------------------------------------

    /// Ensure the pool contains exactly `num_threads - 1` helpers.
    ///
    /// Rebuilds the pool (sending `Quit` to old helpers first) only when the
    /// desired count differs from the current pool size. Existing helpers are
    /// retained to preserve their history tables.
    fn ensure_pool(&mut self) {
        let want    = self.num_threads.saturating_sub(1);
        let current = self.pool.as_ref().map_or(0, |p| p.num_helpers);

        if current == want {
            return;
        }

        // Drop old pool first (Quit broadcast is synchronous via ThreadPool::drop).
        self.pool = None;

        if want > 0 {
            let main   = self.main_td.as_ref().unwrap();
            let shared = Arc::clone(&main.shared);
            let abort  = Arc::clone(&main.abort);
            self.pool  = Some(ThreadPool::new(want, shared, abort));
        }
    }
}
