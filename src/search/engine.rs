use crate::board::Board;
use crate::search::parameters::thread_weight_score_offset;
use crate::search::score;
use crate::search::search;
use crate::search::thread::{SharedContext, ThreadData};
use crate::search::time::SearchLimits;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

pub const MAX_THREADS: usize = 256;

pub struct Engine {
    threads: Option<Vec<Box<ThreadData>>>,
    num_threads: usize,
    abort: Arc<AtomicBool>,
    handle: Option<JoinHandle<Vec<Box<ThreadData>>>>,
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
            threads: Some(vec![main]),
            num_threads: 1,
            abort,
            handle: None,
        }
    }

    /// Resize the transposition table. Rebuilds the `SharedContext` and re-assigns it to worker threads.
    pub fn set_hash(&mut self, mb: usize) {
        let shared = Arc::new(SharedContext::new(mb));
        for td in self.threads.as_mut().unwrap().iter_mut() {
            td.shared = Arc::clone(&shared);
        }
    }

    /// Return the transposition table size in megabytes.
    pub fn hash_mb(&self) -> usize {
        self.threads.as_ref().unwrap()[0].tt().size_mb()
    }

    /// Update the number of search threads. Takes effect on the next `go`.
    pub fn set_threads(&mut self, n: usize) {
        self.num_threads = n.clamp(1, MAX_THREADS);
    }

    /// Return the number of configured search threads.
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Set whether the engine should print minimal output (only the last info line before bestmove).
    pub fn set_minimal_output(&mut self, value: bool) {
        self.threads.as_mut().unwrap()[0].minimal_output = value;
    }

    pub fn set_use_soft_nodes(&mut self, value: bool) {
        self.threads.as_mut().unwrap()[0].use_soft_nodes = value;
    }

    pub fn use_soft_nodes(&self) -> bool {
        self.threads.as_ref().unwrap()[0].use_soft_nodes
    }

    /// Immutable access to the main thread's data.
    pub fn td(&self) -> &ThreadData {
        &self.threads.as_ref().unwrap()[0]
    }

    /// Mutable access to the main thread's data.
    pub fn td_mut(&mut self) -> &mut ThreadData {
        &mut self.threads.as_mut().unwrap()[0]
    }

    pub fn eval(&mut self, board: Board) -> i32 {
        let td = self.td_mut();
        td.nnue.activate(&board);
        td.nnue.evaluate(&board)
    }

    /// Start a new search on the given `board` with the given `limits`. Spawns a coordinator
    /// thread that runs the main search alongside `num_threads - 1` helpers.
    pub fn go(&mut self, board: Board, limits: SearchLimits) {
        self.sync_thread_pool();

        let mut threads = self.threads.take().unwrap();

        let start_time = Instant::now();
        let keys = threads[0].keys.clone();
        let root_ply = threads[0].root_ply;
        let minimal = threads[0].minimal_output;
        let use_soft_nodes = threads[0].use_soft_nodes;

        // Configure the main thread.
        threads[0].reset();
        threads[0].start_time = start_time;
        threads[0].limits = limits.clone();

        // Configure helpers.
        for helper in threads[1..].iter_mut() {
            helper.keys = keys.clone();
            helper.root_ply = root_ply;
            helper.minimal_output = minimal;
            helper.use_soft_nodes = use_soft_nodes;
            helper.reset_local();
            helper.start_time = start_time;
            helper.limits = limits.clone();
        }

        // Age the TT and clear the abort flag.
        threads[0].tt().birthday();
        self.abort.store(false, Relaxed);

        self.handle = Some(std::thread::spawn(move || {
            std::thread::scope(|s| {
                let (main_td, helpers) = threads.split_first_mut().unwrap();
                let abort = Arc::clone(&main_td.abort);
                for helper in helpers.iter_mut() {
                    let guard = AbortOnPanic(Arc::clone(&abort));
                    s.spawn(move || {
                        let _guard = guard;
                        search(&board, helper);
                    });
                }
                let _guard = AbortOnPanic(Arc::clone(&abort));
                search(&board, main_td);
            });

            let best_idx = select_best_thread(&threads);
            println!("bestmove {}", threads[best_idx].best_move.to_uci());
            threads
        }));
    }

    /// Set the global abort flag, signalling all search threads to stop.
    pub fn stop(&self) {
        self.abort.store(true, Relaxed);
    }

    /// Block until the current search finishes and reclaim the thread vec.
    pub fn join(&mut self) {
        if let Some(handle) = self.handle.take() {
            match handle.join() {
                Ok(threads) => self.threads = Some(threads),
                Err(_) => {
                    eprintln!("info string error: main thread panicked, reinitialising");
                    let main = Box::new(ThreadData::default());
                    self.abort = Arc::clone(&main.abort);
                    self.threads = Some(vec![main]);
                }
            }
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

    /// Clear the TT and all thread-local state. Called on `ucinewgame`.
    pub fn new_game(&mut self) {
        let threads = self.threads.as_mut().unwrap();
        threads[0].clear();
        for helper in threads[1..].iter_mut() {
            helper.clear_local();
        }
    }

    /// Ensure the thread vec has exactly `num_threads` entries. `threads[0]` (the main thread) is
    /// always retained. Surplus helpers are dropped and new ones are created as needed.
    fn sync_thread_pool(&mut self) {
        let threads = self.threads.as_mut().unwrap();
        threads.truncate(self.num_threads);

        let shared = Arc::clone(&threads[0].shared);
        let abort = Arc::clone(&threads[0].abort);

        while threads.len() < self.num_threads {
            let id = threads.len();
            threads.push(Box::new(ThreadData::new(
                id,
                false,
                Arc::clone(&shared),
                Arc::clone(&abort),
            )));
        }

        // Re-point every helper at the current shared context in case they changed (e.g. after the
        // TT was resized).
        for helper in threads[1..].iter_mut() {
            helper.shared = Arc::clone(&shared);
            helper.abort = Arc::clone(&abort);
        }

        self.abort = abort;
    }
}

/// Select the best thread to use for the final bestmove. Each thread contributes its weight to the
/// total score for its best move. The thread whose best_move has the highest accumulated score wins.
fn select_best_thread(threads: &[Box<ThreadData>]) -> usize {
    if threads.len() == 1 {
        return 0;
    }

    let lowest_root_score = threads
        .iter()
        .filter(|td| score::is_defined(td.best_score))
        .map(|td| td.best_score)
        .min()
        .unwrap_or(score::MIN);

    let thread_weight = |td: &ThreadData| -> i64 {
        (td.best_score - lowest_root_score + thread_weight_score_offset()) as i64
            * td.completed_depth as i64
    };

    // Accumulate weights per move.
    let mut move_scores: HashMap<u16, i64> = HashMap::new();
    for td in threads.iter().filter(|td| score::is_defined(td.best_score)) {
        *move_scores.entry(td.best_move.0).or_insert(0) += thread_weight(td);
    }

    // Pick the thread whose best_move has the highest accumulated score.
    threads
        .iter()
        .enumerate()
        .filter(|(_, td)| score::is_defined(td.best_score))
        .max_by_key(|(_, td)| move_scores.get(&td.best_move.0).copied().unwrap_or(0))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// A guard that sets the global abort flag to true if the thread panics, designed to prevent stalling.
struct AbortOnPanic(Arc<AtomicBool>);

impl Drop for AbortOnPanic {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.store(true, Relaxed);
        }
    }
}
