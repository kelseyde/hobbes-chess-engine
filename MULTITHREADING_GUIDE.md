# Multithreading Guide

## Overview

Your chess engine now supports multithreading with a shared `TranspositionTable` and global `stop` flag.

## Architecture

### Key Components

1. **SharedContext** - Contains data shared across all threads:
   - `tt: TranspositionTable` - The transposition table
   - `stop: AtomicBool` - Global stop flag

2. **ThreadPool** - Manages worker threads:
   - Holds an `Arc<SharedContext>` that's cloned for each thread
   - Contains a vector of `WorkerThread` instances

3. **ThreadData** - Per-thread data:
   - Holds an `Arc<SharedContext>` for accessing shared data
   - Each thread has its own NNUE, histories, search stack, etc.

## How Arc Works

`Arc` (Atomic Reference Counted) is a smart pointer that allows multiple ownership:
- Multiple threads can have `Arc<SharedContext>` clones
- The data is only dropped when the last `Arc` is dropped
- `Arc::clone()` is cheap - it just increments a counter

## Thread Safety

### TranspositionTable
- Uses `UnsafeCell` for interior mutability
- Methods like `insert()`, `probe()`, `clear()` take `&self` instead of `&mut self`
- This allows multiple threads to write/read concurrently
- **Note**: Currently no synchronization - race conditions possible but acceptable for TT

### AtomicBool (stop flag)
- Use `stop.store(true, Ordering::Relaxed)` to set
- Use `stop.load(Ordering::Relaxed)` to read
- Thread-safe without locks

## Usage Example

```rust
// Create a thread pool with 4 threads
let mut pool = ThreadPool::new(64); // 64 MB TT
pool.resize(4);

// Access the main thread
let main_td = pool.main_thread();

// Access shared context
let shared = pool.shared();
shared.tt.insert(...);  // Write to TT
shared.stop.store(true, Ordering::Relaxed);  // Signal stop

// Each thread can access the shared TT independently
// In your search function, just use: td.shared.tt.probe(hash)
```

## When to Use Arc vs References

### Use `Arc` when:
- You need to share data across multiple threads
- Ownership needs to be shared (multiple owners)
- Data outlives the original owner
- Example: `SharedContext` in this engine

### Use references (`&` or `&mut`) when:
- Single-threaded code
- Temporary borrowing within the same thread
- Clear ownership hierarchy
- Example: `Board` passed to `search()`

## Lifetimes

With `Arc`, you don't need explicit lifetimes (`'a`) because:
- `Arc` owns the data
- No borrowing relationship to track
- Rust's reference counting handles cleanup

## Next Steps for Full Multithreading

To implement lazy SMP:

1. Add a spawn method to `ThreadPool`:
```rust
pub fn start_search(&mut self, board: &Board) {
    for worker in &mut self.workers {
        let board_clone = board.clone();
        let td_clone = Arc::clone(&worker.data);
        
        worker.handle = Some(std::thread::spawn(move || {
            search(&board_clone, &mut *td_clone);
        }));
    }
}
```

2. Add a stop method:
```rust
pub fn stop(&mut self) {
    self.shared.stop.store(true, Ordering::Relaxed);
    
    for worker in &mut self.workers {
        if let Some(handle) = worker.handle.take() {
            handle.join().ok();
        }
    }
}
```

3. In your search loop, check the stop flag:
```rust
if td.shared.stop.load(Ordering::Relaxed) {
    break;
}
```

