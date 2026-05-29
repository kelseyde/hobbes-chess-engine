// Credit to Akimbo author for the implementation
#[macro_export]
macro_rules! tunable_params {

    ($($name:ident = $val:expr, $min:literal ..= $max:literal, $spsa:expr;)*) => {
        #[cfg(feature = "tuning")]
        use std::sync::atomic::Ordering;

        #[cfg(feature = "tuning")]
        pub fn list_params() {
            $(
                println!(
                    "option name {} type spin default {} min {} max {}",
                    stringify!($name),
                    $name(),
                    $min,
                    $max,
                );
            )*
        }

        #[cfg(feature = "tuning")]
        pub fn set_param(name: &str, val: i32) {
            match name {
                $(
                    stringify!($name) => vals::$name.store(val, Ordering::Relaxed),
                )*
                _ => println!("info error unknown option"),
            }
        }

        #[cfg(feature = "tuning")]
        pub fn print_params_ob() {
            $(
                if $spsa {
                    let step = ($max - $min) / 20;
                    println!(
                        "{}, int, {}.0, {}.0, {}.0, {}, 0.002",
                        stringify!($name),
                        $name(),
                        $min,
                        $max,
                        step,
                    );
                }
            )*
        }

        #[cfg(feature = "tuning")]
        mod vals {
            use std::sync::atomic::AtomicI32;
            $(
            #[allow(non_upper_case_globals)]
            pub static $name: AtomicI32 = AtomicI32::new($val);
            )*
        }

        $(
        #[cfg(feature = "tuning")]
        #[inline]
        pub fn $name() -> i32 {
            vals::$name.load(Ordering::Relaxed)
        }

        #[cfg(not(feature = "tuning"))]
        #[inline]
        pub fn $name() -> i32 {
            $val
        }
        )*
    };

}

/// Gravity formula for history updates, using the current value of the entry as the base for the update.
pub fn gravity(current: i32, update: i32, max: i32) -> i32 {
    gravity_with_base(current, update, current, max)
}

/// Gravity formula for history updates, weighting the update by the current value of the entry.
pub fn gravity_with_base(current: i32, update: i32, base: i32, max: i32) -> i32 {
    current + update - base * update.abs() / max
}

/// Linearly interpolate between two scores, using the provided factor (0-100).
#[inline]
pub const fn lerp(a: i32, b: i32, factor: i32) -> i32 {
    (a * (100 - factor) + b * factor) / 100
}

// Slide in one direction, accumulating set bits until a blocker is hit.
// `delta` is the square-index step; `stop` returns true when the edge is reached.
pub fn slide(square: usize, blockers: u64, delta: i32, stop: fn(usize) -> bool) -> u64 {
    let mut bb: u64 = 0;
    let mut tgt = square;
    while !stop(tgt) {
        tgt = (tgt as i32 + delta) as usize;
        bb |= 1 << tgt;
        if blockers & (1 << tgt) != 0 {
            break;
        }
    }
    bb
}

pub fn murmur_hash3(key: u64) -> u64 {
    let mut k = key;
    k ^= k >> 33;
    k = k.wrapping_mul(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k = k.wrapping_mul(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    k
}

// Credit to Akimbo author - necessary for boxing large arrays
// without exploding the stack on initialisation.
/// # Safety
///
/// The caller must ensure that `T` is valid for zero-initialization.
/// This function allocates memory for `T` and zeroes it, then returns a `Box<T>`.
/// If `T` contains any non-zeroable types, using this function may cause undefined behavior.
pub unsafe fn boxed_and_zeroed<T>() -> Box<T> {
    let layout = std::alloc::Layout::new::<T>();
    let ptr = std::alloc::alloc_zeroed(layout);
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    Box::from_raw(ptr.cast())
}
