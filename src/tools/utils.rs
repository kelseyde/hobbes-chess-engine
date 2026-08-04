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

#[macro_export]
macro_rules! tunable_arrays {
    ($($name:ident = [$($val:expr),* $(,)?], $min:literal ..= $max:literal;)*) => {
        pub mod arrays {
            $(
                pub mod $name {
                    pub const DEFAULTS: &[i32] = &[$($val),*];
                    pub const LEN: usize = DEFAULTS.len();

                    #[cfg(feature = "tuning")]
                    pub static VALS: [std::sync::atomic::AtomicI32; LEN] =
                        [$(std::sync::atomic::AtomicI32::new($val)),*];
                }
            )*
        }

        $(
            #[cfg(feature = "tuning")]
            #[inline]
            pub fn $name(i: usize) -> i32 {
                arrays::$name::VALS[i].load(std::sync::atomic::Ordering::Relaxed)
            }

            #[cfg(not(feature = "tuning"))]
            #[inline]
            pub const fn $name(i: usize) -> i32 {
                arrays::$name::DEFAULTS[i]
            }
        )*

        #[cfg(feature = "tuning")]
        pub fn list_array_params() {
            $(
                for i in 0..arrays::$name::LEN {
                    println!(
                        "option name {}_{} type spin default {} min {} max {}",
                        stringify!($name), i, $name(i), $min, $max
                    );
                }
            )*
        }

        #[cfg(feature = "tuning")]
        pub fn set_array_param(name: &str, val: i32) -> bool {
            $(
                if let Some(rest) = name.strip_prefix(stringify!($name)) {
                    if let Some(rest) = rest.strip_prefix('_') {
                        if let Ok(i) = rest.parse::<usize>() {
                            if i < arrays::$name::LEN {
                                arrays::$name::VALS[i]
                                    .store(val, std::sync::atomic::Ordering::Relaxed);
                                return true;
                            }
                        }
                    }
                }
            )*
            false
        }

        #[cfg(feature = "tuning")]
        pub fn print_array_params_ob() {
            $(
                let step = ($max - $min) / 20;
                for i in 0..arrays::$name::LEN {
                    println!(
                        "{}_{}, int, {}.0, {}.0, {}.0, {}, 0.002",
                        stringify!($name), i, $name(i), $min, $max, step
                    );
                }
            )*
        }
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

/// Sum every 1, 2, and 3-way interaction between `N` active boolean features.
pub fn convolve<const N: usize>(
    features: [bool; N],
    factor_1: impl Fn(usize) -> i32,
    factor_2: impl Fn(usize) -> i32,
    factor_3: impl Fn(usize) -> i32,
) -> i32 {
    let mut total = 0;
    let mut index_2 = 0;
    let mut index_3 = 0;
    for i in 0..N {
        total += factor_1(i) * features[i] as i32;
        for j in (i + 1)..N {
            total += factor_2(index_2) * (features[i] && features[j]) as i32;
            for k in (j + 1)..N {
                total += factor_3(index_3) * (features[i] && features[j] && features[k]) as i32;
                index_3 += 1;
            }
            index_2 += 1;
        }
    }
    total
}