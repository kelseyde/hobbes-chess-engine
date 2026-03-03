// Credit to Akimbo author for the implementation
#[macro_export]
macro_rules! tunable_params {

    ($($name:ident = $min:expr, $max:expr, $step:expr, $spsa:expr, $vals:expr;)*) => {
        #[cfg(feature = "tuning")]
        use std::sync::atomic::Ordering;

        pub const NUM_KING_BUCKETS: usize = 64;

        #[cfg(feature = "tuning")]
        pub fn list_params() {
            $(
                for k in 0..NUM_KING_BUCKETS {
                    println!(
                        "option name {}_k{} type spin default {} min {} max {}",
                        stringify!($name),
                        k,
                        $name(k),
                        $min,
                        $max,
                    );
                }
            )*
        }

        #[cfg(feature = "tuning")]
        pub fn set_param(name: &str, val: i32) {
            if let Some(pos) = name.rfind("_k") {
                let param_name = &name[..pos];
                if let Ok(bucket) = name[pos + 2..].parse::<usize>() {
                    if bucket < NUM_KING_BUCKETS {
                        match param_name {
                            $(
                                stringify!($name) => vals::$name[bucket].store(val, Ordering::Relaxed),
                            )*
                            _ => println!("info error unknown option"),
                        }
                        return;
                    }
                }
            }
            println!("info error unknown option");
        }

        #[cfg(feature = "tuning")]
        pub fn print_params_ob() {
            $(
                if $spsa {
                    let step = ($max - $min) / 20;
                    for k in 0..NUM_KING_BUCKETS {
                        println!(
                            "{}_k{}, int, {}.0, {}.0, {}.0, {}, 0.002",
                            stringify!($name),
                            k,
                            $name(k),
                            $min,
                            $max,
                            step,
                        );
                    }
                }
            )*
        }

        #[cfg(feature = "tuning")]
        mod vals {
            use std::sync::atomic::AtomicI32;
            $(
            #[allow(non_upper_case_globals)]
            pub static $name: [AtomicI32; super::NUM_KING_BUCKETS] = {
                const BASE: [i32; super::NUM_KING_BUCKETS] = $vals;
                const fn make_atomics() -> [AtomicI32; super::NUM_KING_BUCKETS] {
                    let mut arr: [std::mem::MaybeUninit<AtomicI32>; super::NUM_KING_BUCKETS] =
                        unsafe { std::mem::MaybeUninit::uninit().assume_init() };
                    let mut i = 0;
                    while i < super::NUM_KING_BUCKETS {
                        arr[i] = std::mem::MaybeUninit::new(AtomicI32::new(BASE[i]));
                        i += 1;
                    }
                    unsafe { std::mem::transmute(arr) }
                }
                make_atomics()
            };
            )*
        }

        $(
        #[cfg(feature = "tuning")]
        #[inline]
        pub fn $name(king_sq: usize) -> i32 {
            vals::$name[king_sq].load(Ordering::Relaxed)
        }

        #[cfg(not(feature = "tuning"))]
        #[inline]
        pub fn $name(king_sq: usize) -> i32 {
            const VALS: [i32; NUM_KING_BUCKETS] = $vals;
            VALS[king_sq]
        }
        )*
    };

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
