// Credit to Akimbo author for the implementation
#[macro_export]
macro_rules! tunable_params {
    ($($name:ident = $mg_val:expr, $eg_val:expr, $min:expr, $max:expr, $step:expr;)*) => {
        #[cfg(feature = "tuning")]
        use std::sync::atomic::Ordering;

        #[cfg(feature = "tuning")]
        pub fn list_params() {
            $(
                println!(
                    "option name {}_mg type spin default {} min {} max {}",
                    stringify!($name),
                    $mg_val,
                    $min,
                    $max,
                );
                println!(
                    "option name {}_eg type spin default {} min {} max {}",
                    stringify!($name),
                    $eg_val,
                    $min,
                    $max,
                );
            )*
        }

        #[cfg(feature = "tuning")]
        pub fn set_param(name: &str, val: i32) {
            match name {
                $(
                    concat!(stringify!($name), "_mg") => vals::$name._mg.store(val, Ordering::Relaxed),
                    concat!(stringify!($name), "_eg") => vals::$name._eg.store(val, Ordering::Relaxed),
                )*
                _ => println!("info error unknown option"),
            }
        }

        #[cfg(feature = "tuning")]
        pub fn print_params_ob() {
            $(
                println!(
                    "{}_mg, int, {}.0, {}.0, {}.0, {}, 0.002",
                    stringify!($name),
                    vals::$name._mg.load(Ordering::Relaxed),
                    $min,
                    $max,
                    $step,
                );
                println!(
                    "{}_eg, int, {}.0, {}.0, {}.0, {}, 0.002",
                    stringify!($name),
                    vals::$name._eg.load(Ordering::Relaxed),
                    $min,
                    $max,
                    $step,
                );
            )*
        }

        #[cfg(feature = "tuning")]
        mod vals {
            use std::sync::atomic::AtomicI32;
            $(
                #[allow(non_upper_case_globals)]
                pub struct $name {
                    pub _mg: AtomicI32,
                    pub _eg: AtomicI32,
                }

                #[allow(non_upper_case_globals)]
                pub static $name: $name = $name {
                    _mg: AtomicI32::new($mg_val),
                    _eg: AtomicI32::new($eg_val),
                };
            )*
        }

        $(
            #[cfg(feature = "tuning")]
            #[inline]
            pub fn $name() -> (i32, i32) {
                (
                    vals::$name._mg.load(Ordering::Relaxed),
                    vals::$name._eg.load(Ordering::Relaxed),
                )
            }

            #[cfg(not(feature = "tuning"))]
            #[inline]
            pub fn $name() -> (i32, i32) {
                ($mg_val, $eg_val)
            }

            // Helper function to get interpolated value based on game phase
            // phase should be a value from 0 (endgame) to some max value (middlegame)
            // max_phase should be the maximum phase value (typically 24 in chess)
            #[inline]
            pub fn [<$name _interpolated>](phase: i32, max_phase: i32) -> i32 {
                let (mg, eg) = $name();
                ((mg * phase + eg * (max_phase - phase)) + max_phase / 2) / max_phase
            }
        )*
    };
}
// Credit to Akimbo author - necessary for boxing large arrays
// without exploding the stack on initialisation.
pub unsafe fn boxed_and_zeroed<T>() -> Box<T> {
    let layout = std::alloc::Layout::new::<T>();
    let ptr = std::alloc::alloc_zeroed(layout);
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    Box::from_raw(ptr.cast())
}