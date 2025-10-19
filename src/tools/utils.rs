// Credit to Akimbo author for the implementation
#[macro_export]
macro_rules! tunable_params {

    ($($name:ident = ($val1:expr, $val2:expr, $val3:expr), $min:expr, $max:expr, $step:expr;)*) => {
        #[cfg(feature = "tuning")]
        use std::sync::atomic::Ordering;
        use crate::board::phase::Phase;

        #[cfg(feature = "tuning")]
        pub fn list_params() {
            $(
                println!("option name {}_p1 type spin default {} min {} max {}", stringify!($name), $name(Phase::P1), $min, $max);
                println!("option name {}_p2 type spin default {} min {} max {}", stringify!($name), $name(Phase::P2), $min, $max);
                println!("option name {}_p3 type spin default {} min {} max {}", stringify!($name), $name(Phase::P3), $min, $max);
            )*
        }

        #[cfg(feature = "tuning")]
        pub fn set_param(name: &str, val: i32) {
            match name {
                $(
                    concat!(stringify!($name), "_p1") => vals::$name[0].store(val, Ordering::Relaxed),
                    concat!(stringify!($name), "_p2") => vals::$name[1].store(val, Ordering::Relaxed),
                    concat!(stringify!($name), "_p3") => vals::$name[2].store(val, Ordering::Relaxed),
                )*
                _ => println!("info error unknown option"),
            }
        }

        #[cfg(feature = "tuning")]
        pub fn print_params_ob() {
            $(
                println!("{}", format!("{}_p1, int, {}.0, {}.0, {}.0, {}, 0.002", stringify!($name), $name(Phase::P1), $min, $max, $step));
                println!("{}", format!("{}_p2, int, {}.0, {}.0, {}.0, {}, 0.002", stringify!($name), $name(Phase::P2), $min, $max, $step));
                println!("{}", format!("{}_p3, int, {}.0, {}.0, {}.0, {}, 0.002", stringify!($name), $name(Phase::P3), $min, $max, $step));
            )*
        }

        #[cfg(feature = "tuning")]
        mod vals {
            use std::sync::atomic::AtomicI32;
            $(
            #[allow(non_upper_case_globals)]
            pub static $name: [AtomicI32; 3] = [
                AtomicI32::new($val1),
                AtomicI32::new($val2),
                AtomicI32::new($val3),
            ];
            )*
        }

        $(
        #[cfg(feature = "tuning")]
        #[inline]
        pub fn $name(phase: Phase) -> i32 {
            vals::$name[phase as usize].load(Ordering::Relaxed)
        }

        #[cfg(not(feature = "tuning"))]
        #[inline]
        pub fn $name(phase: Phase) -> i32 {
            match phase {
                Phase::P1 => $val1,
                Phase::P2 => $val2,
                Phase::P3 => $val3,
            }
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

