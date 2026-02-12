// Credit to Akimbo author for the implementation
#[macro_export]
macro_rules! tunable_params {

    ($($name:ident = $val:expr, $min:expr, $max:expr, $step:expr, $spsa:expr;)*) => {
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
