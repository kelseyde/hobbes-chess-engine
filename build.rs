use hobbes_nnue_arch::{preprocess, Network, UntransposedNetwork};
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::env;
use std::fs;
use std::mem::size_of;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Read the raw on-disk network file into an aligned buffer.
    let raw_net = fs::read("hobbes.nnue").expect("network file not found!");
    assert_eq!(
        raw_net.len(),
        size_of::<UntransposedNetwork>(),
        "hobbes.nnue is {} bytes but UntransposedNetwork is {} bytes — wrong network file?",
        raw_net.len(),
        size_of::<UntransposedNetwork>()
    );
    let src_layout = Layout::from_size_align(size_of::<UntransposedNetwork>(), 64).unwrap();
    let src: &UntransposedNetwork = unsafe {
        let ptr = alloc_zeroed(src_layout);
        assert!(!ptr.is_null(), "allocation failed");
        std::ptr::copy_nonoverlapping(raw_net.as_ptr(), ptr, raw_net.len());
        &*(ptr as *const UntransposedNetwork)
    };

    let layout = Layout::from_size_align(size_of::<Network>(), 64).unwrap();
    let dst: &mut Network = unsafe {
        let ptr = alloc_zeroed(layout);
        assert!(!ptr.is_null(), "allocation failed");
        &mut *(ptr as *mut Network)
    };

    preprocess::process_network(src, dst);

    let dst_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(dst as *const Network as *const u8, size_of::<Network>())
    };
    let network_path = out_dir.join("hobbes_converted.nnue");
    fs::write(&network_path, dst_bytes).unwrap();

    unsafe {
        dealloc(src as *const UntransposedNetwork as *mut u8, src_layout);
        dealloc(dst as *mut Network as *mut u8, layout);
    };

    println!("cargo:rustc-env=NETWORK_PATH={}", network_path.display());
    println!("cargo:rerun-if-changed=hobbes.nnue");
}
