use hobbes_nnue_arch::{preprocess, Network, UntransposedNetwork};
use std::env;
use std::fs;
use std::mem::size_of;
use std::path::PathBuf;

const INPUT_NET_FILE: &str = "hobbes.nnue";
const OUTPUT_NET_FILE: &str = "hobbes_converted.nnue";

fn main() {
    // Load the raw network
    let raw_net: Vec<u8> = read_network_bytes(INPUT_NET_FILE);
    let src: Box<UntransposedNetwork> = load_network_from_bytes(&raw_net);
    let mut dst: Box<Network> = unsafe { boxed_and_zeroed() };

    // Transpose and permute the net
    preprocess::process_network(&src, &mut dst);

    // Write the processed network
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let network_path = out_dir.join(OUTPUT_NET_FILE);
    write_network_bytes(&network_path, &dst);

    println!("cargo:rustc-env=NETWORK_PATH={}", network_path.display());
    println!("cargo:rerun-if-changed={}", INPUT_NET_FILE);
}

fn read_network_bytes(path: &str) -> Vec<u8> {
    fs::read(path).expect("network file not found!")
}

fn write_network_bytes(path: &PathBuf, dst: &Network) {
    fs::write(path, unsafe { as_bytes(dst) }).unwrap();
}

/// Load the `UntransposedNetwork` from the raw bytes, verifying the size matches the expected size
/// of the unprocessed net.
fn load_network_from_bytes(bytes: &[u8]) -> Box<UntransposedNetwork> {
    let target_size = size_of::<UntransposedNetwork>();
    let actual_size = bytes.len();
    assert_eq!(
        actual_size,
        target_size,
        "Network size {} does not match the desired {}",
        actual_size,
        target_size
    );
    unsafe {
        let mut b: Box<UntransposedNetwork> = boxed_and_zeroed();
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.as_mut() as *mut _ as *mut u8, bytes.len());
        b
    }
}

unsafe fn as_bytes<T>(val: &T) -> &[u8] {
    std::slice::from_raw_parts(val as *const T as *const u8, size_of::<T>())
}

unsafe fn boxed_and_zeroed<T>() -> Box<T> {
    let layout = std::alloc::Layout::new::<T>();
    let ptr = std::alloc::alloc_zeroed(layout);
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    Box::from_raw(ptr.cast())
}
