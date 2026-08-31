use hobbes_nnue_arch::{preprocess, Network, UntransposedNetwork, L0_THREAT_PP_COMBINED_FEATURES, L1_SIZE};
use std::env;
use std::fs;
use std::mem::size_of;
use std::path::PathBuf;

const INPUT_NET_FILE: &str = "hobbes.nnue";
const OUTPUT_NET_FILE: &str = "hobbes_converted.nnue";

fn main() {
    // Load the raw network
    let raw_net: Vec<u8> = read_network_bytes(INPUT_NET_FILE);
    let mut src: Box<UntransposedNetwork> = load_network_from_bytes(&raw_net);
    unsafe {
        std::ptr::write_bytes(
            src.l0_threat_pp_weights.as_mut_ptr(),
            0,
            L0_THREAT_PP_COMBINED_FEATURES * L1_SIZE,
        );
    }
    let mut dst: Box<Network> = unsafe { boxed_and_zeroed() };

    // Transpose and permute the net
    preprocess::process_network(&src, &mut dst);

    // Write the processed network
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let network_path = out_dir.join(OUTPUT_NET_FILE);
    write_network_bytes(&network_path, &dst);

    println!("cargo:rustc-env=NETWORK_PATH={}", network_path.display());
    println!("cargo:rerun-if-changed=hobbes.nnue");
}

fn read_network_bytes(path: &str) -> Vec<u8> {
    fs::read(path).expect("network file not found!")
}

fn write_network_bytes(path: &PathBuf, dst: &Network) {
    fs::write(path, unsafe { as_bytes(dst) }).unwrap();
}

fn load_network_from_bytes(bytes: &[u8]) -> Box<UntransposedNetwork> {
    assert_eq!(
        bytes.len(),
        size_of::<UntransposedNetwork>(),
        "Byte slice is {} bytes but UntransposedNetwork is {} bytes — wrong network file?",
        bytes.len(),
        size_of::<UntransposedNetwork>()
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
