use std::env;
use std::fs;
use std::path::PathBuf;
use hobbes_nnue_arch::L1_SIZE;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Arch constants are available via nnue_arch, e.g.:
    let _l1_size = L1_SIZE;
    let _l2_size = hobbes_nnue_arch::L2_SIZE;

    // Read the un-permuted network file
    let raw_net = fs::read("hobbes.nnue").expect("hobbes.nnue not found");

    // TODO: permute the network here based on target SIMD variant
    // All arch constants (L1_SIZE, L2_SIZE, etc.) are available from nnue_arch

    // Write the (for now, identical) permuted net to OUT_DIR
    let permuted_path = out_dir.join("hobbes_permuted.nnue");
    fs::write(&permuted_path, &raw_net).unwrap();

    // Export the path so the main code can include_bytes! it
    println!("cargo:rustc-env=PERMUTED_NET_PATH={}", permuted_path.display());
    println!("cargo:rerun-if-changed=hobbes.nnue");
}


