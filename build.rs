use std::env;
use std::fs;
use std::path::PathBuf;
use hobbes_nnue_arch::{Network, UntransposedNetwork, preprocess};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Read the raw on-disk network file as UntransposedNetwork.
    let raw_net = fs::read("hobbes.nnue").expect("network file not found!");
    let src: &UntransposedNetwork = unsafe { &*(raw_net.as_ptr() as *const UntransposedNetwork) };

    // Allocate the runtime Network on the heap and convert (permute L0 + transpose L1).
    let mut dst_bytes = vec![0u8; size_of::<Network>()];
    let dst: &mut Network = unsafe { &mut *(dst_bytes.as_mut_ptr() as *mut Network) };
    preprocess::process_network(src, dst);

    // Write the converted network file.
    let network_path = out_dir.join("hobbes_converted.nnue");
    fs::write(&network_path, &dst_bytes).unwrap();

    // Export the path so the main code can include_bytes! it
    println!("cargo:rustc-env=NETWORK_PATH={}", network_path.display());
    println!("cargo:rerun-if-changed=hobbes.nnue");
}


