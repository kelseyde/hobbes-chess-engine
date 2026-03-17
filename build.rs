use std::env;
use std::fs;
use std::path::PathBuf;
use hobbes_nnue_arch::{Network, permute};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    // Read the unpermuted network file.
    let mut raw_net = fs::read("hobbes.nnue").expect("network file not found!");
    let network: &mut Network = unsafe { &mut *(raw_net.as_mut_ptr() as *mut Network) };
    
    // Permute the network.
    permute::permute_network(network);

    // Write the permuted network file.
    let permuted_path = out_dir.join("hobbes_permuted.nnue");
    fs::write(&permuted_path, &raw_net).unwrap();

    // Export the path so the main code can include_bytes! it
    println!("cargo:rustc-env=NETWORK_PATH={}", permuted_path.display());
    println!("cargo:rerun-if-changed=hobbes.nnue");
}


