extern crate wasm;

use std::env;
use std::fs::File;
use std::io::Read;


fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 2 {
        println!("Usage: {} somefile.wast", args[0]);
        return;
    }

    let mut contents = Vec::new();
    File::open(&args[1]).expect("readable file").read_to_end(&mut contents).expect("read succeeds");

    let test = wasm::TestCase::parse(&contents);

    test.run_all();
}
