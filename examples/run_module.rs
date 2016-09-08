extern crate wasm;

use std::env;
use std::fs::File;
use std::io::Read;
use std::collections::HashMap;

use wasm::Dynamic;

fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 2 {
        println!("Usage: {} somefile.wasm", args[0]);
        return;
    }

    let mut contents = Vec::new();
    File::open(&args[1]).expect("readable file").read_to_end(&mut contents).expect("read succeeds");

    let module = wasm::Module::parse(&contents);

    let mut inst = wasm::Instance::new(&module, HashMap::new());

    let esp = module.find(b"establishStackSpace").unwrap();
    let main = module.find(b"_main").unwrap();

    inst.invoke(esp, &[Dynamic::from_u32(4*1024), Dynamic::from_u32(16*1024)]);

    let res = inst.invoke(main, &[Dynamic::from_u32(0), Dynamic::from_u32(0)]);

    // println!("{}", Pr(res));
}
