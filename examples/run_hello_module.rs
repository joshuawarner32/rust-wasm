extern crate wasm;

use std::{env, str};
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

    let mut import_table = HashMap::new();
    import_table.insert(&b"env"[..], Box::new(HelloEnvModule) as Box<wasm::BoundInstance>);
    let mut inst = wasm::Instance::new(&module, import_table);

    let main = module.find(b"main").unwrap();

    inst.invoke(main, &[]);
}

struct HelloEnvModule;

impl wasm::BoundInstance for HelloEnvModule {
    fn invoke_export(&mut self, memory: &mut wasm::Memory, func: wasm::ExportIndex, args: &[Dynamic]) -> wasm::InterpResult {
        match func.0 {
            0 => {
                println!("{} {} {:?}", args[0].to_u32(), args[1].to_u32(), memory.get_bytes(args[0].to_u32()..(args[1].to_u32() + args[0].to_u32())));
                let bytes = memory.get_bytes(args[0].to_u32()..(args[1].to_u32() + args[0].to_u32()));
                println!("{}", str::from_utf8(bytes).unwrap());
            }
            _ => panic!()
        }
        wasm::InterpResult::Value(None)
    }
    fn export_by_name_and_type(&self, name: &[u8], ty: wasm::FunctionType<&[u8]>) -> wasm::ExportIndex {
        println!("looking for env {}", str::from_utf8(name).unwrap_or("<bad_utf8>"));
        wasm::ExportIndex(match name {
            b"puts" => 0,
            _ => panic!()
        })
    }
}
