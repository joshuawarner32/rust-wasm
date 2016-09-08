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
    import_table.insert(&b"env"[..], Box::new(EmscriptenEnvModule) as Box<wasm::BoundInstance>);
    import_table.insert(&b"asm2wasm"[..], Box::new(EmscriptenAsm2WasmModule) as Box<wasm::BoundInstance>);
    let mut inst = wasm::Instance::new(&module, import_table);

    let esp = module.find(b"establishStackSpace").unwrap();
    let main = module.find(b"_main").unwrap();

    inst.invoke(esp, &[Dynamic::from_u32(4*1024), Dynamic::from_u32(16*1024)]);

    let res = inst.invoke(main, &[Dynamic::from_u32(0), Dynamic::from_u32(0)]);

    // println!("{}", Pr(res));
}

struct EmscriptenEnvModule;

impl wasm::BoundInstance for EmscriptenEnvModule {
    fn invoke_export(&mut self, func: wasm::ExportIndex, args: &[Dynamic]) -> wasm::InterpResult {
        match func.0 {
            0 => {
                panic!("called abort");
            }
            1 => {
                panic!("called abortStackOverflow");
            }
            2 => {
                panic!("called nullFunc_ii");
            }
            3 => {
                panic!("called nullFunc_iiii");
            }
            4 => {
                panic!("called nullFunc_vi");
            }
            5 => {
                panic!("called _pthread_cleanup_pop");
            }
            6 => {
                panic!("called _abort");
            }
            7 => {
                panic!("called ___lock");
            }
            8 => {
                panic!("called ___syscall6");
            }
            9 => {
                panic!("called _pthread_cleanup_push");
            }
            10 => {
                panic!("called _sbrk");
            }
            11 => {
                panic!("called ___syscall140");
            }
            12 => {
                panic!("called _emscripten_memcpy_big");
            }
            13 => {
                panic!("called ___syscall54");
            }
            14 => {
                panic!("called ___unlock");
            }
            15 => {
                panic!("called ___syscall146");
            }
            _ => panic!()
        }
    }
    fn export_by_name_and_type(&self, name: &[u8], ty: wasm::FunctionType<&[u8]>) -> wasm::ExportIndex {
        println!("looking for env {}", str::from_utf8(name).unwrap_or("<bad_utf8>"));
        wasm::ExportIndex(match name {
            b"abort" => 0,
            b"abortStackOverflow" => 1,
            b"nullFunc_ii" => 2,
            b"nullFunc_iiii" => 3,
            b"nullFunc_vi" => 4,
            b"_pthread_cleanup_pop" => 5,
            b"_abort" => 6,
            b"___lock" => 7,
            b"___syscall6" => 8,
            b"_pthread_cleanup_push" => 9,
            b"_sbrk" => 10,
            b"___syscall140" => 11,
            b"_emscripten_memcpy_big" => 12,
            b"___syscall54" => 13,
            b"___unlock" => 14,
            b"___syscall146" => 15,
            _ => panic!()
        })
    }
}

struct EmscriptenAsm2WasmModule;

impl wasm::BoundInstance for EmscriptenAsm2WasmModule {
    fn invoke_export(&mut self, func: wasm::ExportIndex, args: &[Dynamic]) -> wasm::InterpResult {
        for a in args {
            match a {
                &Dynamic::Int32(v) => println!("print: {}", v),
                &Dynamic::Int64(v) => println!("print: {}", v),
                &Dynamic::Float32(v) => println!("print: {}", v),
                &Dynamic::Float64(v) => println!("print: {}", v),
            }
        }
        panic!();
    }
    fn export_by_name_and_type(&self, name: &[u8], ty: wasm::FunctionType<&[u8]>) -> wasm::ExportIndex {
        println!("looking for asm2wasm {}", str::from_utf8(name).unwrap_or("<bad_utf8>"));
        wasm::ExportIndex(0)
    }
}
