extern crate wasm;

use std::env;
use std::fs::File;
use std::io::Read;

fn to_hex_string(bytes: &[u8]) -> String {
    bytes.iter()
        .map(|b| format!("{:02X}", b))
        .collect::<Vec<_>>().join(" ")
}

fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 2 {
        println!("Usage: {} somefile.wasm", args[0]);
        return;
    }

    let mut contents = Vec::new();
    File::open(&args[1]).expect("readable file").read_to_end(&mut contents).expect("read succeeds");

    let m = wasm::Module::parse(&contents);

    println!("imports:");
    for i in m.imports {
        println!("  {}.{}{}", i.module_name, i.function_name, i.function_type);
    }

    println!("functions:");
    for (i, f) in m.functions.iter().enumerate() {
        let name = m.names.get(i).map(|e| e.function_name).unwrap_or("<unnamed>");
        println!("  {}{}", name, f);

        let code = &m.code[i];

        for l in &code.locals {
            println!("    local {}[{}]", l.0, l.1);
        }

        for l in code.linear_ops() {
            println!("    {}", l);
        }
    }

    println!("exports:");
    for e in m.exports {
        let name = m.names.get(e.function_index.0).map(|e| e.function_name).unwrap_or("<unnamed>");
        let ty = m.functions[e.function_index.0];
        println!("  {} = {}{}", e.function_name, name, ty);
    }

    println!("dynamic function table:");
    for (i, t) in m.table.iter().enumerate() {
        let name = m.names.get(t.0).map(|e| e.function_name).unwrap_or("<unnamed>");
        let ty = m.functions[t.0];
        println!("  {} = {}{}", i, name, ty);
    }

    println!("memory info:");
    println!("  initial_64k_pages: {}", m.memory_info.initial_64k_pages);
    println!("  maximum_64k_pages: {}", m.memory_info.maximum_64k_pages);
    println!("  is_exported: {}", m.memory_info.is_exported);

    println!("start function:");
    if let Some(i) = m.start_function_index {
        let name = m.names.get(i.0).map(|e| e.function_name).unwrap_or("<unnamed>");
        let ty = m.functions[i.0];
        println!("  {}{}", name, ty);
    } else {
        println!("  (None)");
    }

    println!("initial memory:");
    for ch in &m.memory_chunks {
        println!("  from {} to {}:", ch.offset, ch.offset + ch.data.len());
        for line in ch.data.chunks(32) {
            println!("    {}", to_hex_string(line));
        }
    }
    if m.memory_chunks.len() == 0 {
        println!("  (None)");
    }
}
