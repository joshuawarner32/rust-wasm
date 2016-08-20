use std::fs::{self, File};
use std::io::{Read, Write};
use std::{str, mem, fmt};
use std::cmp::min;
use std::collections::HashMap;
use std::path::Path;
use std::ops::Deref;
use std::num::Wrapping;

use types::{Type, Pr};
use reader::Reader;

struct Chunk<'a> {
    name: &'a str,
    data: &'a [u8]
}


fn read_chunk<'a>(reader: &mut Reader<'a>) -> Chunk<'a> {
    Chunk {
        name: reader.read_str(),
        data: reader.read_bytes()
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct FunctionIndex(pub usize);

#[derive(Copy, Clone)]
pub struct FunctionType<'a> {
    param_types: &'a[Type],
    return_type: Option<Type>
}

impl<'a> fmt::Display for FunctionType<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "(");
        let mut first = true;
        for p in self.param_types {
            write!(f, "{}{}", if first { first = false; ""} else {", "}, p);
        }
        write!(f, ") -> {}", Pr(self.return_type))
    }
}

pub struct Import<'a> {
    pub function_type: FunctionType<'a>,
    pub module_name: &'a str,
    pub function_name: &'a str,
}

pub struct MemoryInfo {
    pub initial_64k_pages: usize,
    pub maximum_64k_pages: usize,
    pub is_exported: bool
}

pub struct Export<'a> {
    pub function_index: FunctionIndex,
    pub function_name: &'a str
}

pub struct FunctionBody<'a> {
    pub locals: Vec<(Type, usize)>,
    pub ast: &'a [u8]
}

pub struct MemoryChunk<'a> {
    pub offset: usize,
    pub data: &'a [u8]
}

pub struct Names<'a> {
    pub function_name: &'a str,
    pub local_names: Vec<&'a str>
}

pub struct Module<'a> {
    types: Vec<FunctionType<'a>>,
    pub imports: Vec<Import<'a>>,
    pub functions: Vec<FunctionType<'a>>,
    pub table: Vec<FunctionIndex>,
    pub memory_info: MemoryInfo,
    pub start_function_index: Option<FunctionIndex>,
    pub exports: Vec<Export<'a>>,
    pub code: Vec<FunctionBody<'a>>,
    pub memory_chunks: Vec<MemoryChunk<'a>>,
    pub names: Vec<Names<'a>>
}

fn as_type_slice(bytes: &[u8]) -> &[Type] {
    for b in bytes {
        if *b > 4 || *b == 0 {
            panic!();
        }
    }

    unsafe { mem::transmute(bytes) }
}

fn singular<T: Copy>(ts: &[T]) -> Option<T> {
    match ts.len() {
        0 => None,
        1 => Some(ts[0]),
        _ => panic!()
    }
}

impl<'a> Module<'a> {
    fn new() -> Module<'a> {
        Module {
            types: Vec::new(),
            imports: Vec::new(),
            functions: Vec::new(),
            table: Vec::new(),
            memory_info: MemoryInfo {
                initial_64k_pages: 1,
                maximum_64k_pages: 1,
                is_exported: true
            },
            start_function_index: None,
            exports: Vec::new(),
            code: Vec::new(),
            memory_chunks: Vec::new(),
            names: Vec::new(),
        }
    }

    pub fn parse(data: &'a [u8]) -> Module<'a> {
        let mut types = None;
        let mut imports = None;
        let mut functions = None;
        let mut table = None;
        let mut memory_info = None;
        let mut start_function_index = None;
        let mut exports = None;
        let mut code = None;
        let mut memory_chunks = None;
        let mut names = None;

        let mut r = Reader::new(data);

        assert!(r.read_u32() == 0x6d736100);
        assert!(r.read_u32() == 11);

        while !r.at_eof() {
            let c = read_chunk(&mut r);
            let mut r = Reader::new(c.data);
            match c.name {
                "type" => {
                    if types.is_some() {
                        panic!("duplicate type chunk!");
                    }

                    let count = r.read_var_u32() as usize;
                    let mut tys = Vec::with_capacity(count);

                    for _ in 0..count {
                        let form = r.read_var_u32();
                        if form != 0x40 {
                            panic!("unknown type form: {}", form);
                        }

                        tys.push(FunctionType {
                            param_types: as_type_slice(r.read_bytes()),
                            return_type: singular(as_type_slice(r.read_bytes()))
                        });
                    }

                    types = Some(tys);
                }
                "import" => {
                    if imports.is_some() {
                        panic!("duplicate import chunk!");
                    }

                    if let Some(ref tys) = types {
                        let count = r.read_var_u32() as usize;
                        let mut ims = Vec::with_capacity(count);

                        for _ in 0..count {
                            ims.push(Import {
                                function_type: tys[r.read_var_u32() as usize],
                                module_name: r.read_str(),
                                function_name: r.read_str()
                            });
                        }
                        imports = Some(ims);
                    } else {
                        panic!("need type chunk to decode imports!");
                    }
                }
                "function" => {
                    if functions.is_some() {
                        panic!("duplicate function chunk!");
                    }

                    if let Some(ref tys) = types {
                        let count = r.read_var_u32() as usize;
                        let mut fns = Vec::with_capacity(count);

                        for _ in 0..count {
                            fns.push(tys[r.read_var_u32() as usize]);
                        }
                        functions = Some(fns);
                    } else {
                        panic!("need type chunk to decode functions!");
                    }
                }
                "table" => {
                    if table.is_some() {
                        panic!("duplicate table chunk!");
                    }

                    if let Some(ref fns) = functions {
                        let count = r.read_var_u32() as usize;
                        let mut tbl = Vec::with_capacity(count);

                        for _ in 0..count {
                            let index = r.read_var_u32() as usize;
                            if index >= fns.len() {
                                panic!();
                            }
                            tbl.push(FunctionIndex(index));
                        }
                        table = Some(tbl);
                    } else {
                        panic!("need functions chunk to decode table!");
                    }
                }
                "memory" => {
                    if memory_info.is_some() {
                        panic!("duplicate memory chunk!");
                    }
                    memory_info = Some(MemoryInfo {
                        initial_64k_pages: r.read_var_u32() as usize,
                        maximum_64k_pages: r.read_var_u32() as usize,
                        is_exported: r.read_u8() == 1,
                    });
                }
                "export" => {
                    if exports.is_some() {
                        panic!("duplicate export chunk!");
                    }

                    if let Some(ref fns) = functions {
                        let count = r.read_var_u32() as usize;
                        let mut exp = Vec::with_capacity(count);

                        for _ in 0..count {
                            let ind = r.read_var_u32() as usize;
                            if ind >= fns.len() {
                                panic!();
                            }
                            exp.push(Export {
                                function_index: FunctionIndex(ind),
                                function_name: r.read_str(),
                            });
                        }
                        exports = Some(exp);
                    } else {
                        panic!("need functions chunk to decode exports!");
                    }
                }
                "start" => {
                    if start_function_index.is_some() {
                        panic!("duplicate start chunk!");
                    }

                    if let Some(ref fns) = functions {
                        let function_index = r.read_var_u32() as usize;
                        if function_index >= fns.len() {
                            panic!();
                        }
                        start_function_index = Some(FunctionIndex(function_index));
                    } else {
                        panic!("need functions chunk to decode start!");
                    }
                }
                "code" => {
                    if code.is_some() {
                        panic!("duplicate code chunk!");
                    }

                    if let Some(ref fns) = functions {
                        let count = r.read_var_u32() as usize;
                        if count != fns.len() {
                            panic!();
                        }
                        let mut cd = Vec::with_capacity(count);

                        for _ in 0..count {
                            let body = r.read_bytes();
                            let mut r = Reader::new(body);

                            let local_type_count = r.read_var_u32() as usize;
                            let mut locals = Vec::with_capacity(local_type_count);
                            for _ in 0..local_type_count {
                                let count_of_this_type = r.read_var_u32() as usize;
                                let ty = Type::from_u8(r.read_u8());
                                locals.push((ty, count_of_this_type)); 
                            }

                            let ast = r.into_remaining();

                            cd.push(FunctionBody {
                                locals: locals,
                                ast: ast
                            });
                        }
                        code = Some(cd);
                    } else {
                        panic!("need functions chunk to decode code!");
                    }
                }
                "data" => {
                    if memory_chunks.is_some() {
                        panic!("duplicate data chunk!");
                    }

                    let count = r.read_var_u32() as usize;
                    let mut mc = Vec::with_capacity(count);

                    for _ in 0..count {
                        mc.push(MemoryChunk {
                            offset: r.read_var_u32() as usize,
                            data: r.read_bytes(),
                        });
                    }
                    memory_chunks = Some(mc);
                }
                "name" => {
                    if names.is_some() {
                        panic!("duplicate data chunk!");
                    }
                    let count = r.read_var_u32() as usize;
                    let mut nm = Vec::with_capacity(count);

                    for _ in 0..count {
                        let fn_name = r.read_str();
                        let local_count = r.read_var_i32() as usize;
                        let mut local_names = Vec::with_capacity(local_count);
                        
                        for _ in 0..local_count {
                            local_names.push(r.read_str());
                        }

                        nm.push(Names {
                            function_name: fn_name,
                            local_names: local_names
                        });
                    }
                    names = Some(nm);
                }
                _ => panic!()
            }
        }

        if let Some(types) = types {
            if let Some(imports) = imports {
                if let Some(functions) = functions {
                    if let Some(table) = table {
                        if let Some(memory_info) = memory_info {
                            if let Some(exports) = exports {
                                if let Some(code) = code {
                                    return Module {
                                        types: types,
                                        imports: imports,
                                        functions: functions,
                                        table: table,
                                        memory_info: memory_info,
                                        start_function_index: start_function_index,
                                        exports: exports,
                                        code: code,
                                        memory_chunks: memory_chunks.unwrap_or(Vec::new()),
                                        names: names.unwrap_or(Vec::new())
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        panic!("missing critical chunk!");
    }

    fn find(&self, name: &str) -> Option<FunctionIndex> {
        for e in &self.exports {
            if e.function_name == name {
                return Some(e.function_index);
            }
        }
        None
    }

    fn find_name(&self, index: FunctionIndex) -> Option<&'a str> {
        for e in &self.exports {
            if e.function_index == index {
                return Some(e.function_name);
            }
        }
        if self.names.len() > index.0 {
            return Some(self.names[index.0].function_name);
        }
        None
    }
}