use std::{str, mem, fmt};

use types::{Type, Pr};
use reader::Reader;
use ops::{NormalOp, LinearOpReader, BlockOpReader};

struct Chunk<'a> {
    name: &'a [u8],
    data: &'a [u8]
}

pub trait AsBytes {
    fn as_bytes(&self) -> &[u8];
}

impl<'a> AsBytes for &'a [u8] {
    fn as_bytes(&self) -> &[u8] {
        *self
    }
}

impl AsBytes for Vec<u8> {
    fn as_bytes(&self) -> &[u8] {
        self.as_slice()
    }
}

fn read_chunk<'a>(reader: &mut Reader<'a>) -> Chunk<'a> {
    Chunk {
        name: reader.read_bytes(),
        data: reader.read_bytes()
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct FunctionIndex(pub usize);

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct TableIndex(pub usize);

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct ImportIndex(pub usize);

#[derive(Copy, Clone)]
pub struct FunctionType<B: AsBytes> {
    pub param_types: B,
    pub return_type: Option<Type>
}

impl<B: AsBytes> fmt::Display for FunctionType<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "(");
        let mut first = true;
        for p in as_type_slice(self.param_types.as_bytes()) {
            write!(f, "{}{}", if first { first = false; ""} else {", "}, p);
        }
        write!(f, ") -> {}", Pr(self.return_type))
    }
}

#[test]
fn test_fn_ty_display() {
    assert_eq!("(i32, i64, f32, f64) -> void", format!("{}", FunctionType {
        param_types: vec!(1, 2, 3, 4),
        return_type: None
    }));
}

pub struct Import<B: AsBytes> {
    pub function_type: FunctionType<B>,
    pub module_name: B,
    pub function_name: B,
}

pub struct MemoryInfo {
    pub initial_64k_pages: usize,
    pub maximum_64k_pages: usize,
    pub is_exported: bool
}

pub struct Export<B: AsBytes> {
    pub function_index: FunctionIndex,
    pub function_name: B
}

pub struct FunctionBody<B: AsBytes> {
    pub locals: Vec<(Type, usize)>,
    pub ast: B
}

impl<B: AsBytes> FunctionBody<B> {
    pub fn linear_ops(&self) -> LinearOpReader {
        LinearOpReader::new(self.ast.as_bytes())
    }
    pub fn block_ops(&self) -> BlockOpReader {
        BlockOpReader::new(self.ast.as_bytes())
    }
}

pub struct MemoryChunk<B: AsBytes> {
    pub offset: usize,
    pub data: B,
}

pub struct Names<B: AsBytes> {
    pub function_name: B,
    pub local_names: Vec<B>,
}

pub struct Module<B: AsBytes> {
    types: Vec<FunctionType<B>>,
    pub imports: Vec<Import<B>>,
    pub functions: Vec<FunctionType<B>>,
    pub table: Vec<FunctionIndex>,
    pub memory_info: MemoryInfo,
    pub start_function_index: Option<FunctionIndex>,
    pub exports: Vec<Export<B>>,
    pub code: Vec<FunctionBody<B>>,
    pub memory_chunks: Vec<MemoryChunk<B>>,
    pub names: Vec<Names<B>>
}

pub struct FunctionBuilder {
    pub ty: FunctionType<Vec<u8>>,
    pub ops: Vec<NormalOp<'static>>,
    pub local_types: Vec<Type>,
}

impl FunctionBuilder {
    pub fn new() -> FunctionBuilder {
        FunctionBuilder {
            ty: FunctionType {
                param_types: Vec::new(),
                return_type: None,
            },
            ops: Vec::new(),
            local_types: Vec::new(),
        }
    }
    pub fn build(self) -> FunctionBody<Vec<u8>> {
        let mut locals = Vec::new();
        let mut last = (Type::Int32, 0);

        for ty in self.local_types {
            if last.0 == ty {
                last.1 += 1;
            } else {
                locals.push(last);
                last = (ty, 1);
            }
        }

        let mut ast = Vec::new();

        FunctionBody {
            locals: locals,
            ast: ast,
        }
    }
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

impl<B: AsBytes> Module<B> {
    pub fn new() -> Module<B> {
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

    pub fn find(&self, name: &[u8]) -> Option<FunctionIndex> {
        for e in &self.exports {
            if e.function_name.as_bytes() == name {
                return Some(e.function_index);
            }
        }
        None
    }

    pub fn find_name(&self, index: FunctionIndex) -> Option<&[u8]> {
        if self.names.len() > index.0 {
            return Some(self.names[index.0].function_name.as_bytes());
        }
        for e in &self.exports {
            if e.function_index == index {
                return Some(e.function_name.as_bytes());
            }
        }
        None
    }
}

impl<'a> Module<&'a [u8]> {
    pub fn parse(data: &'a [u8]) -> Module<&'a [u8]> {
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
                b"type" => {
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
                            param_types: r.read_bytes(),
                            return_type: singular(as_type_slice(r.read_bytes()))
                        });
                    }

                    types = Some(tys);
                }
                b"import" => {
                    if imports.is_some() {
                        panic!("duplicate import chunk!");
                    }

                    if let Some(ref tys) = types {
                        let count = r.read_var_u32() as usize;
                        let mut ims = Vec::with_capacity(count);

                        for _ in 0..count {
                            ims.push(Import {
                                function_type: tys[r.read_var_u32() as usize],
                                module_name: r.read_bytes(),
                                function_name: r.read_bytes()
                            });
                        }
                        imports = Some(ims);
                    } else {
                        panic!("need type chunk to decode imports!");
                    }
                }
                b"function" => {
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
                b"table" => {
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
                b"memory" => {
                    if memory_info.is_some() {
                        panic!("duplicate memory chunk!");
                    }
                    memory_info = Some(MemoryInfo {
                        initial_64k_pages: r.read_var_u32() as usize,
                        maximum_64k_pages: r.read_var_u32() as usize,
                        is_exported: r.read_u8() == 1,
                    });
                }
                b"export" => {
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
                                function_name: r.read_bytes(),
                            });
                        }
                        exports = Some(exp);
                    } else {
                        panic!("need functions chunk to decode exports!");
                    }
                }
                b"start" => {
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
                b"code" => {
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
                b"data" => {
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
                b"name" => {
                    if names.is_some() {
                        panic!("duplicate data chunk!");
                    }
                    let count = r.read_var_u32() as usize;
                    let mut nm = Vec::with_capacity(count);

                    for _ in 0..count {
                        let fn_name = r.read_bytes();
                        let local_count = r.read_var_i32() as usize;
                        let mut local_names = Vec::with_capacity(local_count);

                        for _ in 0..local_count {
                            local_names.push(r.read_bytes());
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
}
