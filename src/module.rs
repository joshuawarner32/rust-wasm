use std::{str, mem, fmt};

use types::{Type, Pr, IntType, FloatType, Sign, Dynamic};
use reader::Reader;
use ops::{LinearOp, NormalOp, LinearOpReader, BlockOpReader,
    IntBinOp, IntCmpOp, IntUnOp,
    FloatBinOp, FloatCmpOp, FloatUnOp};

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
    pub ops: Vec<LinearOp<'static>>,
    pub local_types: Vec<Type>,
}

fn write_var_u32(ast: &mut Vec<u8>, v: u32) {
    let mut v = v;
    while v >= 0x80 {
        ast.push(((v & 0xff) as u8) | 0x80);
        v >>= 7;
    }
    ast.push((v & 0xff) as u8);
}

fn write_var_i32(ast: &mut Vec<u8>, v: i32) {
    let mut v = v;
    while v >= 64 || v < -64 {
        ast.push(unsafe { mem::transmute::<i32, u32>(v & 0xff) as u8 } | 0x80);
        v >>= 7;
    }
    ast.push((v & 0x7f) as u8);
}

fn write_var_u64(ast: &mut Vec<u8>, v: u64) {
    let mut v = v;
    while v >= 0x80 {
        ast.push(((v & 0xff) as u8) | 0x80);
        v >>= 7;
    }
    ast.push((v & 0xff) as u8);
}

fn write_var_i64(ast: &mut Vec<u8>, v: i64) {
    let mut v = v;
    while v >= 64 || v < -64 {
        ast.push(unsafe { mem::transmute::<i64, u64>(v & 0xff) as u8 } | 0x80);
        v >>= 7;
    }
    ast.push((v & 0x7f) as u8);
}

#[test]
fn test_write_var_i64() {
    let mut buf = Vec::new();
    for i in -256..256 {
        buf.clear();
        write_var_i64(&mut buf, i);
        let ib = Reader::new(&mut buf).read_var_i64();
        assert_eq!(i, ib);
    }
}

#[test]
fn test_write_var_u64() {
    let mut buf = Vec::new();
    for i in 0..256 {
        buf.clear();
        write_var_u64(&mut buf, i);
        let ib = Reader::new(&mut buf).read_var_u64();
        assert_eq!(i, ib);
    }
}

fn write_u32(ast: &mut Vec<u8>, v: u32) {
    ast.push(((v >> 0*8) & 0xff) as u8);
    ast.push(((v >> 1*8) & 0xff) as u8);
    ast.push(((v >> 2*8) & 0xff) as u8);
    ast.push(((v >> 3*8) & 0xff) as u8);
}

fn write_u64(ast: &mut Vec<u8>, v: u64) {
    ast.push(((v >> 0*8) & 0xff) as u8);
    ast.push(((v >> 1*8) & 0xff) as u8);
    ast.push(((v >> 2*8) & 0xff) as u8);
    ast.push(((v >> 3*8) & 0xff) as u8);
    ast.push(((v >> (4+0)*8) & 0xff) as u8);
    ast.push(((v >> (4+1)*8) & 0xff) as u8);
    ast.push(((v >> (4+2)*8) & 0xff) as u8);
    ast.push(((v >> (4+3)*8) & 0xff) as u8);
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

        if last.1 > 0 {
            locals.push(last);
        }

        let mut ast = Vec::new();

        for op in self.ops {
            match op {
                LinearOp::Block => ast.push(0x01),
                LinearOp::Loop => ast.push(0x02),
                LinearOp::If => ast.push(0x03),
                LinearOp::Else => ast.push(0x04),
                LinearOp::End => ast.push(0x0f),
                LinearOp::Normal(op) => match op {
                    NormalOp::Nop => ast.push(0x00),
                    NormalOp::Select => ast.push(0x05),
                    NormalOp::Br{has_arg, relative_depth} => {
                        ast.push(0x06);
                        ast.push(if has_arg { 1 } else { 0 });
                        write_var_u32(&mut ast, relative_depth);
                    }
                    NormalOp::BrIf{has_arg, relative_depth} => {
                        ast.push(0x07);
                        ast.push(if has_arg { 1 } else { 0 });
                        write_var_u32(&mut ast, relative_depth);
                    }
                    NormalOp::BrTable{has_arg, target_data, default} => {
                        ast.push(0x08);
                        ast.push(if has_arg { 1 } else { 0 });
                        assert!(target_data.len() % 4 == 0);
                        write_var_u32(&mut ast, (target_data.len() / 4) as u32);
                        ast.extend(target_data);
                        write_u32(&mut ast, default);
                    }
                    NormalOp::Return{has_arg} => {
                        ast.push(0x09);
                        ast.push(if has_arg { 1 } else { 0 });
                    }
                    NormalOp::Unreachable => ast.push(0x0a),
                    NormalOp::GetLocal(index) => {
                        ast.push(0x14);
                        ast.push(index as u8);
                    }
                    NormalOp::SetLocal(index) => {
                        ast.push(0x15);
                        ast.push(index as u8);
                    }
                    NormalOp::TeeLocal(index) => {
                        ast.push(0x19);
                        ast.push(index as u8);
                    }
                    NormalOp::Const(Dynamic::Int32(v)) => {
                        ast.push(0x10);
                        write_var_i32(&mut ast, unsafe { mem::transmute(v) })
                    }
                    NormalOp::Const(Dynamic::Int64(v)) => {
                        ast.push(0x11);
                        write_var_i64(&mut ast, unsafe { mem::transmute(v) })
                    }
                    NormalOp::Const(Dynamic::Float32(v)) => {
                        ast.push(0x13);
                        write_u32(&mut ast, unsafe { mem::transmute(v) })
                    }
                    NormalOp::Const(Dynamic::Float64(v)) => {
                        ast.push(0x12);
                        write_u64(&mut ast, unsafe { mem::transmute(v) })
                    }
                    NormalOp::Call{argument_count, index} => {
                        ast.push(0x16);
                        write_var_u32(&mut ast, argument_count);
                        write_var_u32(&mut ast, index.0 as u32);
                    }
                    NormalOp::IntBin(IntType::Int32, IntBinOp::Add) => ast.push(0x40),
                    NormalOp::IntBin(IntType::Int32, IntBinOp::Sub) => ast.push(0x41),
                    NormalOp::IntBin(IntType::Int32, IntBinOp::Mul) => ast.push(0x42),
                    NormalOp::IntBin(IntType::Int32, IntBinOp::DivS) => ast.push(0x43),
                    NormalOp::IntBin(IntType::Int32, IntBinOp::DivU) => ast.push(0x44),
                    NormalOp::IntBin(IntType::Int32, IntBinOp::RemS) => ast.push(0x45),
                    NormalOp::IntBin(IntType::Int32, IntBinOp::RemU) => ast.push(0x46),
                    NormalOp::IntBin(IntType::Int32, IntBinOp::And) => ast.push(0x47),
                    NormalOp::IntBin(IntType::Int32, IntBinOp::Or) => ast.push(0x48),
                    NormalOp::IntBin(IntType::Int32, IntBinOp::Xor) => ast.push(0x49),
                    NormalOp::IntBin(IntType::Int32, IntBinOp::Shl) => ast.push(0x4a),
                    NormalOp::IntBin(IntType::Int32, IntBinOp::ShrU) => ast.push(0x4b),
                    NormalOp::IntBin(IntType::Int32, IntBinOp::ShrS) => ast.push(0x4c),
                    NormalOp::IntBin(IntType::Int32, IntBinOp::Rotr) => ast.push(0xb6),
                    NormalOp::IntBin(IntType::Int32, IntBinOp::Rotl) => ast.push(0xb7),
                    NormalOp::IntCmp(IntType::Int32, IntCmpOp::Eq) => ast.push(0x4d),
                    NormalOp::IntCmp(IntType::Int32, IntCmpOp::Ne) => ast.push(0x4e),
                    NormalOp::IntCmp(IntType::Int32, IntCmpOp::LtS) => ast.push(0x4f),
                    NormalOp::IntCmp(IntType::Int32, IntCmpOp::LeS) => ast.push(0x50),
                    NormalOp::IntCmp(IntType::Int32, IntCmpOp::LtU) => ast.push(0x51),
                    NormalOp::IntCmp(IntType::Int32, IntCmpOp::LeU) => ast.push(0x52),
                    NormalOp::IntCmp(IntType::Int32, IntCmpOp::GtS) => ast.push(0x53),
                    NormalOp::IntCmp(IntType::Int32, IntCmpOp::GeS) => ast.push(0x54),
                    NormalOp::IntCmp(IntType::Int32, IntCmpOp::GtU) => ast.push(0x55),
                    NormalOp::IntCmp(IntType::Int32, IntCmpOp::GeU) => ast.push(0x56),
                    NormalOp::IntUn(IntType::Int32, IntUnOp::Clz) => ast.push(0x57),
                    NormalOp::IntUn(IntType::Int32, IntUnOp::Ctz) => ast.push(0x58),
                    NormalOp::IntUn(IntType::Int32, IntUnOp::Popcnt) => ast.push(0x59),
                    NormalOp::IntEqz(IntType::Int32) => ast.push(0x5a),
                    NormalOp::IntBin(IntType::Int64, IntBinOp::Add) => ast.push(0x5b),
                    NormalOp::IntBin(IntType::Int64, IntBinOp::Sub) => ast.push(0x5c),
                    NormalOp::IntBin(IntType::Int64, IntBinOp::Mul) => ast.push(0x5d),
                    NormalOp::IntBin(IntType::Int64, IntBinOp::DivS) => ast.push(0x5e),
                    NormalOp::IntBin(IntType::Int64, IntBinOp::DivU) => ast.push(0x5f),
                    NormalOp::IntBin(IntType::Int64, IntBinOp::RemS) => ast.push(0x60),
                    NormalOp::IntBin(IntType::Int64, IntBinOp::RemU) => ast.push(0x61),
                    NormalOp::IntBin(IntType::Int64, IntBinOp::And) => ast.push(0x62),
                    NormalOp::IntBin(IntType::Int64, IntBinOp::Or) => ast.push(0x63),
                    NormalOp::IntBin(IntType::Int64, IntBinOp::Xor) => ast.push(0x64),
                    NormalOp::IntBin(IntType::Int64, IntBinOp::Shl) => ast.push(0x65),
                    NormalOp::IntBin(IntType::Int64, IntBinOp::ShrU) => ast.push(0x66),
                    NormalOp::IntBin(IntType::Int64, IntBinOp::ShrS) => ast.push(0x67),
                    NormalOp::IntBin(IntType::Int64, IntBinOp::Rotr) => ast.push(0xb8),
                    NormalOp::IntBin(IntType::Int64, IntBinOp::Rotl) => ast.push(0xb9),
                    NormalOp::IntCmp(IntType::Int64, IntCmpOp::Eq) => ast.push(0x68),
                    NormalOp::IntCmp(IntType::Int64, IntCmpOp::Ne) => ast.push(0x69),
                    NormalOp::IntCmp(IntType::Int64, IntCmpOp::LtS) => ast.push(0x6a),
                    NormalOp::IntCmp(IntType::Int64, IntCmpOp::LeS) => ast.push(0x6b),
                    NormalOp::IntCmp(IntType::Int64, IntCmpOp::LtU) => ast.push(0x6c),
                    NormalOp::IntCmp(IntType::Int64, IntCmpOp::LeU) => ast.push(0x6d),
                    NormalOp::IntCmp(IntType::Int64, IntCmpOp::GtS) => ast.push(0x6e),
                    NormalOp::IntCmp(IntType::Int64, IntCmpOp::GeS) => ast.push(0x6f),
                    NormalOp::IntCmp(IntType::Int64, IntCmpOp::GtU) => ast.push(0x70),
                    NormalOp::IntCmp(IntType::Int64, IntCmpOp::GeU) => ast.push(0x71),
                    NormalOp::IntUn(IntType::Int64, IntUnOp::Clz) => ast.push(0x72),
                    NormalOp::IntUn(IntType::Int64, IntUnOp::Ctz) => ast.push(0x73),
                    NormalOp::IntUn(IntType::Int64, IntUnOp::Popcnt) => ast.push(0x74),
                    NormalOp::IntEqz(IntType::Int64) => ast.push(0xba),
                    NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Add) => ast.push(0x75),
                    NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Sub) => ast.push(0x76),
                    NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Mul) => ast.push(0x77),
                    NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Div) => ast.push(0x78),
                    NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Min) => ast.push(0x79),
                    NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Max) => ast.push(0x7a),
                    NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Copysign) => ast.push(0x7d),
                    NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Abs) => ast.push(0x7b),
                    NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Neg) => ast.push(0x7c),
                    NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Ceil) => ast.push(0x7e),
                    NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Floor) => ast.push(0x7f),
                    NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Trunc) => ast.push(0x80),
                    NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Nearest) => ast.push(0x81),
                    NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Sqrt) => ast.push(0x82),
                    NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Eq) => ast.push(0x83),
                    NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Ne) => ast.push(0x84),
                    NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Lt) => ast.push(0x85),
                    NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Le) => ast.push(0x86),
                    NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Gt) => ast.push(0x87),
                    NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Ge) => ast.push(0x88),
                    NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Add) => ast.push(0x89),
                    NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Sub) => ast.push(0x8a),
                    NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Mul) => ast.push(0x8b),
                    NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Div) => ast.push(0x8c),
                    NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Min) => ast.push(0x8d),
                    NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Max) => ast.push(0x8e),
                    NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Copysign) => ast.push(0x91),
                    NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Abs) => ast.push(0x8f),
                    NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Neg) => ast.push(0x90),
                    NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Ceil) => ast.push(0x92),
                    NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Floor) => ast.push(0x93),
                    NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Trunc) => ast.push(0x94),
                    NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Nearest) => ast.push(0x95),
                    NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Sqrt) => ast.push(0x96),
                    NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Eq) => ast.push(0x97),
                    NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Ne) => ast.push(0x98),
                    NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Lt) => ast.push(0x99),
                    NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Le) => ast.push(0x9a),
                    NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Gt) => ast.push(0x9b),
                    NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Ge) => ast.push(0x9c),
                    NormalOp::FloatToInt(FloatType::Float32, IntType::Int32, Sign::Signed) => ast.push(0x9d),
                    NormalOp::FloatToInt(FloatType::Float64, IntType::Int32, Sign::Signed) => ast.push(0x9e),
                    NormalOp::FloatToInt(FloatType::Float32, IntType::Int32, Sign::Unsigned) => ast.push(0x9f),
                    NormalOp::FloatToInt(FloatType::Float64, IntType::Int32, Sign::Unsigned) => ast.push(0xa0),
                    NormalOp::IntTruncate => ast.push(0xa1),
                    NormalOp::FloatToInt(FloatType::Float32, IntType::Int64, Sign::Signed) => ast.push(0xa2),
                    NormalOp::FloatToInt(FloatType::Float64, IntType::Int64, Sign::Signed) => ast.push(0xa3),
                    NormalOp::FloatToInt(FloatType::Float32, IntType::Int64, Sign::Unsigned) => ast.push(0xa4),
                    NormalOp::FloatToInt(FloatType::Float64, IntType::Int64, Sign::Unsigned) => ast.push(0xa5),
                    NormalOp::IntExtend(Sign::Signed) => ast.push(0xa6),
                    NormalOp::IntExtend(Sign::Unsigned) => ast.push(0xa7),
                    NormalOp::IntToFloat(IntType::Int32, Sign::Signed, FloatType::Float32) => ast.push(0xa8),
                    NormalOp::IntToFloat(IntType::Int32, Sign::Unsigned, FloatType::Float32) => ast.push(0xa9),
                    NormalOp::IntToFloat(IntType::Int64, Sign::Signed, FloatType::Float32) => ast.push(0xaa),
                    NormalOp::IntToFloat(IntType::Int64, Sign::Unsigned, FloatType::Float32) => ast.push(0xab),
                    NormalOp::FloatConvert(FloatType::Float32) => ast.push(0xac),
                    NormalOp::Reinterpret(Type::Int32, Type::Float32) => ast.push(0xad),
                    NormalOp::IntToFloat(IntType::Int32, Sign::Signed, FloatType::Float64) => ast.push(0xae),
                    NormalOp::IntToFloat(IntType::Int32, Sign::Unsigned, FloatType::Float64) => ast.push(0xaf),
                    NormalOp::IntToFloat(IntType::Int64, Sign::Signed, FloatType::Float64) => ast.push(0xb0),
                    NormalOp::IntToFloat(IntType::Int64, Sign::Unsigned, FloatType::Float64) => ast.push(0xb1),
                    NormalOp::FloatConvert(FloatType::Float64) => ast.push(0xb2),
                    NormalOp::Reinterpret(Type::Float64, Type::Int64) => ast.push(0xb3),
                    NormalOp::Reinterpret(Type::Float32, Type::Int32) => ast.push(0xb4),
                    NormalOp::Reinterpret(Type::Int64, Type::Float64) => ast.push(0xb5),
                    _ => panic!("unhandled: {}", op)
                }
            }
        }

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
        println!("looking for {}", str::from_utf8(name).unwrap());
        for e in &self.exports {
            println!("checking {}", str::from_utf8(e.function_name.as_bytes()).unwrap());
            if e.function_name.as_bytes() == name {
                return Some(e.function_index);
            }
        }
        None
    }

    pub fn find_by_debug_name(&self, name: &[u8]) -> Option<FunctionIndex> {
        println!("looking for debug {}", str::from_utf8(name).unwrap());
        for (i, e) in self.names.iter().enumerate() {
            println!("checking debug {}", str::from_utf8(e.function_name.as_bytes()).unwrap());
            if e.function_name.as_bytes() == name {
                return Some(FunctionIndex(i));
            }
        }
        None
    }

    pub fn find_name(&self, index: FunctionIndex) -> Option<&[u8]> {
        if self.names.len() > index.0 && self.names[index.0].function_name.as_bytes().len() > 0 {
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
