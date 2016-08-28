use std::iter::{Iterator, IntoIterator};
use std::{mem, fmt};

use types::{Dynamic, Type, NoType, IntType, FloatType, Size, Sign};
use module::{FunctionIndex, TableIndex, ImportIndex};
use reader::Reader;

// #[derive(Copy, Clone)]
// enum Opcode {
//     Nop = 0x00,
//     Block = 0x01,
//     Loop = 0x02,
//     If = 0x03,
//     Else = 0x04,
//     Select = 0x05,
//     Br = 0x06,
//     BrIf = 0x07,
//     BrTable = 0x08,
//     Return = 0x09,
//     Unreachable = 0x0a,
//     Drop = 0x0b,
//     End = 0x0f,
//     I32Const = 0x10,
//     I64Const = 0x11,
//     F64Const = 0x12,
//     F32Const = 0x13,
//     GetLocal = 0x14,
//     SetLocal = 0x15,
//     TeeLocal = 0x19,
//     Call = 0x16,
//     CallIndirect = 0x17,
//     CallImport = 0x18,
//     I32Load8S = 0x20,
//     I32Load8U = 0x21,
//     I32Load16S = 0x22,
//     I32Load16U = 0x23,
//     I64Load8S = 0x24,
//     I64Load8U = 0x25,
//     I64Load16S = 0x26,
//     I64Load16U = 0x27,
//     I64Load32S = 0x28,
//     I64Load32U = 0x29,
//     I32Load = 0x2a,
//     I64Load = 0x2b,
//     F32Load = 0x2c,
//     F64Load = 0x2d,
//     I32Store8 = 0x2e,
//     I32Store16 = 0x2f,
//     I64Store8 = 0x30,
//     I64Store16 = 0x31,
//     I64Store32 = 0x32,
//     I32Store = 0x33,
//     I64Store = 0x34,
//     F32Store = 0x35,
//     F64Store = 0x36,
//     CurrentMemory = 0x3b,
//     GrowMemory = 0x39,
//     I32Add = 0x40,
//     I32Sub = 0x41,
//     I32Mul = 0x42,
//     I32DivS = 0x43,
//     I32DivU = 0x44,
//     I32RemS = 0x45,
//     I32RemU = 0x46,
//     I32And = 0x47,
//     I32Or = 0x48,
//     I32Xor = 0x49,
//     I32Shl = 0x4a,
//     I32ShrU = 0x4b,
//     I32ShrS = 0x4c,
//     I32Rotr = 0xb6,
//     I32Rotl = 0xb7,
//     I32Eq = 0x4d,
//     I32Ne = 0x4e,
//     I32LtS = 0x4f,
//     I32LeS = 0x50,
//     I32LtU = 0x51,
//     I32LeU = 0x52,
//     I32GtS = 0x53,
//     I32GeS = 0x54,
//     I32GtU = 0x55,
//     I32GeU = 0x56,
//     I32Clz = 0x57,
//     I32Ctz = 0x58,
//     I32Popcnt = 0x59,
//     I32Eqz = 0x5a,
//     I64Add = 0x5b,
//     I64Sub = 0x5c,
//     I64Mul = 0x5d,
//     I64DivS = 0x5e,
//     I64DivU = 0x5f,
//     I64RemS = 0x60,
//     I64RemU = 0x61,
//     I64And = 0x62,
//     I64Or = 0x63,
//     I64Xor = 0x64,
//     I64Shl = 0x65,
//     I64ShrU = 0x66,
//     I64ShrS = 0x67,
//     I64Rotr = 0xb8,
//     I64Rotl = 0xb9,
//     I64Eq = 0x68,
//     I64Ne = 0x69,
//     I64LtS = 0x6a,
//     I64LeS = 0x6b,
//     I64LtU = 0x6c,
//     I64LeU = 0x6d,
//     I64GtS = 0x6e,
//     I64GeS = 0x6f,
//     I64GtU = 0x70,
//     I64GeU = 0x71,
//     I64Clz = 0x72,
//     I64Ctz = 0x73,
//     I64Popcnt = 0x74,
//     I64Eqz = 0xba,
//     F32Add = 0x75,
//     F32Sub = 0x76,
//     F32Mul = 0x77,
//     F32Div = 0x78,
//     F32Min = 0x79,
//     F32Max = 0x7a,
//     F32Abs = 0x7b,
//     F32Neg = 0x7c,
//     F32Copysign = 0x7d,
//     F32Ceil = 0x7e,
//     F32Floor = 0x7f,
//     F32Trunc = 0x80,
//     F32Nearest = 0x81,
//     F32Sqrt = 0x82,
//     F32Eq = 0x83,
//     F32Ne = 0x84,
//     F32Lt = 0x85,
//     F32Le = 0x86,
//     F32Gt = 0x87,
//     F32Ge = 0x88,
//     F64Add = 0x89,
//     F64Sub = 0x8a,
//     F64Mul = 0x8b,
//     F64Div = 0x8c,
//     F64Min = 0x8d,
//     F64Max = 0x8e,
//     F64Abs = 0x8f,
//     F64Neg = 0x90,
//     F64Copysign = 0x91,
//     F64Ceil = 0x92,
//     F64Floor = 0x93,
//     F64Trunc = 0x94,
//     F64Nearest = 0x95,
//     F64Sqrt = 0x96,
//     F64Eq = 0x97,
//     F64Ne = 0x98,
//     F64Lt = 0x99,
//     F64Le = 0x9a,
//     F64Gt = 0x9b,
//     F64Ge = 0x9c,
//     I32TruncSF32 = 0x9d,
//     I32TruncSF64 = 0x9e,
//     I32TruncUF32 = 0x9f,
//     I32TruncUF64 = 0xa0,
//     I32WrapI64 = 0xa1,
//     I64TruncSF32 = 0xa2,
//     I64TruncSF64 = 0xa3,
//     I64TruncUF32 = 0xa4,
//     I64TruncUF64 = 0xa5,
//     I64ExtendSI32 = 0xa6,
//     I64ExtendUI32 = 0xa7,
//     F32ConvertSI32 = 0xa8,
//     F32ConvertUI32 = 0xa9,
//     F32ConvertSI64 = 0xaa,
//     F32ConvertUI64 = 0xab,
//     F32DemoteF64 = 0xac,
//     F32ReinterpretI32 = 0xad,
//     F64ConvertSI32 = 0xae,
//     F64ConvertUI32 = 0xaf,
//     F64ConvertSI64 = 0xb0,
//     F64ConvertUI64 = 0xb1,
//     F64PromoteF32 = 0xb2,
//     F64ReinterpretI64 = 0xb3,
//     I32ReinterpretF32 = 0xb4,
//     I64ReinterpretF64 = 0xb5,
// }

#[derive(Copy, Clone)]
pub struct MemImm {
    pub log_of_alignment: u32,
    pub offset: u32
}

fn read_mem_imm<'a>(reader: &mut Reader<'a>) -> MemImm {
    let log_of_alignment = reader.read_var_u32();
    let offset = reader.read_var_u32();
    MemImm {
        log_of_alignment: log_of_alignment,
        offset: offset
    }
}

#[derive(Copy, Clone)]
pub enum IntBinOp {
    Add,
    Sub,
    Mul,
    DivS,
    DivU,
    RemS,
    RemU,
    And,
    Or,
    Xor,
    Shl,
    ShrU,
    ShrS,
    Rotr,
    Rotl,
}

#[derive(Copy, Clone)]
pub enum IntUnOp {
    Clz,
    Ctz,
    Popcnt,
}

#[derive(Copy, Clone)]
pub enum IntCmpOp {
    Eq,
    Ne,
    LtS,
    LeS,
    LtU,
    LeU,
    GtS,
    GeS,
    GtU,
    GeU,
}

#[derive(Copy, Clone)]
pub enum FloatBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Copysign,
}

#[derive(Copy, Clone)]
pub enum FloatUnOp {
    Abs,
    Neg,
    Ceil,
    Floor,
    Trunc,
    Nearest,
    Sqrt,
}

#[derive(Copy, Clone)]
pub enum FloatCmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

pub enum NormalOp<'a> {
    Nop,
    Select,
    Br{has_arg: bool, relative_depth: u32},
    BrIf{has_arg: bool, relative_depth: u32},
    BrTable{has_arg: bool, target_data: &'a [u8], default: u32},
    Return{has_arg: bool},
    Unreachable,
    Drop,
    Const(Dynamic),
    GetLocal(usize),
    SetLocal(usize),
    TeeLocal(usize),
    Call{argument_count: u32, index: FunctionIndex},
    CallIndirect{argument_count: u32, index: TableIndex},
    CallImport{argument_count: u32, index: ImportIndex},
    IntLoad(IntType, Sign, Size, MemImm),
    FloatLoad(FloatType, MemImm),
    IntStore(IntType, Size, MemImm),
    FloatStore(FloatType, MemImm),

    CurrentMemory,
    GrowMemory,

    IntBin(IntType, IntBinOp),
    IntCmp(IntType, IntCmpOp),
    IntUn(IntType, IntUnOp),
    IntEqz(IntType),
    FloatBin(FloatType, FloatBinOp),
    FloatUn(FloatType, FloatUnOp),
    FloatCmp(FloatType, FloatCmpOp),
    FloatToInt(FloatType, IntType, Sign),
    IntExtend(Sign),
    IntTruncate,
    IntToFloat(IntType, Sign, FloatType),
    FloatConvert(FloatType),
    Reinterpret(Type, Type),
}

pub enum LinearOp<'a> {
    Block,
    Loop,
    If,
    Else,
    End,
    Normal(NormalOp<'a>),
}

pub enum BlockOp<'a> {
    Block(Block<'a>),
    Normal(NormalOp<'a>),
}

pub enum Block<'a> {
    Block(Vec<BlockOp<'a>>),
    Loop(Vec<BlockOp<'a>>),
    If(Vec<BlockOp<'a>>, Vec<BlockOp<'a>>),
}

impl<'a> fmt::Display for LinearOp<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            &LinearOp::Block => write!(f, "block"),
            &LinearOp::Loop => write!(f, "loop"),
            &LinearOp::If => write!(f, "if"),
            &LinearOp::Else => write!(f, "else"),
            &LinearOp::End => write!(f, "end"),
            &LinearOp::Normal(ref x) => write!(f, "{}", x),
        }
    }
}

impl<'a> fmt::Display for NormalOp<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            &NormalOp::Nop => write!(f, "nop"),
            &NormalOp::Select => write!(f, "select"),
            &NormalOp::Br{has_arg, relative_depth} => write!(f, "br {}{}", if has_arg { "arg " } else { "" }, relative_depth),
            &NormalOp::BrIf{has_arg, relative_depth} => write!(f, "br_if {}{}", if has_arg { "arg " } else { "" }, relative_depth),
            &NormalOp::BrTable{has_arg, target_data: _, default: _} => write!(f, "br_table {}", if has_arg { "arg " } else { "" }),
            &NormalOp::Return{has_arg} => write!(f, "return {}", if has_arg { "arg" } else { "" }),
            &NormalOp::Unreachable => write!(f, "unreachable"),
            &NormalOp::Drop => write!(f, "drop"),

            &NormalOp::Const(val) => write!(f, "{}.const {}", val.get_type(), NoType(val)),
            &NormalOp::GetLocal(index) => write!(f, "get_local {}", index),
            &NormalOp::SetLocal(index) => write!(f, "set_local {}", index),
            &NormalOp::TeeLocal(index) => write!(f, "tee_local {}", index),
            &NormalOp::Call{argument_count, index} => write!(f, "call {} {}", argument_count, index.0),
            &NormalOp::CallIndirect{argument_count, index} => write!(f, "call_indirect {} {}", argument_count, index.0),
            &NormalOp::CallImport{argument_count, index} => write!(f, "call_import {} {}", argument_count, index.0),
            &NormalOp::IntLoad(ty, sign, size, _) => {
                if size == ty.to_type().size() {
                    write!(f, "{}.load", ty)
                } else {
                    write!(f, "{}.load{}_{}", ty, size.to_int(), sign.text())
                }
            }
            &NormalOp::FloatLoad(float_type, _) => write!(f, "FloatLoad"),
            &NormalOp::IntStore(int_type, size, _) => write!(f, "IntStore"),
            &NormalOp::FloatStore(float_type, _) => write!(f, "{}.store", float_type),

            &NormalOp::CurrentMemory => write!(f, "current_memory"),
            &NormalOp::GrowMemory => write!(f, "grow_memory"),

            &NormalOp::IntBin(ty, op) => {
                match op {
                    IntBinOp::Add => write!(f, "{}.add", ty),
                    IntBinOp::Sub => write!(f, "{}.sub", ty),
                    IntBinOp::Mul => write!(f, "{}.mul", ty),
                    IntBinOp::DivS => write!(f, "{}.divs", ty),
                    IntBinOp::DivU => write!(f, "{}.divu", ty),
                    IntBinOp::RemS => write!(f, "{}.rems", ty),
                    IntBinOp::RemU => write!(f, "{}.remu", ty),
                    IntBinOp::And => write!(f, "{}.and", ty),
                    IntBinOp::Or => write!(f, "{}.or", ty),
                    IntBinOp::Xor => write!(f, "{}.xor", ty),
                    IntBinOp::Shl => write!(f, "{}.shl", ty),
                    IntBinOp::ShrU => write!(f, "{}.shru", ty),
                    IntBinOp::ShrS => write!(f, "{}.shrs", ty),
                    IntBinOp::Rotr => write!(f, "{}.rotr", ty),
                    IntBinOp::Rotl => write!(f, "{}.rotl", ty),
                }
            }
            &NormalOp::IntCmp(ty, op) => {
                match op {
                    IntCmpOp::Eq => write!(f, "{}.eq", ty),
                    IntCmpOp::Ne => write!(f, "{}.ne", ty),
                    IntCmpOp::LtS => write!(f, "{}.lts", ty),
                    IntCmpOp::LeS => write!(f, "{}.les", ty),
                    IntCmpOp::LtU => write!(f, "{}.ltu", ty),
                    IntCmpOp::LeU => write!(f, "{}.leu", ty),
                    IntCmpOp::GtS => write!(f, "{}.gts", ty),
                    IntCmpOp::GeS => write!(f, "{}.ges", ty),
                    IntCmpOp::GtU => write!(f, "{}.gtu", ty),
                    IntCmpOp::GeU => write!(f, "{}.geu", ty),
                }
            }
            &NormalOp::IntUn(ty, op) => {
                match op {
                    IntUnOp::Clz => write!(f, "{}.clz", ty),
                    IntUnOp::Ctz => write!(f, "{}.ctz", ty),
                    IntUnOp::Popcnt => write!(f, "{}.popcnt", ty),
                }
            }
            &NormalOp::IntEqz(ty) => write!(f, "{}.eqz", ty),
            &NormalOp::FloatBin(ty, op) => {
                match op {
                    FloatBinOp::Add => write!(f, "{}.add", ty),
                    FloatBinOp::Sub => write!(f, "{}.sub", ty),
                    FloatBinOp::Mul => write!(f, "{}.mul", ty),
                    FloatBinOp::Div => write!(f, "{}.div", ty),
                    FloatBinOp::Min => write!(f, "{}.min", ty),
                    FloatBinOp::Max => write!(f, "{}.max", ty),
                    FloatBinOp::Copysign => write!(f, "{}.copysign", ty),
                }
            }
            &NormalOp::FloatUn(ty, op) => {
                match op {
                    FloatUnOp::Abs => write!(f, "{}.abs", ty),
                    FloatUnOp::Neg => write!(f, "{}.neg", ty),
                    FloatUnOp::Ceil => write!(f, "{}.ceil", ty),
                    FloatUnOp::Floor => write!(f, "{}.floor", ty),
                    FloatUnOp::Trunc => write!(f, "{}.trunc", ty),
                    FloatUnOp::Nearest => write!(f, "{}.nearest", ty),
                    FloatUnOp::Sqrt => write!(f, "{}.sqrt", ty),
                }
            }
            &NormalOp::FloatCmp(ty, op) => {
                match op {
                    FloatCmpOp::Eq => write!(f, "{}.eq", ty),
                    FloatCmpOp::Ne => write!(f, "{}.ne", ty),
                    FloatCmpOp::Lt => write!(f, "{}.lt", ty),
                    FloatCmpOp::Le => write!(f, "{}.le", ty),
                    FloatCmpOp::Gt => write!(f, "{}.gt", ty),
                    FloatCmpOp::Ge => write!(f, "{}.ge", ty),
                }
            }
            &NormalOp::FloatToInt(float_type, int_type, sign) =>
                write!(f, "{}.convert_{}/{}", int_type, sign.text(), float_type),
            &NormalOp::IntExtend(sign) => write!(f, "i64.extend_{}/i32", sign.text()),
            &NormalOp::IntTruncate => write!(f, "i32.wrap/i64"),
            &NormalOp::IntToFloat(int_type, sign, float_type) =>
                write!(f, "{}.convert_{}/{}", float_type, sign.text(), int_type),
            &NormalOp::FloatConvert(float_type) => {
                match float_type {
                    FloatType::Float32 => write!(f, "f32.demote/f64"),
                    FloatType::Float64 => write!(f, "f64.promote/f32"),
                }
            }
            &NormalOp::Reinterpret(Type::Int32, Type::Float32) => write!(f, "f32.reinterpret/i32"),
            &NormalOp::Reinterpret(Type::Float64, Type::Int64) => write!(f, "f64.reinterpret/i64"),
            &NormalOp::Reinterpret(Type::Float32, Type::Int32) => write!(f, "i32.reinterpret/f32"),
            &NormalOp::Reinterpret(Type::Int64, Type::Float64) => write!(f, "i64.reinterpret/f64"),
            &NormalOp::Reinterpret(_, _) => panic!()
        }
    }
}

pub struct Indented<T: fmt::Display>(pub usize, pub T);

impl<T: fmt::Display> fmt::Display for Indented<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        // Caution!!! This could get expensive if printing recursively!!!
        with_indent(self.0, &format!("{}", self.1), f)
    }
}

fn with_indent<'a>(indent: usize, text: &str, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    // Caution!!! This could get expensive if printing recursively!!!
    let itext = ::std::iter::repeat(' ').take(indent).collect::<String>();
    for (i, l) in text.split('\n').enumerate() {
        try!(write!(f, "{}{}{}", if i > 0 { "\n" } else { "" }, itext, l));
    }
    Ok(())
}

fn write_indented_ops<'a>(ops: &[BlockOp<'a>], f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    for o in ops {
        try!(with_indent(2, &format!("{}", o), f));
        writeln!(f, "");
    }
    Ok(())
}

impl<'a> fmt::Display for Block<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            &Block::Block(ref ops) => {
                try!(writeln!(f, "block"));
                write_indented_ops(ops, f)
            }
            &Block::Loop(ref ops) => {
                try!(writeln!(f, "loop"));
                write_indented_ops(ops, f)
            }
            &Block::If(ref then, ref otherwise) => {
                try!(writeln!(f, "if"));
                try!(write_indented_ops(then, f));
                if otherwise.len() > 0 {
                    try!(writeln!(f, "else"));
                    try!(write_indented_ops(otherwise, f))
                }
                Ok(())
            }
        }
    }
}

impl<'a> fmt::Display for BlockOp<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            &BlockOp::Block(ref b) => write!(f, "{}", b),
            &BlockOp::Normal(ref n) => write!(f, "{}", n),
        }
    }
}

pub struct LinearOpReader<'a> {
    r: Reader<'a>
}

impl<'a> LinearOpReader<'a> {
    pub fn new(data: &'a [u8]) -> LinearOpReader<'a> {
        LinearOpReader {
            r: Reader::new(data)
        }
    }
}

impl<'a> Iterator for LinearOpReader<'a> {
    type Item = LinearOp<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        read_linear_op(&mut self.r)
    }
}

fn read_linear_op<'a>(r: &mut Reader<'a>) -> Option<LinearOp<'a>> {
    if r.at_eof() {
        None
    } else {
        let b = r.read_u8();
        Some(match b {
            0x01 => LinearOp::Block,
            0x02 => LinearOp::Loop,
            0x03 => LinearOp::If,
            0x04 => LinearOp::Else,
            0x0f => LinearOp::End,
            x => LinearOp::Normal(match x {
                0x00 => NormalOp::Nop,
                0x05 => NormalOp::Select,
                0x06 => {
                    let has_arg = r.read_var_u1().unwrap();
                    let relative_depth = r.read_var_u32();
                    NormalOp::Br {
                        has_arg: has_arg,
                        relative_depth: relative_depth
                    }
                }
                0x07 => {
                    let has_arg = r.read_var_u1().unwrap();
                    let relative_depth = r.read_var_u32();
                    NormalOp::BrIf {
                        has_arg: has_arg,
                        relative_depth: relative_depth
                    }
                }
                0x08 => {
                    let has_arg = r.read_var_u1().unwrap_or(true);
                    let target_count = r.read_var_u32();
                    let target_data = r.read_bytes_with_len((target_count as usize) * 4);
                    let default = r.read_u32();

                    NormalOp::BrTable {
                        has_arg: has_arg,
                        target_data: target_data,
                        default: default
                    }
                }
                0x09 => {
                    let has_arg = r.read_var_u1().unwrap();
                    NormalOp::Return{has_arg: has_arg}
                }
                0x0a => NormalOp::Unreachable,
                0x0b => NormalOp::Drop,
                0x10 => NormalOp::Const(Dynamic::from_i32(r.read_var_i32())),
                0x11 => NormalOp::Const(Dynamic::from_i64(r.read_var_i64())),
                0x12 => NormalOp::Const(Dynamic::Float64(unsafe { mem::transmute(r.read_u64()) })),
                0x13 => NormalOp::Const(Dynamic::Float32(unsafe { mem::transmute(r.read_u32()) })),
                0x14 => NormalOp::GetLocal(r.read_var_u32() as usize),
                0x15 => NormalOp::SetLocal(r.read_var_u32() as usize),
                0x19 => NormalOp::TeeLocal(r.read_var_u32() as usize),
                0x16 => {
                    let argument_count = r.read_var_u32();
                    let index = r.read_var_u32() as usize;
                    NormalOp::Call{
                        argument_count: argument_count,
                        index: FunctionIndex(index)
                    }
                }
                0x17 => {
                    let argument_count = r.read_var_u32();
                    let index = r.read_var_u32() as usize;
                    NormalOp::CallIndirect{
                        argument_count: argument_count,
                        index: TableIndex(index)
                    }
                }
                0x18 => {
                    let argument_count = r.read_var_u32();
                    let index = r.read_var_u32() as usize;
                    NormalOp::CallImport{
                        argument_count: argument_count,
                        index: ImportIndex(index)
                    }
                }
                0x20 => NormalOp::IntLoad(IntType::Int32, Sign::Signed, Size::I8, read_mem_imm(r)),
                0x21 => NormalOp::IntLoad(IntType::Int32, Sign::Unsigned, Size::I8, read_mem_imm(r)),
                0x22 => NormalOp::IntLoad(IntType::Int32, Sign::Signed, Size::I16, read_mem_imm(r)),
                0x23 => NormalOp::IntLoad(IntType::Int32, Sign::Unsigned, Size::I16, read_mem_imm(r)),
                0x24 => NormalOp::IntLoad(IntType::Int64, Sign::Signed, Size::I8, read_mem_imm(r)),
                0x25 => NormalOp::IntLoad(IntType::Int64, Sign::Unsigned, Size::I8, read_mem_imm(r)),
                0x26 => NormalOp::IntLoad(IntType::Int64, Sign::Signed, Size::I16, read_mem_imm(r)),
                0x27 => NormalOp::IntLoad(IntType::Int64, Sign::Unsigned, Size::I16, read_mem_imm(r)),
                0x28 => NormalOp::IntLoad(IntType::Int64, Sign::Signed, Size::I32, read_mem_imm(r)),
                0x29 => NormalOp::IntLoad(IntType::Int64, Sign::Unsigned, Size::I32, read_mem_imm(r)),
                0x2a => NormalOp::IntLoad(IntType::Int32, Sign::Unsigned, Size::I32, read_mem_imm(r)),
                0x2b => NormalOp::IntLoad(IntType::Int64, Sign::Unsigned, Size::I64, read_mem_imm(r)),
                0x2c => NormalOp::FloatLoad(FloatType::Float32, read_mem_imm(r)),
                0x2d => NormalOp::FloatLoad(FloatType::Float64, read_mem_imm(r)),
                0x2e => NormalOp::IntStore(IntType::Int32, Size::I8, read_mem_imm(r)),
                0x2f => NormalOp::IntStore(IntType::Int32, Size::I16, read_mem_imm(r)),
                0x30 => NormalOp::IntStore(IntType::Int64, Size::I8, read_mem_imm(r)),
                0x31 => NormalOp::IntStore(IntType::Int64, Size::I16, read_mem_imm(r)),
                0x32 => NormalOp::IntStore(IntType::Int64, Size::I32, read_mem_imm(r)),
                0x33 => NormalOp::IntStore(IntType::Int32, Size::I32, read_mem_imm(r)),
                0x34 => NormalOp::IntStore(IntType::Int64, Size::I64, read_mem_imm(r)),
                0x35 => NormalOp::FloatStore(FloatType::Float32, read_mem_imm(r)),
                0x36 => NormalOp::FloatStore(FloatType::Float64, read_mem_imm(r)),
                0x3b => NormalOp::CurrentMemory,
                0x39 => NormalOp::GrowMemory,
                0x40 => NormalOp::IntBin(IntType::Int32, IntBinOp::Add),
                0x41 => NormalOp::IntBin(IntType::Int32, IntBinOp::Sub),
                0x42 => NormalOp::IntBin(IntType::Int32, IntBinOp::Mul),
                0x43 => NormalOp::IntBin(IntType::Int32, IntBinOp::DivS),
                0x44 => NormalOp::IntBin(IntType::Int32, IntBinOp::DivU),
                0x45 => NormalOp::IntBin(IntType::Int32, IntBinOp::RemS),
                0x46 => NormalOp::IntBin(IntType::Int32, IntBinOp::RemU),
                0x47 => NormalOp::IntBin(IntType::Int32, IntBinOp::And),
                0x48 => NormalOp::IntBin(IntType::Int32, IntBinOp::Or),
                0x49 => NormalOp::IntBin(IntType::Int32, IntBinOp::Xor),
                0x4a => NormalOp::IntBin(IntType::Int32, IntBinOp::Shl),
                0x4b => NormalOp::IntBin(IntType::Int32, IntBinOp::ShrU),
                0x4c => NormalOp::IntBin(IntType::Int32, IntBinOp::ShrS),
                0xb6 => NormalOp::IntBin(IntType::Int32, IntBinOp::Rotr),
                0xb7 => NormalOp::IntBin(IntType::Int32, IntBinOp::Rotl),
                0x4d => NormalOp::IntCmp(IntType::Int32, IntCmpOp::Eq),
                0x4e => NormalOp::IntCmp(IntType::Int32, IntCmpOp::Ne),
                0x4f => NormalOp::IntCmp(IntType::Int32, IntCmpOp::LtS),
                0x50 => NormalOp::IntCmp(IntType::Int32, IntCmpOp::LeS),
                0x51 => NormalOp::IntCmp(IntType::Int32, IntCmpOp::LtU),
                0x52 => NormalOp::IntCmp(IntType::Int32, IntCmpOp::LeU),
                0x53 => NormalOp::IntCmp(IntType::Int32, IntCmpOp::GtS),
                0x54 => NormalOp::IntCmp(IntType::Int32, IntCmpOp::GeS),
                0x55 => NormalOp::IntCmp(IntType::Int32, IntCmpOp::GtU),
                0x56 => NormalOp::IntCmp(IntType::Int32, IntCmpOp::GeU),
                0x57 => NormalOp::IntUn(IntType::Int32, IntUnOp::Clz),
                0x58 => NormalOp::IntUn(IntType::Int32, IntUnOp::Ctz),
                0x59 => NormalOp::IntUn(IntType::Int32, IntUnOp::Popcnt),
                0x5a => NormalOp::IntEqz(IntType::Int32),
                0x5b => NormalOp::IntBin(IntType::Int64, IntBinOp::Add),
                0x5c => NormalOp::IntBin(IntType::Int64, IntBinOp::Sub),
                0x5d => NormalOp::IntBin(IntType::Int64, IntBinOp::Mul),
                0x5e => NormalOp::IntBin(IntType::Int64, IntBinOp::DivS),
                0x5f => NormalOp::IntBin(IntType::Int64, IntBinOp::DivU),
                0x60 => NormalOp::IntBin(IntType::Int64, IntBinOp::RemS),
                0x61 => NormalOp::IntBin(IntType::Int64, IntBinOp::RemU),
                0x62 => NormalOp::IntBin(IntType::Int64, IntBinOp::And),
                0x63 => NormalOp::IntBin(IntType::Int64, IntBinOp::Or),
                0x64 => NormalOp::IntBin(IntType::Int64, IntBinOp::Xor),
                0x65 => NormalOp::IntBin(IntType::Int64, IntBinOp::Shl),
                0x66 => NormalOp::IntBin(IntType::Int64, IntBinOp::ShrU),
                0x67 => NormalOp::IntBin(IntType::Int64, IntBinOp::ShrS),
                0xb8 => NormalOp::IntBin(IntType::Int64, IntBinOp::Rotr),
                0xb9 => NormalOp::IntBin(IntType::Int64, IntBinOp::Rotl),
                0x68 => NormalOp::IntCmp(IntType::Int64, IntCmpOp::Eq),
                0x69 => NormalOp::IntCmp(IntType::Int64, IntCmpOp::Ne),
                0x6a => NormalOp::IntCmp(IntType::Int64, IntCmpOp::LtS),
                0x6b => NormalOp::IntCmp(IntType::Int64, IntCmpOp::LeS),
                0x6c => NormalOp::IntCmp(IntType::Int64, IntCmpOp::LtU),
                0x6d => NormalOp::IntCmp(IntType::Int64, IntCmpOp::LeU),
                0x6e => NormalOp::IntCmp(IntType::Int64, IntCmpOp::GtS),
                0x6f => NormalOp::IntCmp(IntType::Int64, IntCmpOp::GeS),
                0x70 => NormalOp::IntCmp(IntType::Int64, IntCmpOp::GtU),
                0x71 => NormalOp::IntCmp(IntType::Int64, IntCmpOp::GeU),
                0x72 => NormalOp::IntUn(IntType::Int64, IntUnOp::Clz),
                0x73 => NormalOp::IntUn(IntType::Int64, IntUnOp::Ctz),
                0x74 => NormalOp::IntUn(IntType::Int64, IntUnOp::Popcnt),
                0xba => NormalOp::IntEqz(IntType::Int64),
                0x75 => NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Add),
                0x76 => NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Sub),
                0x77 => NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Mul),
                0x78 => NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Div),
                0x79 => NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Min),
                0x7a => NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Max),
                0x7d => NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Copysign),
                0x7b => NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Abs),
                0x7c => NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Neg),
                0x7e => NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Ceil),
                0x7f => NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Floor),
                0x80 => NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Trunc),
                0x81 => NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Nearest),
                0x82 => NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Sqrt),
                0x83 => NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Eq),
                0x84 => NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Ne),
                0x85 => NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Lt),
                0x86 => NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Le),
                0x87 => NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Gt),
                0x88 => NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Ge),
                0x89 => NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Add),
                0x8a => NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Sub),
                0x8b => NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Mul),
                0x8c => NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Div),
                0x8d => NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Min),
                0x8e => NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Max),
                0x91 => NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Copysign),
                0x8f => NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Abs),
                0x90 => NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Neg),
                0x92 => NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Ceil),
                0x93 => NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Floor),
                0x94 => NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Trunc),
                0x95 => NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Nearest),
                0x96 => NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Sqrt),
                0x97 => NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Eq),
                0x98 => NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Ne),
                0x99 => NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Lt),
                0x9a => NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Le),
                0x9b => NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Gt),
                0x9c => NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Ge),
                0x9d => NormalOp::FloatToInt(FloatType::Float32, IntType::Int32, Sign::Signed),
                0x9e => NormalOp::FloatToInt(FloatType::Float64, IntType::Int32, Sign::Signed),
                0x9f => NormalOp::FloatToInt(FloatType::Float32, IntType::Int32, Sign::Unsigned),
                0xa0 => NormalOp::FloatToInt(FloatType::Float64, IntType::Int32, Sign::Unsigned),
                0xa1 => NormalOp::IntTruncate,
                0xa2 => NormalOp::FloatToInt(FloatType::Float32, IntType::Int64, Sign::Signed),
                0xa3 => NormalOp::FloatToInt(FloatType::Float64, IntType::Int64, Sign::Signed),
                0xa4 => NormalOp::FloatToInt(FloatType::Float32, IntType::Int64, Sign::Unsigned),
                0xa5 => NormalOp::FloatToInt(FloatType::Float64, IntType::Int64, Sign::Unsigned),
                0xa6 => NormalOp::IntExtend(Sign::Signed),
                0xa7 => NormalOp::IntExtend(Sign::Unsigned),
                0xa8 => NormalOp::IntToFloat(IntType::Int32, Sign::Signed, FloatType::Float32),
                0xa9 => NormalOp::IntToFloat(IntType::Int32, Sign::Unsigned, FloatType::Float32),
                0xaa => NormalOp::IntToFloat(IntType::Int64, Sign::Signed, FloatType::Float32),
                0xab => NormalOp::IntToFloat(IntType::Int64, Sign::Unsigned, FloatType::Float32),
                0xac => NormalOp::FloatConvert(FloatType::Float32),
                0xad => NormalOp::Reinterpret(Type::Int32, Type::Float32),
                0xae => NormalOp::IntToFloat(IntType::Int32, Sign::Signed, FloatType::Float64),
                0xaf => NormalOp::IntToFloat(IntType::Int32, Sign::Unsigned, FloatType::Float64),
                0xb0 => NormalOp::IntToFloat(IntType::Int64, Sign::Signed, FloatType::Float64),
                0xb1 => NormalOp::IntToFloat(IntType::Int64, Sign::Unsigned, FloatType::Float64),
                0xb2 => NormalOp::FloatConvert(FloatType::Float64),
                0xb3 => NormalOp::Reinterpret(Type::Float64, Type::Int64),
                0xb4 => NormalOp::Reinterpret(Type::Float32, Type::Int32),
                0xb5 => NormalOp::Reinterpret(Type::Int64, Type::Float64),
                x => panic!("unknown op: {:x} at {}/{}", x, r.position() - 1, r.len())
            })
        })
    }
}

enum BlockStackEl<'a> {
    Block(Vec<BlockOp<'a>>),
    Loop(Vec<BlockOp<'a>>),
    If(bool, Vec<BlockOp<'a>>, Vec<BlockOp<'a>>),
}

fn push_block<'a>(op: BlockOp<'a>, blocks: &mut Vec<BlockStackEl<'a>>) -> Option<BlockOp<'a>> {
    match blocks.last_mut() {
        Some(b) => {
            match b {
                &mut BlockStackEl::Block(ref mut ops) => ops.push(op),
                &mut BlockStackEl::Loop(ref mut ops) => ops.push(op),
                &mut BlockStackEl::If(in_cond, ref mut then, ref mut otherwise) =>
                    if in_cond { then } else { otherwise }.push(op),
            }
            None
        }
        None => Some(op),
    }
}

impl<'a> BlockOp<'a> {
    pub fn parse(r: &mut Reader<'a>) -> BlockOp<'a> {
        let mut blocks = Vec::new();

        while let Some(l) = read_linear_op(r) {
            match l {
                LinearOp::Block => blocks.push(BlockStackEl::Block(Vec::new())),
                LinearOp::Loop => blocks.push(BlockStackEl::Loop(Vec::new())),
                LinearOp::If => blocks.push(BlockStackEl::If(true, Vec::new(), Vec::new())),
                LinearOp::Else => {
                    match blocks.last_mut().unwrap() {
                        &mut BlockStackEl::If(ref mut in_cond, _, _) => {
                            assert!(*in_cond);
                            *in_cond = false;
                        }
                        _ => panic!()
                    }
                }
                LinearOp::End => {
                    let b = match blocks.pop().unwrap() {
                        BlockStackEl::Block(ops) => BlockOp::Block(Block::Block(ops)),
                        BlockStackEl::Loop(ops) => BlockOp::Block(Block::Loop(ops)),
                        BlockStackEl::If(_, then, otherwise) => BlockOp::Block(Block::If(then, otherwise)),
                    };
                    match push_block(b, &mut blocks) {
                        None => {}
                        Some(val) => return val,
                    }
                }
                LinearOp::Normal(x) => match push_block(BlockOp::Normal(x), &mut blocks) {
                    None => {}
                    Some(val) => return val,
                },
            }
        }

        panic!();
    }
}

pub struct BlockOpReader<'a> {
    r: Reader<'a>
}

impl<'a> BlockOpReader<'a> {
    pub fn new(data: &'a [u8]) -> BlockOpReader<'a> {
        BlockOpReader {
            r: Reader::new(data)
        }
    }
}

impl<'a> Iterator for BlockOpReader<'a> {
    type Item = BlockOp<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.r.at_eof() {
            None
        } else {
            Some(BlockOp::parse(&mut self.r))
        }
    }
}
