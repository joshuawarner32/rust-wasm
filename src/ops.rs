use std::iter::{Iterator, IntoIterator};
use std::{mem, fmt};

use types::{Dynamic, Type, NoType};
use module::{FunctionIndex, TableIndex, ImportIndex};
use reader::Reader;

#[derive(Copy, Clone)]
enum Opcode {
    Nop = 0x00,
    Block = 0x01,
    Loop = 0x02,
    If = 0x03,
    Else = 0x04,
    Select = 0x05,
    Br = 0x06,
    BrIf = 0x07,
    BrTable = 0x08,
    Return = 0x09,
    Unreachable = 0x0a,
    Drop = 0x0b,
    End = 0x0f,
    I32Const = 0x10,
    I64Const = 0x11,
    F64Const = 0x12,
    F32Const = 0x13,
    GetLocal = 0x14,
    SetLocal = 0x15,
    TeeLocal = 0x19,
    Call = 0x16,
    CallIndirect = 0x17,
    CallImport = 0x18,
    I32Load8S = 0x20,
    I32Load8U = 0x21,
    I32Load16S = 0x22,
    I32Load16U = 0x23,
    I64Load8S = 0x24,
    I64Load8U = 0x25,
    I64Load16S = 0x26,
    I64Load16U = 0x27,
    I64Load32S = 0x28,
    I64Load32U = 0x29,
    I32Load = 0x2a,
    I64Load = 0x2b,
    F32Load = 0x2c,
    F64Load = 0x2d,
    I32Store8 = 0x2e,
    I32Store16 = 0x2f,
    I64Store8 = 0x30,
    I64Store16 = 0x31,
    I64Store32 = 0x32,
    I32Store = 0x33,
    I64Store = 0x34,
    F32Store = 0x35,
    F64Store = 0x36,
    CurrentMemory = 0x3b,
    GrowMemory = 0x39,
    I32Add = 0x40,
    I32Sub = 0x41,
    I32Mul = 0x42,
    I32DivS = 0x43,
    I32DivU = 0x44,
    I32RemS = 0x45,
    I32RemU = 0x46,
    I32And = 0x47,
    I32Or = 0x48,
    I32Xor = 0x49,
    I32Shl = 0x4a,
    I32ShrU = 0x4b,
    I32ShrS = 0x4c,
    I32Rotr = 0xb6,
    I32Rotl = 0xb7,
    I32Eq = 0x4d,
    I32Ne = 0x4e,
    I32LtS = 0x4f,
    I32LeS = 0x50,
    I32LtU = 0x51,
    I32LeU = 0x52,
    I32GtS = 0x53,
    I32GeS = 0x54,
    I32GtU = 0x55,
    I32GeU = 0x56,
    I32Clz = 0x57,
    I32Ctz = 0x58,
    I32Popcnt = 0x59,
    I32Eqz = 0x5a,
    I64Add = 0x5b,
    I64Sub = 0x5c,
    I64Mul = 0x5d,
    I64DivS = 0x5e,
    I64DivU = 0x5f,
    I64RemS = 0x60,
    I64RemU = 0x61,
    I64And = 0x62,
    I64Or = 0x63,
    I64Xor = 0x64,
    I64Shl = 0x65,
    I64ShrU = 0x66,
    I64ShrS = 0x67,
    I64Rotr = 0xb8,
    I64Rotl = 0xb9,
    I64Eq = 0x68,
    I64Ne = 0x69,
    I64LtS = 0x6a,
    I64LeS = 0x6b,
    I64LtU = 0x6c,
    I64LeU = 0x6d,
    I64GtS = 0x6e,
    I64GeS = 0x6f,
    I64GtU = 0x70,
    I64GeU = 0x71,
    I64Clz = 0x72,
    I64Ctz = 0x73,
    I64Popcnt = 0x74,
    I64Eqz = 0xba,
    F32Add = 0x75,
    F32Sub = 0x76,
    F32Mul = 0x77,
    F32Div = 0x78,
    F32Min = 0x79,
    F32Max = 0x7a,
    F32Abs = 0x7b,
    F32Neg = 0x7c,
    F32Copysign = 0x7d,
    F32Ceil = 0x7e,
    F32Floor = 0x7f,
    F32Trunc = 0x80,
    F32Nearest = 0x81,
    F32Sqrt = 0x82,
    F32Eq = 0x83,
    F32Ne = 0x84,
    F32Lt = 0x85,
    F32Le = 0x86,
    F32Gt = 0x87,
    F32Ge = 0x88,
    F64Add = 0x89,
    F64Sub = 0x8a,
    F64Mul = 0x8b,
    F64Div = 0x8c,
    F64Min = 0x8d,
    F64Max = 0x8e,
    F64Abs = 0x8f,
    F64Neg = 0x90,
    F64Copysign = 0x91,
    F64Ceil = 0x92,
    F64Floor = 0x93,
    F64Trunc = 0x94,
    F64Nearest = 0x95,
    F64Sqrt = 0x96,
    F64Eq = 0x97,
    F64Ne = 0x98,
    F64Lt = 0x99,
    F64Le = 0x9a,
    F64Gt = 0x9b,
    F64Ge = 0x9c,
    I32TruncSF32 = 0x9d,
    I32TruncSF64 = 0x9e,
    I32TruncUF32 = 0x9f,
    I32TruncUF64 = 0xa0,
    I32WrapI64 = 0xa1,
    I64TruncSF32 = 0xa2,
    I64TruncSF64 = 0xa3,
    I64TruncUF32 = 0xa4,
    I64TruncUF64 = 0xa5,
    I64ExtendSI32 = 0xa6,
    I64ExtendUI32 = 0xa7,
    F32ConvertSI32 = 0xa8,
    F32ConvertUI32 = 0xa9,
    F32ConvertSI64 = 0xaa,
    F32ConvertUI64 = 0xab,
    F32DemoteF64 = 0xac,
    F32ReinterpretI32 = 0xad,
    F64ConvertSI32 = 0xae,
    F64ConvertUI32 = 0xaf,
    F64ConvertSI64 = 0xb0,
    F64ConvertUI64 = 0xb1,
    F64PromoteF32 = 0xb2,
    F64ReinterpretI64 = 0xb3,
    I32ReinterpretF32 = 0xb4,
    I64ReinterpretF64 = 0xb5,
}

#[derive(Copy, Clone)]
pub enum Sign {
    Signed,
    Unsigned
}

#[derive(Copy, Clone)]
pub enum IntType {
    Int32,
    Int64
}

#[derive(Copy, Clone)]
pub enum Size {
    I8,
    I16,
    I32,
    I64
}

#[derive(Copy, Clone)]
pub enum FloatType {
    Float32,
    Float64
}

#[derive(Copy, Clone)]
pub struct MemImm {
    log_of_alignment: u32,
    offset: u32
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
}

#[derive(Copy, Clone)]
pub enum FloatUnOp {
    Abs,
    Neg,
    Copysign,
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

pub enum LinearOp<'a> {
    Nop,
    Block,
    Loop,
    If,
    Else,
    Select,
    Br{has_arg: bool, relative_depth: u32},
    BrIf{has_arg: bool, relative_depth: u32},
    BrTable{has_arg: bool, target_data: &'a [u8], default: u32},
    Return{has_arg: bool},
    Unreachable,
    Drop,
    End,

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

impl<'a> fmt::Display for LinearOp<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            &LinearOp::Nop => write!(f, "nop"),
            &LinearOp::Block => write!(f, "block"),
            &LinearOp::Loop => write!(f, "loop"),
            &LinearOp::If => write!(f, "if"),
            &LinearOp::Else => write!(f, "else"),
            &LinearOp::Select => write!(f, "select"),
            &LinearOp::Br{has_arg: _, relative_depth: _} => write!(f, "br"),
            &LinearOp::BrIf{has_arg: _, relative_depth: _} => write!(f, "br_if"),
            &LinearOp::BrTable{has_arg: _, target_data: _, default: _} => write!(f, "br_table"),
            &LinearOp::Return{has_arg} => write!(f, "return {}", if has_arg { "arg" } else { "" }),
            &LinearOp::Unreachable => write!(f, "unreachable"),
            &LinearOp::Drop => write!(f, "drop"),
            &LinearOp::End => write!(f, "end"),

            &LinearOp::Const(val) => write!(f, "{}.const {}", val.get_type(), NoType(val)),
            &LinearOp::GetLocal(index) => write!(f, "get_local {}", index),
            &LinearOp::SetLocal(index) => write!(f, "set_local {}", index),
            &LinearOp::TeeLocal(index) => write!(f, "tee_local {}", index),
            &LinearOp::Call{argument_count, index} => write!(f, "call {} {}", argument_count, index.0),
            &LinearOp::CallIndirect{argument_count, index} => write!(f, "call_indirect {} {}", argument_count, index.0),
            &LinearOp::CallImport{argument_count, index} => write!(f, "call_import {} {}", argument_count, index.0),
            &LinearOp::IntLoad(IntType, Sign, Size, MemImm) => write!(f, "IntLoad"),
            &LinearOp::FloatLoad(FloatType, MemImm) => write!(f, "FloatLoad"),
            &LinearOp::IntStore(IntType, Size, MemImm) => write!(f, "IntStore"),
            &LinearOp::FloatStore(FloatType, MemImm) => write!(f, "FloatStore"),

            &LinearOp::CurrentMemory => write!(f, "CurrentMemory"),
            &LinearOp::GrowMemory => write!(f, "GrowMemory"),

            &LinearOp::IntBin(IntType, IntBinOp) => write!(f, "IntBin"),
            &LinearOp::IntCmp(IntType, IntCmpOp) => write!(f, "IntCmp"),
            &LinearOp::IntUn(IntType, IntUnOp) => write!(f, "IntUn"),
            &LinearOp::IntEqz(IntType) => write!(f, "IntEqz"),
            &LinearOp::FloatBin(FloatType, FloatBinOp) => write!(f, "FloatBin"),
            &LinearOp::FloatUn(FloatType, FloatUnOp) => write!(f, "FloatUn"),
            &LinearOp::FloatCmp(FloatType, FloatCmpOp) => write!(f, "FloatCmp"),
            &LinearOp::FloatToInt(FloatType, IntType, Sign) => write!(f, "FloatToInt"),
            &LinearOp::IntExtend(Sign) => write!(f, "IntExtend"),
            &LinearOp::IntTruncate => write!(f, "IntTruncate"),
            &LinearOp::IntToFloat(IntType, Sign, FloatType) => write!(f, "IntToFloat"),
            &LinearOp::FloatConvert(FloatType) => write!(f, "FloatConvert"),
            &LinearOp::Reinterpret(Type, Type2) => write!(f, "Reinterpret"),
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
        if self.r.at_eof() {
            None
        } else {
            let b = self.r.read_u8();
            Some(match b {
                0x00 => LinearOp::Nop,
                0x01 => LinearOp::Block,
                0x02 => LinearOp::Loop,
                0x03 => LinearOp::If,
                0x04 => LinearOp::Else,
                0x05 => LinearOp::Select,
                0x06 => {
                    let has_arg = self.r.read_var_u1().unwrap();
                    let relative_depth = self.r.read_var_u32();
                    LinearOp::Br{
                        has_arg: has_arg,
                        relative_depth: relative_depth
                    }
                }
                0x07 => {
                    let has_arg = self.r.read_var_u1().unwrap();
                    let relative_depth = self.r.read_var_u32();
                    LinearOp::BrIf{
                        has_arg: has_arg,
                        relative_depth: relative_depth
                    }
                }
                0x08 => {
                    let has_arg = self.r.read_var_u1().unwrap_or(true);
                    let target_count = self.r.read_var_u32();
                    let target_data = self.r.read_bytes_with_len((target_count as usize) * 4);
                    let default = self.r.read_u32();

                    LinearOp::BrTable {
                        has_arg: has_arg,
                        target_data: target_data,
                        default: default
                    }
                }
                0x09 => {
                    let has_arg = self.r.read_var_u1().unwrap();
                    LinearOp::Return{has_arg: has_arg}
                }
                0x0a => LinearOp::Unreachable,
                0x0b => LinearOp::Drop,
                0x0f => LinearOp::End,
                0x10 => LinearOp::Const(Dynamic::from_i32(self.r.read_var_i32())),
                0x11 => LinearOp::Const(Dynamic::from_i64(self.r.read_var_i64())),
                0x12 => LinearOp::Const(Dynamic::Float64(unsafe { mem::transmute(self.r.read_u64()) })),
                0x13 => LinearOp::Const(Dynamic::Float32(unsafe { mem::transmute(self.r.read_u32()) })),
                0x14 => LinearOp::GetLocal(self.r.read_var_u32() as usize),
                0x15 => LinearOp::SetLocal(self.r.read_var_u32() as usize),
                0x19 => LinearOp::TeeLocal(self.r.read_var_u32() as usize),
                0x16 => {
                    let argument_count = self.r.read_var_u32();
                    let index = self.r.read_var_u32() as usize;
                    LinearOp::Call{
                        argument_count: argument_count,
                        index: FunctionIndex(index)
                    }
                }
                0x17 => {
                    let argument_count = self.r.read_var_u32();
                    let index = self.r.read_var_u32() as usize;
                    LinearOp::CallIndirect{
                        argument_count: argument_count,
                        index: TableIndex(index)
                    }
                }
                0x18 => {
                    let argument_count = self.r.read_var_u32();
                    let index = self.r.read_var_u32() as usize;
                    LinearOp::CallImport{
                        argument_count: argument_count,
                        index: ImportIndex(index)
                    }
                }
                0x20 => LinearOp::IntLoad(IntType::Int32, Sign::Signed, Size::I8, read_mem_imm(&mut self.r)),
                0x21 => LinearOp::IntLoad(IntType::Int32, Sign::Unsigned, Size::I8, read_mem_imm(&mut self.r)),
                0x22 => LinearOp::IntLoad(IntType::Int32, Sign::Signed, Size::I16, read_mem_imm(&mut self.r)),
                0x23 => LinearOp::IntLoad(IntType::Int32, Sign::Unsigned, Size::I16, read_mem_imm(&mut self.r)),
                0x24 => LinearOp::IntLoad(IntType::Int64, Sign::Signed, Size::I8, read_mem_imm(&mut self.r)),
                0x25 => LinearOp::IntLoad(IntType::Int64, Sign::Unsigned, Size::I8, read_mem_imm(&mut self.r)),
                0x26 => LinearOp::IntLoad(IntType::Int64, Sign::Signed, Size::I16, read_mem_imm(&mut self.r)),
                0x27 => LinearOp::IntLoad(IntType::Int64, Sign::Unsigned, Size::I16, read_mem_imm(&mut self.r)),
                0x28 => LinearOp::IntLoad(IntType::Int64, Sign::Signed, Size::I32, read_mem_imm(&mut self.r)),
                0x29 => LinearOp::IntLoad(IntType::Int64, Sign::Unsigned, Size::I32, read_mem_imm(&mut self.r)),
                0x2a => LinearOp::IntLoad(IntType::Int32, Sign::Unsigned, Size::I32, read_mem_imm(&mut self.r)),
                0x2b => LinearOp::IntLoad(IntType::Int64, Sign::Unsigned, Size::I64, read_mem_imm(&mut self.r)),
                0x2c => LinearOp::FloatLoad(FloatType::Float32, read_mem_imm(&mut self.r)),
                0x2d => LinearOp::FloatLoad(FloatType::Float64, read_mem_imm(&mut self.r)),
                0x2e => LinearOp::IntStore(IntType::Int32, Size::I8, read_mem_imm(&mut self.r)),
                0x2f => LinearOp::IntStore(IntType::Int32, Size::I16, read_mem_imm(&mut self.r)),
                0x30 => LinearOp::IntStore(IntType::Int64, Size::I8, read_mem_imm(&mut self.r)),
                0x31 => LinearOp::IntStore(IntType::Int64, Size::I16, read_mem_imm(&mut self.r)),
                0x32 => LinearOp::IntStore(IntType::Int64, Size::I32, read_mem_imm(&mut self.r)),
                0x33 => LinearOp::IntStore(IntType::Int32, Size::I32, read_mem_imm(&mut self.r)),
                0x34 => LinearOp::IntStore(IntType::Int64, Size::I64, read_mem_imm(&mut self.r)),
                0x35 => LinearOp::FloatStore(FloatType::Float32, read_mem_imm(&mut self.r)),
                0x36 => LinearOp::FloatStore(FloatType::Float64, read_mem_imm(&mut self.r)),
                0x3b => LinearOp::CurrentMemory,
                0x39 => LinearOp::GrowMemory,
                0x40 => LinearOp::IntBin(IntType::Int32, IntBinOp::Add),
                0x41 => LinearOp::IntBin(IntType::Int32, IntBinOp::Sub),
                0x42 => LinearOp::IntBin(IntType::Int32, IntBinOp::Mul),
                0x43 => LinearOp::IntBin(IntType::Int32, IntBinOp::DivS),
                0x44 => LinearOp::IntBin(IntType::Int32, IntBinOp::DivU),
                0x45 => LinearOp::IntBin(IntType::Int32, IntBinOp::RemS),
                0x46 => LinearOp::IntBin(IntType::Int32, IntBinOp::RemU),
                0x47 => LinearOp::IntBin(IntType::Int32, IntBinOp::And),
                0x48 => LinearOp::IntBin(IntType::Int32, IntBinOp::Or),
                0x49 => LinearOp::IntBin(IntType::Int32, IntBinOp::Xor),
                0x4a => LinearOp::IntBin(IntType::Int32, IntBinOp::Shl),
                0x4b => LinearOp::IntBin(IntType::Int32, IntBinOp::ShrU),
                0x4c => LinearOp::IntBin(IntType::Int32, IntBinOp::ShrS),
                0xb6 => LinearOp::IntBin(IntType::Int32, IntBinOp::Rotr),
                0xb7 => LinearOp::IntBin(IntType::Int32, IntBinOp::Rotl),
                0x4d => LinearOp::IntCmp(IntType::Int32, IntCmpOp::Eq),
                0x4e => LinearOp::IntCmp(IntType::Int32, IntCmpOp::Ne),
                0x4f => LinearOp::IntCmp(IntType::Int32, IntCmpOp::LtS),
                0x50 => LinearOp::IntCmp(IntType::Int32, IntCmpOp::LeS),
                0x51 => LinearOp::IntCmp(IntType::Int32, IntCmpOp::LtU),
                0x52 => LinearOp::IntCmp(IntType::Int32, IntCmpOp::LeU),
                0x53 => LinearOp::IntCmp(IntType::Int32, IntCmpOp::GtS),
                0x54 => LinearOp::IntCmp(IntType::Int32, IntCmpOp::GeS),
                0x55 => LinearOp::IntCmp(IntType::Int32, IntCmpOp::GtU),
                0x56 => LinearOp::IntCmp(IntType::Int32, IntCmpOp::GeU),
                0x57 => LinearOp::IntUn(IntType::Int32, IntUnOp::Clz),
                0x58 => LinearOp::IntUn(IntType::Int32, IntUnOp::Ctz),
                0x59 => LinearOp::IntUn(IntType::Int32, IntUnOp::Popcnt),
                0x5a => LinearOp::IntEqz(IntType::Int32),
                0x5b => LinearOp::IntBin(IntType::Int64, IntBinOp::Add),
                0x5c => LinearOp::IntBin(IntType::Int64, IntBinOp::Sub),
                0x5d => LinearOp::IntBin(IntType::Int64, IntBinOp::Mul),
                0x5e => LinearOp::IntBin(IntType::Int64, IntBinOp::DivS),
                0x5f => LinearOp::IntBin(IntType::Int64, IntBinOp::DivU),
                0x60 => LinearOp::IntBin(IntType::Int64, IntBinOp::RemS),
                0x61 => LinearOp::IntBin(IntType::Int64, IntBinOp::RemU),
                0x62 => LinearOp::IntBin(IntType::Int64, IntBinOp::And),
                0x63 => LinearOp::IntBin(IntType::Int64, IntBinOp::Or),
                0x64 => LinearOp::IntBin(IntType::Int64, IntBinOp::Xor),
                0x65 => LinearOp::IntBin(IntType::Int64, IntBinOp::Shl),
                0x66 => LinearOp::IntBin(IntType::Int64, IntBinOp::ShrU),
                0x67 => LinearOp::IntBin(IntType::Int64, IntBinOp::ShrS),
                0xb8 => LinearOp::IntBin(IntType::Int64, IntBinOp::Rotr),
                0xb9 => LinearOp::IntBin(IntType::Int64, IntBinOp::Rotl),
                0x68 => LinearOp::IntCmp(IntType::Int64, IntCmpOp::Eq),
                0x69 => LinearOp::IntCmp(IntType::Int64, IntCmpOp::Ne),
                0x6a => LinearOp::IntCmp(IntType::Int64, IntCmpOp::LtS),
                0x6b => LinearOp::IntCmp(IntType::Int64, IntCmpOp::LeS),
                0x6c => LinearOp::IntCmp(IntType::Int64, IntCmpOp::LtU),
                0x6d => LinearOp::IntCmp(IntType::Int64, IntCmpOp::LeU),
                0x6e => LinearOp::IntCmp(IntType::Int64, IntCmpOp::GtS),
                0x6f => LinearOp::IntCmp(IntType::Int64, IntCmpOp::GeS),
                0x70 => LinearOp::IntCmp(IntType::Int64, IntCmpOp::GtU),
                0x71 => LinearOp::IntCmp(IntType::Int64, IntCmpOp::GeU),
                0x72 => LinearOp::IntUn(IntType::Int64, IntUnOp::Clz),
                0x73 => LinearOp::IntUn(IntType::Int64, IntUnOp::Ctz),
                0x74 => LinearOp::IntUn(IntType::Int64, IntUnOp::Popcnt),
                0xba => LinearOp::IntEqz(IntType::Int64),
                0x75 => LinearOp::FloatBin(FloatType::Float32, FloatBinOp::Add),
                0x76 => LinearOp::FloatBin(FloatType::Float32, FloatBinOp::Sub),
                0x77 => LinearOp::FloatBin(FloatType::Float32, FloatBinOp::Mul),
                0x78 => LinearOp::FloatBin(FloatType::Float32, FloatBinOp::Div),
                0x79 => LinearOp::FloatBin(FloatType::Float32, FloatBinOp::Min),
                0x7a => LinearOp::FloatBin(FloatType::Float32, FloatBinOp::Max),
                0x7b => LinearOp::FloatUn(FloatType::Float32, FloatUnOp::Abs),
                0x7c => LinearOp::FloatUn(FloatType::Float32, FloatUnOp::Neg),
                0x7d => LinearOp::FloatUn(FloatType::Float32, FloatUnOp::Copysign),
                0x7e => LinearOp::FloatUn(FloatType::Float32, FloatUnOp::Ceil),
                0x7f => LinearOp::FloatUn(FloatType::Float32, FloatUnOp::Floor),
                0x80 => LinearOp::FloatUn(FloatType::Float32, FloatUnOp::Trunc),
                0x81 => LinearOp::FloatUn(FloatType::Float32, FloatUnOp::Nearest),
                0x82 => LinearOp::FloatUn(FloatType::Float32, FloatUnOp::Sqrt),
                0x83 => LinearOp::FloatCmp(FloatType::Float32, FloatCmpOp::Eq),
                0x84 => LinearOp::FloatCmp(FloatType::Float32, FloatCmpOp::Ne),
                0x85 => LinearOp::FloatCmp(FloatType::Float32, FloatCmpOp::Lt),
                0x86 => LinearOp::FloatCmp(FloatType::Float32, FloatCmpOp::Le),
                0x87 => LinearOp::FloatCmp(FloatType::Float32, FloatCmpOp::Gt),
                0x88 => LinearOp::FloatCmp(FloatType::Float32, FloatCmpOp::Ge),
                0x89 => LinearOp::FloatBin(FloatType::Float64, FloatBinOp::Add),
                0x8a => LinearOp::FloatBin(FloatType::Float64, FloatBinOp::Sub),
                0x8b => LinearOp::FloatBin(FloatType::Float64, FloatBinOp::Mul),
                0x8c => LinearOp::FloatBin(FloatType::Float64, FloatBinOp::Div),
                0x8d => LinearOp::FloatBin(FloatType::Float64, FloatBinOp::Min),
                0x8e => LinearOp::FloatBin(FloatType::Float64, FloatBinOp::Max),
                0x8f => LinearOp::FloatUn(FloatType::Float64, FloatUnOp::Abs),
                0x90 => LinearOp::FloatUn(FloatType::Float64, FloatUnOp::Neg),
                0x91 => LinearOp::FloatUn(FloatType::Float64, FloatUnOp::Copysign),
                0x92 => LinearOp::FloatUn(FloatType::Float64, FloatUnOp::Ceil),
                0x93 => LinearOp::FloatUn(FloatType::Float64, FloatUnOp::Floor),
                0x94 => LinearOp::FloatUn(FloatType::Float64, FloatUnOp::Trunc),
                0x95 => LinearOp::FloatUn(FloatType::Float64, FloatUnOp::Nearest),
                0x96 => LinearOp::FloatUn(FloatType::Float64, FloatUnOp::Sqrt),
                0x97 => LinearOp::FloatCmp(FloatType::Float64, FloatCmpOp::Eq),
                0x98 => LinearOp::FloatCmp(FloatType::Float64, FloatCmpOp::Ne),
                0x99 => LinearOp::FloatCmp(FloatType::Float64, FloatCmpOp::Lt),
                0x9a => LinearOp::FloatCmp(FloatType::Float64, FloatCmpOp::Le),
                0x9b => LinearOp::FloatCmp(FloatType::Float64, FloatCmpOp::Gt),
                0x9c => LinearOp::FloatCmp(FloatType::Float64, FloatCmpOp::Ge),
                0x9d => LinearOp::FloatToInt(FloatType::Float32, IntType::Int32, Sign::Signed),
                0x9e => LinearOp::FloatToInt(FloatType::Float64, IntType::Int32, Sign::Signed),
                0x9f => LinearOp::FloatToInt(FloatType::Float32, IntType::Int32, Sign::Unsigned),
                0xa0 => LinearOp::FloatToInt(FloatType::Float64, IntType::Int32, Sign::Unsigned),
                0xa1 => LinearOp::IntTruncate,
                0xa2 => LinearOp::FloatToInt(FloatType::Float32, IntType::Int64, Sign::Signed),
                0xa3 => LinearOp::FloatToInt(FloatType::Float64, IntType::Int64, Sign::Signed),
                0xa4 => LinearOp::FloatToInt(FloatType::Float32, IntType::Int64, Sign::Unsigned),
                0xa5 => LinearOp::FloatToInt(FloatType::Float64, IntType::Int64, Sign::Unsigned),
                0xa6 => LinearOp::IntExtend(Sign::Signed),
                0xa7 => LinearOp::IntExtend(Sign::Unsigned),
                0xa8 => LinearOp::IntToFloat(IntType::Int32, Sign::Signed, FloatType::Float32),
                0xa9 => LinearOp::IntToFloat(IntType::Int32, Sign::Unsigned, FloatType::Float32),
                0xaa => LinearOp::IntToFloat(IntType::Int64, Sign::Signed, FloatType::Float32),
                0xab => LinearOp::IntToFloat(IntType::Int64, Sign::Unsigned, FloatType::Float32),
                0xac => LinearOp::FloatConvert(FloatType::Float32),
                0xad => LinearOp::Reinterpret(Type::Int32, Type::Float32),
                0xae => LinearOp::IntToFloat(IntType::Int32, Sign::Signed, FloatType::Float64),
                0xaf => LinearOp::IntToFloat(IntType::Int32, Sign::Unsigned, FloatType::Float64),
                0xb0 => LinearOp::IntToFloat(IntType::Int64, Sign::Signed, FloatType::Float64),
                0xb1 => LinearOp::IntToFloat(IntType::Int64, Sign::Unsigned, FloatType::Float64),
                0xb2 => LinearOp::FloatConvert(FloatType::Float64),
                0xb3 => LinearOp::Reinterpret(Type::Float64, Type::Int64),
                0xb4 => LinearOp::Reinterpret(Type::Int32, Type::Float32),
                0xb5 => LinearOp::Reinterpret(Type::Int64, Type::Float64),
                x => panic!("unknown op: {:x} at {}/{}", x, self.r.position() - 1, self.r.len())
            })
        }
    }
}
