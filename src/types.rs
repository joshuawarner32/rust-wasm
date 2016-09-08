use std::{str, mem, fmt};
use std::num::Wrapping;

#[repr(u8)]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub enum Type {
    Int32 = 1,
    Int64 = 2,
    Float32 = 3,
    Float64 = 4
}

impl Type {
    pub fn from_u8(val: u8) -> Type {
        if val < 1 || val > 4 {
            panic!();
        }
        unsafe { mem::transmute(val) }
    }

    pub fn to_u8(&self) -> u8 {
        unsafe { mem::transmute(*self) }
    }

    pub fn size(&self) -> Size {
        match self {
            &Type::Int32 => Size::I32,
            &Type::Int64 => Size::I64,
            &Type::Float32 => Size::I32,
            &Type::Float64 => Size::I64,
        }
    }

    pub fn zero(&self) -> Dynamic {
        match self {
            &Type::Int32 => Dynamic::Int32(Wrapping(0)),
            &Type::Int64 => Dynamic::Int64(Wrapping(0)),
            &Type::Float32 => Dynamic::Float32(0f32),
            &Type::Float64 => Dynamic::Float64(0f64),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            &Type::Int32 => write!(f, "i32"),
            &Type::Int64 => write!(f, "i64"),
            &Type::Float32 => write!(f, "f32"),
            &Type::Float64 => write!(f, "f64"),
        }
    }
}

#[derive(Copy, Clone)]
pub enum Sign {
    Signed,
    Unsigned
}

impl Sign {
    pub fn text(self) -> &'static str {
        match self {
            Sign::Signed => "s",
            Sign::Unsigned => "u",
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum Size {
    I8,
    I16,
    I32,
    I64
}

impl Size {
    pub fn to_int(self) -> usize {
        match self {
            Size::I8 => 8,
            Size::I16 => 16,
            Size::I32 => 32,
            Size::I64 => 64,
        }
    }
}

#[derive(Copy, Clone)]
pub enum IntType {
    Int32,
    Int64
}

impl IntType {
    pub fn to_type(self) -> Type {
        match self {
            IntType::Int32 => Type::Int32,
            IntType::Int64 => Type::Int64,
        }
    }
    // pub fn min_unsigned_value(self) -> u64 {
    //     match self {
    //         IntType::Int32 => 0u64,
    //         IntType::Int64 => 0u64,
    //     }
    // }
    pub fn max_unsigned_value(self) -> u64 {
        match self {
            IntType::Int32 => 4294967295u64,
            IntType::Int64 => 18446744073709551615u64,
        }
    }
    pub fn min_signed_value(self) -> i64 {
        match self {
            IntType::Int32 => -2147483648i64,
            IntType::Int64 => -9223372036854775808i64,
        }
    }
    pub fn max_signed_value(self) -> i64 {
        match self {
            IntType::Int32 => 2147483647i64,
            IntType::Int64 => 9223372036854775807i64,
        }
    }
}

impl fmt::Display for IntType {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", self.to_type())
    }
}

#[derive(Copy, Clone)]
pub enum FloatType {
    Float32,
    Float64
}

impl FloatType {
    pub fn to_type(self) -> Type {
        match self {
            FloatType::Float32 => Type::Float32,
            FloatType::Float64 => Type::Float64,
        }
    }
}

impl fmt::Display for FloatType {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "{}", self.to_type())
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum Dynamic {
    Int32(Wrapping<u32>),
    Int64(Wrapping<u64>),
    Float32(f32),
    Float64(f64)
}

impl fmt::Display for Dynamic {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            &Dynamic::Int32(val) => write!(f, "i32:{}", val),
            &Dynamic::Int64(val) => write!(f, "i64:{}", val),
            &Dynamic::Float32(val) => write!(f, "f32:{}", val),
            &Dynamic::Float64(val) => write!(f, "f64:{}", val),
        }
    }
}

impl fmt::Debug for Dynamic {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt::Display::fmt(self, f)
    }
}

pub struct Pr<T>(pub T);

impl fmt::Display for Pr<Option<Dynamic>> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            &Pr(Some(Dynamic::Int32(val))) => write!(f, "i32:{}", val),
            &Pr(Some(Dynamic::Int64(val))) => write!(f, "i64:{}", val),
            &Pr(Some(Dynamic::Float32(val))) => write!(f, "f32:{}", val),
            &Pr(Some(Dynamic::Float64(val))) => write!(f, "f64:{}", val),
            &Pr(None) => write!(f, "void")
        }
    }
}

impl fmt::Display for Pr<Option<Type>> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            &Pr(Some(Type::Int32)) => write!(f, "i32"),
            &Pr(Some(Type::Int64)) => write!(f, "i64"),
            &Pr(Some(Type::Float32)) => write!(f, "f32"),
            &Pr(Some(Type::Float64)) => write!(f, "f64"),
            &Pr(None) => write!(f, "void")
        }
    }
}

pub struct NoType(pub Dynamic);

impl fmt::Display for NoType {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self.0 {
            Dynamic::Int32(val) => write!(f, "{}", val),
            Dynamic::Int64(val) => write!(f, "{}", val),
            Dynamic::Float32(val) => write!(f, "{}", val),
            Dynamic::Float64(val) => write!(f, "{}", val),
        }
    }
}

impl Dynamic {
    pub fn to_wu32(self) -> Wrapping<u32> {
        match self {
            Dynamic::Int32(v) => v,
            _ => panic!()
        }
    }
    pub fn to_wu64(self) -> Wrapping<u64> {
        match self {
            Dynamic::Int64(v) => v,
            _ => panic!()
        }
    }
    pub fn to_wi32(self) -> Wrapping<i32> {
        match self {
            Dynamic::Int32(v) => Wrapping(v.0 as i32),
            _ => panic!()
        }
    }
    pub fn to_wi64(self) -> Wrapping<i64> {
        match self {
            Dynamic::Int64(v) => Wrapping(v.0 as i64),
            _ => panic!()
        }
    }
    pub fn to_u32(self) -> u32 {
        self.to_wu32().0
    }
    pub fn to_u64(self) -> u64 {
        self.to_wu64().0
    }
    pub fn to_i32(self) -> i32 {
        self.to_wi32().0
    }
    pub fn to_i64(self) -> i64 {
        self.to_wi64().0
    }
    pub fn to_int(self) -> Wrapping<u64> {
        match self {
            Dynamic::Int32(v) => Wrapping(v.0 as u64),
            Dynamic::Int64(v) => v,
            _ => panic!()
        }
    }
    pub fn to_float(self) -> f64 {
        match self {
            Dynamic::Float32(v) => v as f64,
            Dynamic::Float64(v) => v,
            _ => panic!()
        }
    }
    pub fn to_f32(self) -> f32 {
        match self {
            Dynamic::Float32(v) => v,
            _ => panic!()
        }
    }
    pub fn to_f64(self) -> f64 {
        match self {
            Dynamic::Float64(v) => v,
            _ => panic!()
        }
    }
    pub fn from_i32(val: i32) -> Dynamic {
        Dynamic::Int32(Wrapping(val as u32))
    }
    pub fn from_i64(val: i64) -> Dynamic {
        Dynamic::Int64(Wrapping(val as u64))
    }
    pub fn from_u32(val: u32) -> Dynamic {
        Dynamic::Int32(Wrapping(val))
    }
    pub fn from_u64(val: u64) -> Dynamic {
        Dynamic::Int64(Wrapping(val))
    }
    pub fn from_float(ty: FloatType, val: f64) -> Dynamic {
        match ty {
            FloatType::Float32 => Dynamic::Float32(val as f32),
            FloatType::Float64 => Dynamic::Float64(val),
        }
    }
    pub fn from_int(ty: IntType, val: u64) -> Dynamic {
        match ty {
            IntType::Int32 => Dynamic::Int32(Wrapping(val as u32)),
            IntType::Int64 => Dynamic::Int64(Wrapping(val)),
        }
    }

    pub fn get_type(&self) -> Type {
        match self {
            &Dynamic::Int32(_) => Type::Int32,
            &Dynamic::Int64(_) => Type::Int64,
            &Dynamic::Float32(_) => Type::Float32,
            &Dynamic::Float64(_) => Type::Float64,
        }
    }
}
