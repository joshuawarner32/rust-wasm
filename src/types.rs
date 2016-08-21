use std::{str, mem, fmt};
use std::num::Wrapping;

#[repr(u8)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
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

#[derive(Clone, Copy)]
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
    pub fn to_u32(self) -> Wrapping<u32> {
        match self {
            Dynamic::Int32(v) => v,
            _ => panic!()
        }
    }
    pub fn to_u64(self) -> Wrapping<u64> {
        match self {
            Dynamic::Int64(v) => v,
            _ => panic!()
        }
    }
    pub fn to_i32(self) -> Wrapping<i32> {
        match self {
            Dynamic::Int32(v) => Wrapping(v.0 as i32),
            _ => panic!()
        }
    }
    pub fn to_i64(self) -> Wrapping<i64> {
        match self {
            Dynamic::Int64(v) => Wrapping(v.0 as i64),
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

    pub fn get_type(&self) -> Type {
        match self {
            &Dynamic::Int32(val) => Type::Int32,
            &Dynamic::Int64(val) => Type::Int64,
            &Dynamic::Float32(val) => Type::Float32,
            &Dynamic::Float64(val) => Type::Float64,
        }
    }
}
