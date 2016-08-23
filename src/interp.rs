use std::mem;
use std::num::Wrapping;

use module::{Module, FunctionIndex};
use types::{Type, Dynamic, IntType, FloatType};
use ops::{
    BlockOp, Block, NormalOp, Sign, Size, MemImm,
    IntBinOp, IntCmpOp, IntUnOp, FloatBinOp, FloatUnOp, FloatCmpOp};

pub struct Memory(Vec<u8>);

impl Memory {
    fn store_u8(&mut self, addr: usize, val: u8) {
        self.0[addr] = val;
    }

    fn store_u16(&mut self, addr: usize, val: u16) {
        self.0[addr + 0] = (val >> 0*8) as u8;
        self.0[addr + 1] = (val >> 1*8) as u8;
    }

    fn store_u32(&mut self, addr: usize, val: u32) {
        self.0[addr + 0] = (val >> 0*8) as u8;
        self.0[addr + 1] = (val >> 1*8) as u8;
        self.0[addr + 2] = (val >> 2*8) as u8;
        self.0[addr + 3] = (val >> 3*8) as u8;
    }

    fn store_u64(&mut self, addr: usize, val: u64) {
        self.0[addr + 0] = (val >> 0*8) as u8;
        self.0[addr + 1] = (val >> 1*8) as u8;
        self.0[addr + 2] = (val >> 2*8) as u8;
        self.0[addr + 3] = (val >> 3*8) as u8;
        self.0[addr + 4] = (val >> 4*8) as u8;
        self.0[addr + 5] = (val >> 5*8) as u8;
        self.0[addr + 6] = (val >> 6*8) as u8;
        self.0[addr + 7] = (val >> 7*8) as u8;
    }

    fn load_u8(&self, addr: usize) -> u8 {
        self.0[addr]
    }

    fn load_u16(&self, addr: usize) -> u16 {
        ((self.0[addr + 0] as u16) << 0*8) |
        ((self.0[addr + 1] as u16) << 1*8)
    }

    fn load_u32(&self, addr: usize) -> u32 {
        ((self.0[addr + 0] as u32) << 0*8) |
        ((self.0[addr + 1] as u32) << 1*8) |
        ((self.0[addr + 2] as u32) << 2*8) |
        ((self.0[addr + 3] as u32) << 3*8)
    }

    fn load_u64(&self, addr: usize) -> u64 {
        ((self.0[addr + 0] as u64) << 0*8) |
        ((self.0[addr + 1] as u64) << 1*8) |
        ((self.0[addr + 2] as u64) << 2*8) |
        ((self.0[addr + 3] as u64) << 3*8) |
        ((self.0[addr + 4] as u64) << 4*8) |
        ((self.0[addr + 5] as u64) << 5*8) |
        ((self.0[addr + 6] as u64) << 6*8) |
        ((self.0[addr + 7] as u64) << 7*8)
    }

    fn load_int(&self, addr: u32, inttype: IntType, sign: Sign, size: Size, memimm: MemImm) -> Dynamic {
        match size {
            Size::I8 => extend_u8(self.load_u8((addr + memimm.offset) as usize), inttype, sign),
            Size::I16 => extend_u16(self.load_u16((addr + memimm.offset) as usize), inttype, sign),
            Size::I32 => extend_u32(self.load_u32((addr + memimm.offset) as usize), inttype, sign),
            Size::I64 => extend_u64(self.load_u64((addr + memimm.offset) as usize), inttype, sign),
        }
    }

    fn load_float(&self, addr: u32, floattype: FloatType, memimm: MemImm) -> Dynamic {
        match floattype {
            FloatType::Float32 => Dynamic::Float32(unsafe {
                mem::transmute(self.load_u32((addr + memimm.offset) as usize))
            }),
            FloatType::Float64 => Dynamic::Float64(unsafe {
                mem::transmute(self.load_u64((addr + memimm.offset) as usize))
            })
        }
    }

    fn store_int(&mut self, addr: u32, value: Dynamic, size: Size, memimm: MemImm) {
        match size {
            Size::I8 => self.store_u8((addr + memimm.offset) as usize, (value.to_int().0 & ((1 << 8) - 1)) as u8),
            Size::I16 => self.store_u16((addr + memimm.offset) as usize, (value.to_int().0 & ((1 << 16) - 1)) as u16),
            Size::I32 => self.store_u32((addr + memimm.offset) as usize, (value.to_int().0 & ((1 << 32) - 1)) as u32),
            Size::I64 => self.store_u64((addr + memimm.offset) as usize, value.to_int().0),
        }
    }

    fn store_float(&mut self, addr: u32, value: Dynamic, floattype: FloatType, memimm: MemImm) {
        assert!(value.get_type() == floattype.to_type());
        match floattype {
            FloatType::Float32 => self.store_u32((addr + memimm.offset) as usize, unsafe {
                mem::transmute(value.to_f32())
            }),
            FloatType::Float64 => self.store_u64((addr + memimm.offset) as usize, unsafe {
                mem::transmute(value.to_f64())
            })
        }
    }

}

pub struct Instance<'a> {
    memory: Memory,
    module: &'a Module<'a>,
}

fn read_u32(data: &[u8]) -> u32 {
    ((data[0] as u32) << 0*8) +
    ((data[1] as u32) << 1*8) +
    ((data[2] as u32) << 2*8) +
    ((data[3] as u32) << 3*8)
}

impl<'a> Instance<'a> {
    pub fn new(module: &'a Module<'a>) -> Instance<'a> {
        let mut memory = Vec::with_capacity(module.memory_info.initial_64k_pages * 64 * 1024);
        memory.resize(module.memory_info.initial_64k_pages * 64 * 1024, 0);

        for m in &module.memory_chunks {
            let newlen = ::std::cmp::min(m.offset + m.data.len(), memory.len());
            memory.resize(newlen, 0);
            memory[m.offset..m.offset + m.data.len()].copy_from_slice(&m.data);
        }

        Instance {
            memory: Memory(memory),
            module: module
        }
    }

    pub fn invoke(&mut self, func: FunctionIndex, args: &[Dynamic]) -> Option<Dynamic> {
        println!("running {}", self.module.find_name(func).unwrap_or("<unknown>"));

        let ty = self.module.functions[func.0];
        if args.len() != ty.param_types.len() {
            panic!("expected {} args, but got {}", ty.param_types.len(), args.len());
        }
        let f = &self.module.code[func.0];

        let root_ops = f.block_ops().collect::<Vec<_>>();

        let local_count: usize = f.locals.iter().map(|e|e.1).fold(0, |a, b|a+b);
        let mut locals = Vec::with_capacity(args.len() + local_count);
        locals.resize(args.len() + local_count, Dynamic::from_u32(0));
        locals[..args.len()].copy_from_slice(args);

        struct Context<'b, 'a: 'b> {
            instance: &'b mut Instance<'a>,
            locals: Vec<Dynamic>,
            stack: Vec<Dynamic>,
        }

        enum Res {
            Value(Option<Dynamic>),
            Branch(u32, Option<Dynamic>),
            Return(Option<Dynamic>)
        }

        macro_rules! prv {
            ($val:expr) => {{ let val = $val; match val { Res::Value(v) => v, val => return val}}}
        }

        macro_rules! pr {
            ($val:expr) => {{ let val = $val; match val { Res::Value(Some(v)) => v, Res::Value(None) => panic!(), val => return val}}}
        }

        macro_rules! prb {
            ($val:expr) => {{ let val = $val; match val { Res::Value(v) => {}, val => return val}}}
        }

        fn run_block<'a>(context: &'a mut Context, ops: &[BlockOp]) -> Res {
            if ops.len() > 0 {
                for i in &ops[..ops.len() - 1] {
                    match run_instr(context, i) {
                        Res::Value(Some(v)) => context.stack.push(v),
                        Res::Value(None) => {}
                        val => return val
                    }
                }
                run_instr(context, &ops[ops.len() - 1])
            } else {
                Res::Value(None)
            }
        }

        fn run_instr<'a>(context: &'a mut Context, op: &BlockOp) -> Res {
            println!("run {}", op);
            match op {
                &BlockOp::Block(Block::Block(ref ops)) => match run_block(context, ops) {
                    Res::Branch(0, val) => Res::Value(val),
                    Res::Branch(n, val) => Res::Branch(n - 1, val),
                    x => x
                },
                &BlockOp::Block(Block::Loop(ref ops)) => {
                    loop {
                        match run_block(context, ops) {
                            val@ Res::Return(_) => return val,
                            val@ Res::Value(_) => return val,
                            Res::Branch(0, _) => continue,
                            Res::Branch(1, val) => return Res::Value(val),
                            Res::Branch(x, val) => return Res::Branch(x - 2, val),
                        }
                    }
                }
                &BlockOp::Block(Block::If(ref then, ref otherwise)) => {
                    let cond = context.stack.pop().unwrap();
                    match run_block(context, if cond.to_u32() != 0 { then } else { otherwise }) {
                        Res::Branch(0, val) => Res::Value(val),
                        Res::Branch(n, val) => Res::Branch(n - 1, val),
                        x => x
                    }
                }
                &BlockOp::Normal(ref op) => match op {
                    &NormalOp::Nop => Res::Value(None),
                    &NormalOp::Select => {
                        let b = context.stack.pop().unwrap();
                        let a = context.stack.pop().unwrap();
                        let cond = context.stack.pop().unwrap();
                        Res::Value(Some(if cond.to_u32() != 0 { a } else { b }))
                    },
                    &NormalOp::Br{has_arg, relative_depth} => {
                        let val = if has_arg {
                            Some(context.stack.pop().unwrap())
                        } else {
                            None
                        };
                        Res::Branch(relative_depth, val)
                    }
                    &NormalOp::BrIf{has_arg, relative_depth} => {
                        let val = if has_arg {
                            Some(context.stack.pop().unwrap())
                        } else {
                            None
                        };
                        let cond = context.stack.pop().unwrap();
                        if cond.to_u32() != 0 {
                            Res::Branch(relative_depth, val)
                        } else {
                            Res::Value(None)
                        }
                    }
                    &NormalOp::BrTable{has_arg, target_data, default} => {
                        let val = if has_arg {
                            Some(context.stack.pop().unwrap())
                        } else {
                            None
                        };
                        let value = context.stack.pop().unwrap().to_u32() as usize;
                        let relative_depth = if value >= target_data.len() / 4 {
                            default
                        } else {
                            read_u32(&target_data[value * 4.. value * 4 + 4])
                        };
                        Res::Branch(relative_depth, val)
                    }
                    &NormalOp::Return{has_arg} => {
                        if has_arg {
                            Res::Return(Some(context.stack.pop().unwrap()))
                        } else {
                            Res::Return(None)
                        }
                    }
                    &NormalOp::Unreachable => {
                        panic!()
                    }
                    &NormalOp::Drop => {
                        context.stack.pop().unwrap();
                        Res::Value(None)
                    }
                    &NormalOp::Const(val) => {
                        Res::Value(Some(val))
                    }
                    &NormalOp::GetLocal(local) => {
                        Res::Value(Some(context.locals[local as usize]))
                    }
                    &NormalOp::SetLocal(local) => {
                        let val = context.stack.pop().unwrap();
                        context.locals[local as usize] = val;
                        Res::Value(Some(val)) // TODO: this should be None.
                    }
                    &NormalOp::TeeLocal(local) => {
                        let val = context.stack.pop().unwrap();
                        context.locals[local as usize] = val;
                        Res::Value(Some(val))
                    }
                    &NormalOp::Call{argument_count, index} => {
                        let stack_len = context.stack.len();
                        let res = {
                            let args = &context.stack[stack_len - argument_count as usize..];
                            Res::Value(context.instance.invoke(index, &args))
                        };
                        context.stack.drain(stack_len - argument_count as usize..);
                        res
                    }
                    &NormalOp::CallIndirect{argument_count, index: type_index} => {
                        let table_index = context.stack.pop().unwrap().to_u32();

                        let index = context.instance.module.table[table_index as usize];

                        let stack_len = context.stack.len();
                        let res = {
                            let args = &context.stack[stack_len - argument_count as usize..];
                            Res::Value(context.instance.invoke(index, &args))
                        };
                        context.stack.drain(stack_len - argument_count as usize..);
                        res
                    }
                    &NormalOp::CallImport{argument_count, index} => {
                        panic!();
                    }
                    &NormalOp::IntLoad(inttype, sign, size, memimm) => {
                        let addr = context.stack.pop().unwrap();
                        Res::Value(Some(context.instance.memory.load_int(addr.to_u32(), inttype, sign, size, memimm)))
                    }
                    &NormalOp::FloatLoad(floattype, memimm) => {
                        let addr = context.stack.pop().unwrap();
                        Res::Value(Some(context.instance.memory.load_float(addr.to_u32(), floattype, memimm)))
                    }
                    &NormalOp::IntStore(inttype, size, memimm) => {
                        let addr = context.stack.pop().unwrap();
                        let value = context.stack.pop().unwrap();
                        assert!(value.get_type() == inttype.to_type());
                        context.instance.memory.store_int(addr.to_u32(), value, size, memimm);
                        Res::Value(Some(value))
                    }
                    &NormalOp::FloatStore(floattype, memimm) => {
                        let addr = context.stack.pop().unwrap();
                        let value = context.stack.pop().unwrap();
                        context.instance.memory.store_float(addr.to_u32(), value, floattype, memimm);
                        Res::Value(Some(value))
                    }

                    &NormalOp::CurrentMemory => {
                        panic!();
                    }
                    &NormalOp::GrowMemory => {
                        panic!();
                    }

                    &NormalOp::IntBin(inttype, intbinop) => {
                        let b = context.stack.pop().unwrap();
                        let a = context.stack.pop().unwrap();
                        Res::Value(Some(interp_int_bin(inttype, intbinop, a, b)))
                    }
                    &NormalOp::IntCmp(inttype, intcmpop) => {
                        let b = context.stack.pop().unwrap();
                        let a = context.stack.pop().unwrap();
                        Res::Value(Some(interp_int_cmp(inttype, intcmpop, a, b)))
                    }
                    &NormalOp::IntUn(inttype, intunop) => {
                        let a = context.stack.pop().unwrap();
                        Res::Value(Some(interp_int_un(inttype, intunop, a)))
                    }
                    &NormalOp::IntEqz(inttype) => {
                        let a = context.stack.pop().unwrap();
                        Res::Value(Some(interp_int_eqz(inttype, a)))
                    }
                    &NormalOp::FloatBin(floattype, floatbinop) => {
                        let b = context.stack.pop().unwrap();
                        let a = context.stack.pop().unwrap();
                        Res::Value(Some(interp_float_bin(floattype, floatbinop, a, b)))
                    }
                    &NormalOp::FloatUn(floattype, floatunop) => {
                        let a = context.stack.pop().unwrap();
                        Res::Value(Some(interp_float_un(floattype, floatunop, a)))
                    }
                    &NormalOp::FloatCmp(floattype, floatcmpop) => {
                        let b = context.stack.pop().unwrap();
                        let a = context.stack.pop().unwrap();
                        Res::Value(Some(interp_float_cmp(floattype, floatcmpop, a, b)))
                    }
                    &NormalOp::FloatToInt(floattype, inttype, sign) => {
                        let a = context.stack.pop().unwrap();
                        Res::Value(Some(interp_float_to_int(floattype, inttype, sign, a)))
                    }
                    &NormalOp::IntExtend(sign) => {
                        let a = context.stack.pop().unwrap();
                        Res::Value(Some(interp_int_extend(sign, a)))
                    }
                    &NormalOp::IntTruncate => {
                        let a = context.stack.pop().unwrap();
                        Res::Value(Some(interp_int_truncate(a)))
                    }
                    &NormalOp::IntToFloat(inttype, sign, floattype) => {
                        let a = context.stack.pop().unwrap();
                        Res::Value(Some(interp_int_to_float(inttype, sign, floattype, a)))
                    }
                    &NormalOp::FloatConvert(floattype) => {
                        let a = context.stack.pop().unwrap();
                        Res::Value(Some(interp_float_convert(floattype, a)))
                    }
                    &NormalOp::Reinterpret(type_from, type_to) => {
                        let a = context.stack.pop().unwrap();
                        Res::Value(Some(interp_reinterpret(type_from, type_to, a)))
                    }
                }
            }
        };

        let mut context = Context {
            instance: self,
            locals: locals,
            stack: Vec::new(),
        };

        let res = match run_block(&mut context, &root_ops) {
            Res::Value(v) | Res::Return(v) => v,
            _ => panic!()
        };

        res
    }
}

fn i64_from_u64(val: Wrapping<u64>) -> Wrapping<i64> {
    unsafe { mem::transmute(val) }
}

fn i64_from_i32(val: Wrapping<u64>) -> Wrapping<i64> {
    let v: i32 = unsafe { mem::transmute(val.0 as u32) };
    Wrapping(v as i64)
}

fn u64_from_real_i32(val: i32) -> Wrapping<u64> {
    let v: i32 = unsafe { mem::transmute(val as u32) };
    u64_from_i64(Wrapping(v as i64))
}

fn u64_from_i64(val: Wrapping<i64>) -> Wrapping<u64> {
    unsafe { mem::transmute(val) }
}

fn interp_int_bin(ty: IntType, op: IntBinOp, a: Dynamic, b: Dynamic) -> Dynamic {
    assert!(a.get_type() == ty.to_type());
    assert!(b.get_type() == ty.to_type());

    let a = a.to_int();
    let b = b.to_int();

    let res = match op {
        IntBinOp::Add => a + b,
        IntBinOp::Sub => a - b,
        IntBinOp::Mul => a * b,
        IntBinOp::DivS => {
            match ty {
                IntType::Int32 => u64_from_i64(i64_from_i32(a) / i64_from_i32(b)),
                IntType::Int64 => u64_from_i64(i64_from_u64(a) / i64_from_u64(b)),
            }
        }
        IntBinOp::DivU => a / b,
        IntBinOp::RemS => {
            match ty {
                IntType::Int32 => u64_from_i64(i64_from_i32(a) % i64_from_i32(b)),
                IntType::Int64 => u64_from_i64(i64_from_u64(a) % i64_from_u64(b)),
            }
        }
        IntBinOp::RemU => a % b,
        IntBinOp::And => a & b,
        IntBinOp::Or => a | b,
        IntBinOp::Xor => a ^ b,
        IntBinOp::Shl => a << b.0 as usize,
        IntBinOp::ShrU => a >> b.0 as usize,
        IntBinOp::ShrS => {
            match ty {
                IntType::Int32 => u64_from_i64(i64_from_i32(a) >> b.0 as usize),
                IntType::Int64 => u64_from_i64(i64_from_u64(a) >> b.0 as usize),
            }
        }
        IntBinOp::Rotr => {
            Wrapping(match ty {
                IntType::Int32 => (a.0 as u32).rotate_right(b.0 as u32) as u64,
                IntType::Int64 => a.0.rotate_right(b.0 as u32),
            })
        }
        IntBinOp::Rotl => {
            Wrapping(match ty {
                IntType::Int32 => (a.0 as u32).rotate_left(b.0 as u32) as u64,
                IntType::Int64 => a.0.rotate_left(b.0 as u32),
            })
        }
    };

    match ty {
        IntType::Int32 => Dynamic::from_u32(res.0 as u32),
        IntType::Int64 => Dynamic::from_u64(res.0),
    }
}

fn extend_signed(val: Wrapping<u64>, ty: IntType) -> Wrapping<i64> {
    match ty {
        IntType::Int32 => i64_from_i32(val),
        IntType::Int64 => i64_from_u64(val),
    }
}

fn interp_int_cmp(ty: IntType, op: IntCmpOp, a: Dynamic, b: Dynamic) -> Dynamic {
    assert!(a.get_type() == ty.to_type());
    assert!(b.get_type() == ty.to_type());

    let a = a.to_int();
    let b = b.to_int();

    Dynamic::from_u32(if match op {
        IntCmpOp::Eq => a == b,
        IntCmpOp::Ne => a != b,
        IntCmpOp::LtS => extend_signed(a, ty) < extend_signed(b, ty),
        IntCmpOp::LeS => extend_signed(a, ty) <= extend_signed(b, ty),
        IntCmpOp::LtU => a < b,
        IntCmpOp::LeU => a <= b,
        IntCmpOp::GtS => extend_signed(a, ty) > extend_signed(b, ty),
        IntCmpOp::GeS => extend_signed(a, ty) >= extend_signed(b, ty),
        IntCmpOp::GtU => a > b,
        IntCmpOp::GeU => a >= b,
    } { 1 } else { 0 })
}

fn interp_int_un(ty: IntType, op: IntUnOp, a: Dynamic) -> Dynamic {
    let res = match op {
        IntUnOp::Clz => match ty {
            IntType::Int32 => a.to_u32().leading_zeros(),
            IntType::Int64 => a.to_u64().leading_zeros(),
        },
        IntUnOp::Ctz => match ty {
            IntType::Int32 => a.to_u32().trailing_zeros(),
            IntType::Int64 => a.to_u64().trailing_zeros(),
        },
        IntUnOp::Popcnt => match ty {
            IntType::Int32 => a.to_u32().count_ones(),
            IntType::Int64 => a.to_u64().count_ones(),
        },
    };

    Dynamic::from_u32(res)
}

fn interp_int_eqz(ty: IntType, a: Dynamic) -> Dynamic {
    assert!(a.get_type() == ty.to_type());

    let a = a.to_int();

    Dynamic::from_u32(if a.0 == 0 { 1 } else { 0 })
}

fn interp_float_bin(ty: FloatType, op: FloatBinOp, a: Dynamic, b: Dynamic) -> Dynamic {
    assert!(a.get_type() == ty.to_type());
    assert!(b.get_type() == ty.to_type());

    let a = a.to_float();
    let b = b.to_float();

    let res = match op {
        FloatBinOp::Add => a + b,
        FloatBinOp::Sub => a - b,
        FloatBinOp::Mul => a * b,
        FloatBinOp::Div => a / b,
        FloatBinOp::Min => a.min(b),
        FloatBinOp::Max => a.max(b),
    };

    Dynamic::from_float(ty, res)
}

fn interp_float_un(ty: FloatType, op: FloatUnOp, a: Dynamic) -> Dynamic {
    assert!(a.get_type() == ty.to_type());

    let a = a.to_float();

    let res = match op {
        FloatUnOp::Abs => a.abs(),
        FloatUnOp::Neg => -a,
        FloatUnOp::Copysign => a.signum(),
        FloatUnOp::Ceil => a.ceil(),
        FloatUnOp::Floor => a.floor(),
        FloatUnOp::Trunc => a.trunc(),
        FloatUnOp::Nearest => a.round(),
        FloatUnOp::Sqrt => a.sqrt(),
    };

    Dynamic::from_float(ty, res)
}

fn interp_float_cmp(ty: FloatType, op: FloatCmpOp, a: Dynamic, b: Dynamic) -> Dynamic {
    assert!(a.get_type() == ty.to_type());
    assert!(b.get_type() == ty.to_type());

    let a = a.to_float();
    let b = b.to_float();

    Dynamic::from_u32(if match op {
        FloatCmpOp::Eq => a == b,
        FloatCmpOp::Ne => a != b,
        FloatCmpOp::Lt => a < b,
        FloatCmpOp::Le => a <= b,
        FloatCmpOp::Gt => a > b,
        FloatCmpOp::Ge => a >= b,
    } { 1 } else { 0 })
}

fn interp_float_to_int(floattype: FloatType, inttype: IntType, sign: Sign, a: Dynamic) -> Dynamic {
    assert!(a.get_type() == floattype.to_type());

    let a = a.to_float();

    Dynamic::from_int(inttype, match (sign, inttype) {
        (Sign::Signed, IntType::Int32) => u64_from_real_i32(a as i32),
        (Sign::Unsigned, IntType::Int32) => Wrapping(a as u64),
        (Sign::Signed, IntType::Int64) => u64_from_i64(Wrapping(a as i64)),
        (Sign::Unsigned, IntType::Int64) => Wrapping(a as u64),
    }.0)
}

fn interp_int_extend(sign: Sign, a: Dynamic) -> Dynamic {
    assert!(a.get_type() == Type::Int32);

    Dynamic::Int64(match sign {
        Sign::Signed => u64_from_i64(Wrapping(a.to_i32() as i64)),
        Sign::Unsigned => a.to_wu64()
    })
}

fn interp_int_truncate(a: Dynamic) -> Dynamic {
    Dynamic::from_u32(a.to_u64() as u32)
}

fn interp_int_to_float(inttype: IntType, sign: Sign, floattype: FloatType, a: Dynamic) -> Dynamic {
    assert!(a.get_type() == inttype.to_type());

    let a = a.to_int();

    Dynamic::from_float(floattype, match (sign, inttype) {
        (Sign::Signed, IntType::Int32) => i64_from_i32(a).0 as f64,
        (Sign::Unsigned, IntType::Int32) => a.0 as f64,
        (Sign::Signed, IntType::Int64) => i64_from_u64(a).0 as f64,
        (Sign::Unsigned, IntType::Int64) => a.0 as f64,
    })
}

fn interp_float_convert(ty: FloatType, a: Dynamic) -> Dynamic {
    Dynamic::from_float(ty, a.to_float())
}

fn interp_reinterpret(type_from: Type, type_to: Type, a: Dynamic) -> Dynamic {
    assert!(type_from == a.get_type());

    let res = match a {
        Dynamic::Int32(v) => v.0 as u64,
        Dynamic::Int64(v) => v.0,
        Dynamic::Float32(v) => unsafe { mem::transmute(v as f64) },
        Dynamic::Float64(v) => unsafe { mem::transmute(v) },
    };

    match type_to {
        Type::Int32 => Dynamic::from_u32(res as u32),
        Type::Int64 => Dynamic::from_u64(res),
        Type::Float32 => Dynamic::Float32(unsafe { mem::transmute((res & 0xffffffff) as u32) }),
        Type::Float64 => Dynamic::Float64(unsafe { mem::transmute(res) })
    }
}

fn extend_u8(val: u8, inttype: IntType, sign: Sign) -> Dynamic {
    match (sign, inttype) {
        (Sign::Signed, IntType::Int32) => Dynamic::from_i32(val as i8 as i32),
        (Sign::Unsigned, IntType::Int32) => Dynamic::from_u32(val as u32),
        (Sign::Signed, IntType::Int64) => Dynamic::from_i64(val as i8 as i64),
        (Sign::Unsigned, IntType::Int64) => Dynamic::from_u64(val as u64),
    }
}

fn extend_u16(val: u16, inttype: IntType, sign: Sign) -> Dynamic {
    match (sign, inttype) {
        (Sign::Signed, IntType::Int32) => Dynamic::from_i32(val as i16 as i32),
        (Sign::Unsigned, IntType::Int32) => Dynamic::from_u32(val as u32),
        (Sign::Signed, IntType::Int64) => Dynamic::from_i64(val as i16 as i64),
        (Sign::Unsigned, IntType::Int64) => Dynamic::from_u64(val as u64),
    }
}

fn extend_u32(val: u32, inttype: IntType, sign: Sign) -> Dynamic {
    match (sign, inttype) {
        (Sign::Signed, IntType::Int32) => Dynamic::from_u32(val),
        (Sign::Unsigned, IntType::Int32) => Dynamic::from_u32(val),
        (Sign::Signed, IntType::Int64) => Dynamic::from_i64(val as i32 as i64),
        (Sign::Unsigned, IntType::Int64) => Dynamic::from_u64(val as u64),
    }
}

fn extend_u64(val: u64, inttype: IntType, sign: Sign) -> Dynamic {
    match (sign, inttype) {
        (Sign::Signed, IntType::Int32) => Dynamic::from_u32(val as u32),
        (Sign::Unsigned, IntType::Int32) => Dynamic::from_u32(val as u32),
        (Sign::Signed, IntType::Int64) => Dynamic::from_u64(val),
        (Sign::Unsigned, IntType::Int64) => Dynamic::from_u64(val),
    }
}
