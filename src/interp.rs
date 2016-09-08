use std::{mem, str, iter};
use std::num::Wrapping;
use std::collections::HashMap;

use module::{Module, FunctionIndex, ImportIndex, ExportIndex, AsBytes, FunctionType};
use types::{Type, Dynamic, Sign, Size, IntType, FloatType};
use ops::{
    BlockOp, Block, NormalOp, MemImm,
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

#[test]
fn test_store_load() {
    let mut m = Memory(iter::repeat(0).take(1024).collect::<Vec<_>>());

    for i in 0..10 {
        m.store_u32(i*4, i as u32);
        assert_eq!(m.load_u32(i*4), i as u32);
    }
    for i in 0..10 {
        m.store_u64(i*4, i as u64);
        assert_eq!(m.load_u64(i*4), i as u64);
    }
}

pub trait BoundInstance {
    fn invoke_export(&mut self, func: ExportIndex, args: &[Dynamic]) -> InterpResult;
    fn export_by_name_and_type(&self, name: &[u8], ty: FunctionType<&[u8]>) -> ExportIndex;
}

pub struct Instance<'a, B: AsBytes + 'a> {
    pub memory: Memory,
    pub module: &'a Module<B>,
    pub call_stack_depth: usize,
    pub bound_imports: Vec<(usize, ExportIndex)>,
    pub bound_instances: Vec<Box<BoundInstance>>,
}

fn read_u32(data: &[u8]) -> u32 {
    ((data[0] as u32) << 0*8) +
    ((data[1] as u32) << 1*8) +
    ((data[2] as u32) << 2*8) +
    ((data[3] as u32) << 3*8)
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum InterpResult {
    Value(Option<Dynamic>),
    Trap,
}

impl<'a, B: AsBytes> BoundInstance for Instance<'a, B> {
    fn invoke_export(&mut self, func: ExportIndex, args: &[Dynamic]) -> InterpResult {
        self.invoke(self.module.exports[func.0].function_index, args)
    }
    fn export_by_name_and_type(&self, name: &[u8], ty: FunctionType<&[u8]>) -> ExportIndex {
        self.module.find_export(name, ty).unwrap()
    }
}

impl<'a, B: AsBytes> Instance<'a, B> {
    pub fn new(module: &'a Module<B>, imports: HashMap<&[u8], Box<BoundInstance>>) -> Instance<'a, B> {
        let mut memory = Vec::with_capacity(module.memory_info.initial_64k_pages * 64 * 1024);
        memory.resize(module.memory_info.initial_64k_pages * 64 * 1024, 0);

        for m in &module.memory_chunks {
            let data = m.data.as_bytes();
            let newlen = ::std::cmp::max(m.offset + data.len(), memory.len());
            memory.resize(newlen, 0);
            memory[m.offset..m.offset + data.len()].copy_from_slice(data);
        }

        let mut bound_instances = Vec::new();

        let mut instance_indices = HashMap::new();

        for (k, v) in imports {
            instance_indices.insert(k, bound_instances.len());
            bound_instances.push(v);
        }

        let mut bound_imports = module.imports.iter().map(|i| {
            let instance_index = *instance_indices.get(i.module_name.as_bytes()).unwrap();
            let export_index = bound_instances[instance_index]
                .export_by_name_and_type(i.function_name.as_bytes(), module.types[i.function_type.0].as_ref());
            (instance_index, export_index)
        }).collect::<Vec<_>>();

        Instance {
            memory: Memory(memory),
            module: module,
            call_stack_depth: 0,
            bound_imports: bound_imports,
            bound_instances: bound_instances,
        }
    }

    pub fn invoke(&mut self, func: FunctionIndex, args: &[Dynamic]) -> InterpResult {
        println!("running {}",
            self.module.find_name(func)
                .and_then(|n| str::from_utf8(n).ok())
                .unwrap_or("<unknown>"));

        if self.call_stack_depth > 200 {
            return InterpResult::Trap;
        }

        self.call_stack_depth += 1;

        let ty = &self.module.types[self.module.functions[func.0].0];
        if args.len() != ty.param_types.as_bytes().len() {
            panic!("expected {} args, but got {}", ty.param_types.as_bytes().len(), args.len());
        }
        let f = &self.module.code[func.0];

        let root_ops = f.block_ops().collect::<Vec<_>>();

        let local_count: usize = f.locals.iter().map(|e|e.1).fold(0, |a, b|a+b);
        let mut locals = Vec::with_capacity(args.len() + local_count);
        locals.extend(args);
        for &(ty, count) in &f.locals {
            for _ in 0..count {
                locals.push(ty.zero());
            }
        }

        println!("locals: {}", locals.len());

        struct Context<'b, 'a: 'b, B: AsBytes + 'a> {
            instance: &'b mut Instance<'a, B>,
            locals: Vec<Dynamic>,
            stack: Vec<Option<Dynamic>>,
        }

        #[derive(Debug)]
        enum Res {
            Value(Option<Dynamic>),
            Branch(u32, Option<Dynamic>),
            Return(Option<Dynamic>),
            Trap,
        }

        fn run_block<'a, B: AsBytes>(context: &'a mut Context<B>, ops: &[BlockOp]) -> Res {
            if ops.len() > 0 {
                for i in &ops[..ops.len() - 1] {
                    match run_instr(context, i) {
                        Res::Value(v) => context.stack.push(v),
                        val => return val
                    }
                }
                run_instr(context, &ops[ops.len() - 1])
            } else {
                Res::Value(None)
            }
        }

        fn run_instr<'a, B: AsBytes>(context: &'a mut Context<B>, op: &BlockOp) -> Res {
            println!("run {}", op);
            let res = match op {
                &BlockOp::Block(Block::Block(ref ops)) => {
                    let stack_depth = context.stack.len();
                    let res = match run_block(context, ops) {
                        Res::Branch(0, val) => Res::Value(val),
                        Res::Branch(n, val) => Res::Branch(n - 1, val),
                        x => x
                    };
                    context.stack.resize(stack_depth, None);
                    res
                }
                &BlockOp::Block(Block::Loop(ref ops)) => {
                    loop {
                        let stack_depth = context.stack.len();
                        match run_block(context, ops) {
                            val@ Res::Return(_) => {
                                context.stack.resize(stack_depth, None);
                                return val
                            }
                            val@ Res::Trap => {
                                context.stack.resize(stack_depth, None);
                                return val
                            }
                            val@ Res::Value(_) => {
                                context.stack.resize(stack_depth, None);
                                return val
                            }
                            Res::Branch(0, _) => continue,
                            Res::Branch(1, val) => {
                                context.stack.resize(stack_depth, None);
                                return Res::Value(val)
                            }
                            Res::Branch(x, val) => {
                                context.stack.resize(stack_depth, None);
                                return Res::Branch(x - 2, val)
                            }
                        }
                        context.stack.resize(stack_depth, None);
                    }
                }
                &BlockOp::Block(Block::If(ref then, ref otherwise)) => {
                    let cond = context.stack.pop().unwrap().unwrap();
                    let stack_depth = context.stack.len();
                    let res = match run_block(context, if cond.to_u32() != 0 { then } else { otherwise }) {
                        Res::Branch(0, val) => Res::Value(val),
                        Res::Branch(n, val) => Res::Branch(n - 1, val),
                        x => x
                    };
                    context.stack.resize(stack_depth, None);
                    res
                }
                &BlockOp::Normal(ref op) => match op {
                    &NormalOp::Nop => Res::Value(None),
                    &NormalOp::Select => {
                        let cond = context.stack.pop().unwrap().unwrap();
                        let b = context.stack.pop().unwrap();
                        let a = context.stack.pop().unwrap();
                        Res::Value(if cond.to_u32() != 0 { a } else { b })
                    },
                    &NormalOp::Br{has_arg, relative_depth} => {
                        let val = if has_arg {
                            context.stack.pop().unwrap()
                        } else {
                            None
                        };
                        Res::Branch(relative_depth, val)
                    }
                    &NormalOp::BrIf{has_arg, relative_depth} => {
                        let cond = context.stack.pop().unwrap().unwrap();
                        let val = if has_arg {
                            context.stack.pop().unwrap()
                        } else {
                            None
                        };
                        if cond.to_u32() != 0 {
                            Res::Branch(relative_depth, val)
                        } else {
                            Res::Value(None)
                        }
                    }
                    &NormalOp::BrTable{has_arg, target_data, default} => {
                        let value = context.stack.pop().unwrap().unwrap().to_u32() as usize;
                        let val = if has_arg {
                            context.stack.pop().unwrap()
                        } else {
                            None
                        };
                        let relative_depth = if value >= target_data.len() / 4 {
                            default
                        } else {
                            read_u32(&target_data[value * 4.. value * 4 + 4])
                        };
                        Res::Branch(relative_depth, val)
                    }
                    &NormalOp::Return{has_arg} => {
                        if has_arg {
                            Res::Return(context.stack.pop().unwrap())
                        } else {
                            Res::Return(None)
                        }
                    }
                    &NormalOp::Unreachable => {
                        Res::Trap
                    }
                    &NormalOp::Drop => {
                        context.stack.pop().unwrap();
                        Res::Value(None)
                    }
                    &NormalOp::Const(val) => {
                        Res::Value(Some(val))
                    }
                    &NormalOp::GetLocal(local) => {
                        let val = context.locals[local as usize];
                        println!("val {}", val);
                        Res::Value(Some(val))
                    }
                    &NormalOp::SetLocal(local) => {
                        let val = context.stack.pop().unwrap().unwrap();
                        context.locals[local as usize] = val;
                        Res::Value(Some(val)) // TODO: this should be None.
                    }
                    &NormalOp::TeeLocal(local) => {
                        let val = context.stack.pop().unwrap().unwrap();
                        context.locals[local as usize] = val;
                        Res::Value(Some(val))
                    }
                    &NormalOp::Call{argument_count, index} => {
                        let stack_len = context.stack.len();
                        let res = {
                            let args = context.stack[stack_len - argument_count as usize..]
                                .iter().map(|e| e.unwrap()).collect::<Vec<_>>();
                            match context.instance.invoke(index, &args) {
                                InterpResult::Value(v) => Res::Value(v),
                                InterpResult::Trap => return Res::Trap,
                            }
                        };
                        context.stack.drain(stack_len - argument_count as usize..);
                        res
                    }
                    &NormalOp::CallIndirect{argument_count, index: type_index} => {
                        let table_index = context.stack[context.stack.len() - 1 - argument_count as usize].unwrap().to_u32();

                        let ti = table_index as usize;
                        if ti >= context.instance.module.table.len() {
                            Res::Trap
                        } else {
                            let index = context.instance.module.table[ti];

                            println!("index {} a {} b {}", index.0, context.instance.module.functions[index.0].0,  type_index.0);

                            if context.instance.module.functions[index.0] == type_index {
                                let stack_len = context.stack.len();
                                let res = {
                                    let args = context.stack[stack_len - argument_count as usize..]
                                        .iter().map(|e| e.unwrap()).collect::<Vec<_>>();
                                    match context.instance.invoke(index, &args) {
                                        InterpResult::Value(v) => Res::Value(v),
                                        InterpResult::Trap => return Res::Trap,
                                    }
                                };
                                context.stack.drain(stack_len - argument_count as usize - 1..);
                                res
                            } else {
                                Res::Trap
                            }
                        }
                    }
                    &NormalOp::CallImport{argument_count, index} => {
                        let stack_len = context.stack.len();
                        let res = {
                            let args = context.stack[stack_len - argument_count as usize..]
                                .iter().map(|e| e.unwrap()).collect::<Vec<_>>();

                            println!("import {} of {}", index.0, context.instance.bound_imports.len());

                            let (module, index) = context.instance.bound_imports[index.0];
                            println!("module {} index {}", module, index.0);
                            match context.instance.bound_instances[module].invoke_export(index, args.as_slice()) {
                                InterpResult::Value(v) => Res::Value(v),
                                InterpResult::Trap => return Res::Trap,
                            }
                        };
                        context.stack.drain(stack_len - argument_count as usize..);
                        res
                    }
                    &NormalOp::IntLoad(ty, sign, size, memimm) => {
                        let addr = context.stack.pop().unwrap().unwrap().to_u32();
                        if addr as usize + size.to_int()/8 <= context.instance.memory.0.len() {
                            Res::Value(Some(context.instance.memory.load_int(addr, ty, sign, size, memimm)))
                        } else {
                            Res::Trap
                        }
                    }
                    &NormalOp::FloatLoad(ty, memimm) => {
                        let addr = context.stack.pop().unwrap().unwrap().to_u32();
                        if addr as usize + ty.to_type().size().to_int()/8 <= context.instance.memory.0.len() {
                            Res::Value(Some(context.instance.memory.load_float(addr, ty, memimm)))
                        } else {
                            Res::Trap
                        }
                    }
                    &NormalOp::IntStore(ty, size, memimm) => {
                        let value = context.stack.pop().unwrap().unwrap();
                        let addr = context.stack.pop().unwrap().unwrap().to_u32();
                        if addr as usize + size.to_int()/8 <= context.instance.memory.0.len() {
                            assert!(value.get_type() == ty.to_type());
                            context.instance.memory.store_int(addr, value, size, memimm);
                            Res::Value(Some(value))
                        } else {
                            Res::Trap
                        }
                    }
                    &NormalOp::FloatStore(ty, memimm) => {
                        let value = context.stack.pop().unwrap().unwrap();
                        let addr = context.stack.pop().unwrap().unwrap().to_u32();
                        if addr as usize + ty.to_type().size().to_int()/8 <= context.instance.memory.0.len() {
                            context.instance.memory.store_float(addr, value, ty, memimm);
                            Res::Value(Some(value))
                        } else {
                            Res::Trap
                        }
                    }

                    &NormalOp::CurrentMemory => {
                        Res::Value(Some(Dynamic::from_u32(context.instance.memory.0.len() as u32 / 0x10000)))
                    }
                    &NormalOp::GrowMemory => {
                        let len = context.instance.memory.0.len();
                        let extra_pages = context.stack.pop().unwrap().unwrap().to_u32() as usize;
                        let new_len = len + extra_pages * 0x10000;
                        if new_len < 0x8000_0000 {
                            context.instance.memory.0.resize(new_len, 0);
                            Res::Value(Some(Dynamic::from_u32(len as u32 / 0x10000)))
                        } else {
                            Res::Trap
                        }
                    }

                    &NormalOp::IntBin(inttype, intbinop) => {
                        let b = context.stack.pop().unwrap().unwrap();
                        let a = context.stack.pop().unwrap().unwrap();
                        match interp_int_bin(inttype, intbinop, a, b) {
                            InterpResult::Value(v) => Res::Value(v),
                            InterpResult::Trap => Res::Trap,
                        }
                    }
                    &NormalOp::IntCmp(inttype, intcmpop) => {
                        let b = context.stack.pop().unwrap().unwrap();
                        let a = context.stack.pop().unwrap().unwrap();
                        Res::Value(Some(interp_int_cmp(inttype, intcmpop, a, b)))
                    }
                    &NormalOp::IntUn(inttype, intunop) => {
                        let a = context.stack.pop().unwrap().unwrap();
                        Res::Value(Some(interp_int_un(inttype, intunop, a)))
                    }
                    &NormalOp::IntEqz(inttype) => {
                        let a = context.stack.pop().unwrap().unwrap();
                        Res::Value(Some(interp_int_eqz(inttype, a)))
                    }
                    &NormalOp::FloatBin(floattype, floatbinop) => {
                        let b = context.stack.pop().unwrap().unwrap();
                        let a = context.stack.pop().unwrap().unwrap();
                        Res::Value(Some(interp_float_bin(floattype, floatbinop, a, b)))
                    }
                    &NormalOp::FloatUn(floattype, floatunop) => {
                        let a = context.stack.pop().unwrap().unwrap();
                        Res::Value(Some(interp_float_un(floattype, floatunop, a)))
                    }
                    &NormalOp::FloatCmp(floattype, floatcmpop) => {
                        let b = context.stack.pop().unwrap().unwrap();
                        let a = context.stack.pop().unwrap().unwrap();
                        Res::Value(Some(interp_float_cmp(floattype, floatcmpop, a, b)))
                    }
                    &NormalOp::FloatToInt(floattype, inttype, sign) => {
                        let a = context.stack.pop().unwrap().unwrap();
                        match interp_float_to_int(floattype, inttype, sign, a) {
                            None => Res::Trap,
                            Some(v) => Res::Value(Some(v))
                        }
                    }
                    &NormalOp::IntExtend(sign) => {
                        let a = context.stack.pop().unwrap().unwrap();
                        Res::Value(Some(interp_int_extend(sign, a)))
                    }
                    &NormalOp::IntTruncate => {
                        let a = context.stack.pop().unwrap().unwrap();
                        Res::Value(Some(interp_int_truncate(a)))
                    }
                    &NormalOp::IntToFloat(inttype, sign, floattype) => {
                        let a = context.stack.pop().unwrap().unwrap();
                        Res::Value(Some(interp_int_to_float(inttype, sign, floattype, a)))
                    }
                    &NormalOp::FloatConvert(floattype) => {
                        let a = context.stack.pop().unwrap().unwrap();
                        Res::Value(Some(interp_float_convert(floattype, a)))
                    }
                    &NormalOp::Reinterpret(type_from, type_to) => {
                        let a = context.stack.pop().unwrap().unwrap();
                        Res::Value(Some(interp_reinterpret(type_from, type_to, a)))
                    }
                }
            };
            println!("res {} -> {:?}", op, res);
            res
        };

        fn verify_return_type(ty: Option<Type>, v: Option<Dynamic>) -> InterpResult {
            match ty {
                Some(ty) => {
                    assert_eq!(ty, v.unwrap().get_type());
                    InterpResult::Value(v)
                }
                None => {
                    // TODO: we really ought to be asserting that v is None here... but
                    // we're having some problems with Dropping values (see block.wast: drop-last)
                    InterpResult::Value(None)
                }
            }
        }

        let res = {
            let mut context = Context {
                instance: self,
                locals: locals,
                stack: Vec::new(),
            };
            match run_block(&mut context, &root_ops) {
                Res::Value(v) | Res::Return(v) => verify_return_type(ty.return_type, v),
                Res::Trap => InterpResult::Trap,
                Res::Branch(0, v) => verify_return_type(ty.return_type, v),
                _ => panic!()
            }
        };

        self.call_stack_depth -= 1;

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


fn copysign_f32(a: f32, b: f32) -> f32 {
    if (unsafe { mem::transmute::<f32, u32>(b) } & 0x8000_0000u32) == 0 {
        a.abs()
    } else {
        (-a.abs())
    }
}
fn copysign_f64(a: f64, b: f64) -> f64 {
    if (unsafe { mem::transmute::<f64, u64>(b) } & 0x8000_0000_0000_0000u64) == 0 {
        a.abs()
    } else {
        -a.abs()
    }
}

fn interp_int_bin(ty: IntType, op: IntBinOp, a: Dynamic, b: Dynamic) -> InterpResult {
    assert_eq!(a.get_type(), ty.to_type());
    assert_eq!(b.get_type(), ty.to_type());

    let a = a.to_int();
    let b = b.to_int();

    let res = match op {
        IntBinOp::Add => a + b,
        IntBinOp::Sub => a - b,
        IntBinOp::Mul => a * b,
        IntBinOp::DivS => {
            match ty {
                IntType::Int32 => {
                    let a = i64_from_i32(a);
                    let b = i64_from_i32(b);
                    if b.0 == 0 || (b.0 == -1 && a.0 == i32::min_value() as i64) {
                        return InterpResult::Trap;
                    } else {
                        u64_from_i64(a / b)
                    }
                },
                IntType::Int64 => {
                    let a = i64_from_u64(a);
                    let b = i64_from_u64(b);
                    if b.0 == 0 || (b.0 == -1 && a.0 == i64::min_value()) {
                        return InterpResult::Trap;
                    } else {
                        u64_from_i64(a / b)
                    }
                },
            }
        }
        IntBinOp::DivU => if b.0 == 0 {
            return InterpResult::Trap;
        } else {
            a / b
        },
        IntBinOp::RemS => {
            match ty {
                IntType::Int32 => {
                    let a = i64_from_i32(a);
                    let b = i64_from_i32(b);
                    if b.0 == 0 {
                        return InterpResult::Trap;
                    } else {
                        u64_from_i64(a % b)
                    }
                },
                IntType::Int64 => {
                    let a = i64_from_u64(a);
                    let b = i64_from_u64(b);
                    if b.0 == 0 {
                        return InterpResult::Trap;
                    } else {
                        u64_from_i64(a % b)
                    }
                },
            }
        }
        IntBinOp::RemU => if b.0 == 0 {
            return InterpResult::Trap;
        } else {
            a % b
        },
        IntBinOp::And => a & b,
        IntBinOp::Or => a | b,
        IntBinOp::Xor => a ^ b,
        IntBinOp::Shl => {
            match ty {
                IntType::Int32 => a << ((b.0 as usize) & 31),
                IntType::Int64 => a << ((b.0 as usize) & 63),
            }
        }
        IntBinOp::ShrU => {
            match ty {
                IntType::Int32 => a >> ((b.0 as usize) & 31),
                IntType::Int64 => a >> ((b.0 as usize) & 63),
            }
        }
        IntBinOp::ShrS => {
            match ty {
                IntType::Int32 => u64_from_i64(i64_from_i32(a) >> ((b.0 as usize) & 31)),
                IntType::Int64 => u64_from_i64(i64_from_u64(a) >> ((b.0 as usize) & 63)),
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

    InterpResult::Value(Some(match ty {
        IntType::Int32 => Dynamic::from_u32(res.0 as u32),
        IntType::Int64 => Dynamic::from_u64(res.0),
    }))
}

fn extend_signed(val: Wrapping<u64>, ty: IntType) -> Wrapping<i64> {
    match ty {
        IntType::Int32 => i64_from_i32(val),
        IntType::Int64 => i64_from_u64(val),
    }
}

fn interp_int_cmp(ty: IntType, op: IntCmpOp, a: Dynamic, b: Dynamic) -> Dynamic {
    assert_eq!(a.get_type(), ty.to_type());
    assert_eq!(b.get_type(), ty.to_type());

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

    match ty {
        IntType::Int32 => Dynamic::from_u32(res),
        IntType::Int64 => Dynamic::from_u64(res as u64),
    }
}

fn interp_int_eqz(ty: IntType, a: Dynamic) -> Dynamic {
    assert_eq!(a.get_type(), ty.to_type());

    let a = a.to_int();

    Dynamic::from_u32(if a.0 == 0 { 1 } else { 0 })
}

fn interp_float_bin(ty: FloatType, op: FloatBinOp, a: Dynamic, b: Dynamic) -> Dynamic {
    assert_eq!(a.get_type(), ty.to_type());
    assert_eq!(b.get_type(), ty.to_type());

    let ao = a.to_float();
    let bo = b.to_float();

    let res = match op {
        FloatBinOp::Add => ao + bo,
        FloatBinOp::Sub => ao - bo,
        FloatBinOp::Mul => ao * bo,
        FloatBinOp::Div => ao / bo,
        FloatBinOp::Min => if ao.is_nan() { ao } else if bo.is_nan() { bo } else { ao.min(bo) },
        FloatBinOp::Max => if ao.is_nan() { ao } else if bo.is_nan() { bo } else { ao.max(bo) },
        FloatBinOp::Copysign => {
            match (a, b) {
                (Dynamic::Float32(a), Dynamic::Float32(b)) => copysign_f32(a, b) as f64,
                (Dynamic::Float64(a), Dynamic::Float64(b)) => copysign_f64(a, b),
                _ => panic!()
            }
        }
    };

    Dynamic::from_float(ty, res)
}

fn round_to_even(a: f64) -> f64 {
    let b = a.floor();
    if (a - b).abs() == 0.5f64 {
        if (b as u64) & 1 == 0 {
            b
        } else {
            b + 1f64
        }
    } else {
        a.round()
    }
}

fn interp_float_un(ty: FloatType, op: FloatUnOp, a: Dynamic) -> Dynamic {
    assert_eq!(a.get_type(), ty.to_type());

    let a = a.to_float();

    let res = match op {
        FloatUnOp::Abs => a.abs(),
        FloatUnOp::Neg => -a,
        FloatUnOp::Ceil => a.ceil(),
        FloatUnOp::Floor => a.floor(),
        FloatUnOp::Trunc => a.trunc(),
        FloatUnOp::Nearest => round_to_even(a),
        FloatUnOp::Sqrt => a.sqrt(),
    };

    Dynamic::from_float(ty, res)
}

fn interp_float_cmp(ty: FloatType, op: FloatCmpOp, a: Dynamic, b: Dynamic) -> Dynamic {
    assert_eq!(a.get_type(), ty.to_type());
    assert_eq!(b.get_type(), ty.to_type());

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

fn next_f32(val: f32) -> f32 {
    unsafe {
        let u: u32 = mem::transmute(val);
        mem::transmute(u.wrapping_add(1))
    }
}

fn next_f64(val: f64) -> f64 {
    unsafe {
        let u: u64 = mem::transmute(val);
        mem::transmute(u.wrapping_add(1))
    }
}

fn interp_float_to_int(floattype: FloatType, inttype: IntType, sign: Sign, a: Dynamic) -> Option<Dynamic> {
    assert_eq!(a.get_type(), floattype.to_type());

    if a.to_float().is_nan() {
        return None;
    }

    match (sign, inttype, a) {
        (Sign::Signed, IntType::Int32, Dynamic::Float32(a)) => {
            let a = a.trunc();
            if a >= 2147483648f32 || a <= next_f32(-2147483648f32) {
                return None;
            }
        }
        (Sign::Signed, IntType::Int32, Dynamic::Float64(a)) => {
            let a = a.trunc();
            if a >= 2147483648f64 || a <= next_f64(-2147483648f64) {
                return None;
            }
        }
        (Sign::Signed, IntType::Int64, Dynamic::Float32(a)) => {
            let a = a.trunc();
            if a >= 9223372036854775808f32 || a <= next_f32(-9223372036854775808f32) {
                return None;
            }
        }
        (Sign::Signed, IntType::Int64, Dynamic::Float64(a)) => {
            let a = a.trunc();
            if a >= 9223372036854775808f64 || a <= next_f64(-9223372036854775808f64) {
                return None;
            }
        }
        (Sign::Unsigned, IntType::Int32, Dynamic::Float32(a)) => {
            let a = a.trunc();
            if a >= 4294967296f32 || a < 0f32 {
                return None;
            }
        }
        (Sign::Unsigned, IntType::Int32, Dynamic::Float64(a)) => {
            let a = a.trunc();
            if a >= 4294967296f64 || a < 0f64 {
                return None;
            }
        }
        (Sign::Unsigned, IntType::Int64, Dynamic::Float32(a)) => {
            let a = a.trunc();
            if a >= 18446744073709551616f32 || a < 0f32 {
                return None;
            }
        }
        (Sign::Unsigned, IntType::Int64, Dynamic::Float64(a)) => {
            let a = a.trunc();
            if a >= 18446744073709551616f64 || a < 0f64 {
                return None;
            }
        }
        _ => panic!()
    }

    Some(Dynamic::from_int(inttype, match (sign, inttype) {
        (Sign::Signed, IntType::Int32) => u64_from_real_i32(a.to_float() as i32),
        (Sign::Unsigned, IntType::Int32) => Wrapping(a.to_float() as u64),
        (Sign::Signed, IntType::Int64) => u64_from_i64(Wrapping(a.to_float() as i64)),
        (Sign::Unsigned, IntType::Int64) => Wrapping(a.to_float() as u64),
    }.0))
}

fn interp_int_extend(sign: Sign, a: Dynamic) -> Dynamic {
    assert_eq!(a.get_type(), Type::Int32);

    Dynamic::Int64(match sign {
        Sign::Signed => u64_from_i64(Wrapping(a.to_i32() as i64)),
        Sign::Unsigned => a.to_int()
    })
}

fn interp_int_truncate(a: Dynamic) -> Dynamic {
    Dynamic::from_u32(a.to_u64() as u32)
}

fn interp_int_to_float(inttype: IntType, sign: Sign, floattype: FloatType, a: Dynamic) -> Dynamic {
    assert_eq!(a.get_type(), inttype.to_type());

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
    assert_eq!(type_from, a.get_type());


    let res = match a {
        Dynamic::Int32(v) => v.0 as u64,
        Dynamic::Int64(v) => v.0,
        Dynamic::Float32(v) => unsafe { mem::transmute::<f32, u32>(v) as u64 },
        Dynamic::Float64(v) => unsafe { mem::transmute(v) },
    };

    let fin = match type_to {
        Type::Int32 => Dynamic::from_u32(res as u32),
        Type::Int64 => Dynamic::from_u64(res),
        Type::Float32 => Dynamic::Float32(unsafe { mem::transmute((res & 0xffffffff) as u32) }),
        Type::Float64 => Dynamic::Float64(unsafe { mem::transmute(res) })
    };

    println!("a {} res {} fin {}", a, res, fin);

    fin
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
