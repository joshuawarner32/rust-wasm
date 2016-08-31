use std::str::{self, FromStr};
use std::{mem, f32, f64, fmt};
use std::collections::HashMap;
use std::num::Wrapping;

use sexpr::Sexpr;
use module::{AsBytes, Module, MemoryInfo, FunctionBuilder, Export, FunctionIndex, Names};
use types::{Type, Dynamic, IntType, FloatType, Sign};
use ops::{LinearOp, NormalOp, IntBinOp, IntUnOp, IntCmpOp, FloatBinOp, FloatUnOp, FloatCmpOp};
use interp::{Instance, InterpResult};
use hexfloat;

macro_rules! vec_form {
    ($val:expr => () => $code:expr) => {{
        if $val.len() == 0 {
            Some($code)
        } else {
            None
        }
    }};
    ($val:expr => (*$rest:ident) => $code:expr) => {{
        let $rest = &$val[..];
        Some($code)
    }};
    ($val:expr => (ident:&$ident:ident $($parts:tt)*) => $code:expr) => {{
        if $val.len() > 0 {
            if let &Sexpr::Identifier(ref $ident) = &$val[0] {
                vec_form!($val[1..] => ($($parts)*) => $code)
            } else {
                None
            }
        } else {
            None
        }
    }};
    ($val:expr => (str:&$ident:ident $($parts:tt)*) => $code:expr) => {{
        if $val.len() > 0 {
            if let &Sexpr::String(ref $ident) = &$val[0] {
                vec_form!($val[1..] => ($($parts)*) => $code)
            } else {
                None
            }
        } else {
            None
        }
    }};
    ($val:expr => (&$ident:ident $($parts:tt)*) => $code:expr) => {{
        if $val.len() > 0 {
            let $ident = &$val[0];

            vec_form!($val[1..] => ($($parts)*) => $code)
        } else {
            None
        }
    }};
    ($val:expr => ($ident:ident $($parts:tt)*) => $code:expr) => {{
        if $val.len() > 0 {
            if let &Sexpr::Identifier(ref name) = &$val[0] {
                if name == stringify!($ident) {
                    vec_form!($val[1..] => ($($parts)*) => $code)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    }};
}

macro_rules! sexpr_match {
    ($val:expr;) => {{ None }};
    ($val:expr; _ => $code:expr) => {{ Some($code) }};
    ($val:expr; $sexpr:tt => $code:expr; $($sexpr_rest:tt => $code_rest:expr);*) => {{
        let val = $val;
        let res = if let &Sexpr::List(ref items) = val {
            vec_form!(items => $sexpr => $code)
        } else {
            None
        };
        if let None = res {
            sexpr_match!(val; $($sexpr_rest => $code_rest);*)
        } else {
            res
        }
    }};
}

pub struct Invoke {
    function_name: String,
    arguments: Vec<Dynamic>,
}

impl fmt::Display for Invoke {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        try!(write!(f, "{}(", self.function_name));
        for (i, a) in self.arguments.iter().enumerate() {
            try!(write!(f, "{}{}", if i == 0 { "" } else { ", " }, a));
        }
        write!(f, ")")
    }
}

impl Invoke {
    fn run<'a, B: AsBytes>(&self, instance: &mut Instance<'a, B>) -> InterpResult {
        let func =
            instance.module.find(self.function_name.as_bytes())
            .or_else(|| instance.module.find_by_debug_name(self.function_name.as_bytes()))
            .unwrap();
        let res = instance.invoke(func, &self.arguments);
        assert_eq!(instance.call_stack_depth, 0);
        res
    }
}

pub enum Assert {
    Return(Invoke, Dynamic),
    ReturnNan(Invoke),
    Trap(Invoke)
}

impl Assert {
    fn run<'a, B: AsBytes>(&self, instance: &mut Instance<'a, B>) {
        match self {
            &Assert::Return(ref invoke, result) => {
                println!("testing {} => {}", invoke, result);
                let a = invoke.run(instance);
                match (a, result) {
                    (InterpResult::Value(Some(Dynamic::Int32(a))), Dynamic::Int32(b)) => assert_eq!(a, b),
                    (InterpResult::Value(Some(Dynamic::Int64(a))), Dynamic::Int64(b)) => assert_eq!(a, b),
                    (InterpResult::Value(Some(Dynamic::Float32(a))), Dynamic::Float32(b)) => {
                        println!("{} {}", a, b);
                        assert!(a == b ||
                            (a.is_nan() && b.is_nan() && a.is_sign_negative() == b.is_sign_negative()));
                    }
                    (InterpResult::Value(Some(Dynamic::Float64(a))), Dynamic::Float64(b)) => {
                        println!("{} {}", a, b);
                        assert!(a == b ||
                            (a.is_nan() && b.is_nan() && a.is_sign_negative() == b.is_sign_negative()));
                    }
                    _ => panic!("no match: {:?} vs {:}", a, result)
                }
            }
            &Assert::ReturnNan(ref invoke) => {
                println!("testing {} returns nan", invoke);
                match invoke.run(instance) {
                    InterpResult::Value(Some(Dynamic::Float32(v))) => assert!(v.is_nan()),
                    InterpResult::Value(Some(Dynamic::Float64(v))) => assert!(v.is_nan()),
                    _ => panic!()
                }
            }
            &Assert::Trap(ref invoke) => {
                println!("testing {} traps", invoke);
                assert_eq!(invoke.run(instance), InterpResult::Trap);
            }
        }
    }
}

pub struct TestCase {
    modules: Vec<(Module<Vec<u8>>, Vec<Assert>)>
}

fn parse_type(text: &str) -> Type {
    match text {
        "i32" => Type::Int32,
        "i64" => Type::Int64,
        "f32" => Type::Float32,
        "f64" => Type::Float64,
        _ => panic!()
    }
}

fn parse_invoke(s: &Sexpr) -> Invoke {
    sexpr_match!(s;
        (invoke str:&name *args) => {
            let args = args.iter().map(parse_const).collect::<Vec<_>>();
            return Invoke {
                function_name: name.clone(),
                arguments: args
            };
        };
        _ => panic!()
    );
    panic!();
}

fn parse_const(s: &Sexpr) -> Dynamic {
    sexpr_match!(s;
        (ident:&ty &value) => {
            return match ty.as_str() {
                "i32.const" => parse_int(value, IntType::Int32),
                "i64.const" => parse_int(value, IntType::Int64),
                "f32.const" => parse_float(value, FloatType::Float32),
                "f64.const" => parse_float(value, FloatType::Float64),
                _ => panic!()
            };
        };
        _ => panic!()
    );
    panic!();
}

impl TestCase {
    pub fn parse(bytes: &[u8]) -> TestCase {
        let text = str::from_utf8(bytes).unwrap();
        let exprs = Sexpr::parse(text);

        let mut modules = Vec::new();

        for s in &exprs {
            sexpr_match!(s;
                (module *it) => {
                    let mut m = Module::<Vec<u8>>::new();

                    let mut function_names = HashMap::new();
                    let mut function_index = 0;

                    for s in it {
                        sexpr_match!(s;
                            (func *it) => {
                                let mut it = it.iter();

                                let mut name = None;

                                let mut text = None;

                                while let Some(s) = it.next() {
                                    match s {
                                        &Sexpr::Variable(ref v) => {
                                            name = Some(v);
                                            continue;
                                        }
                                        &Sexpr::String(ref v) => {
                                            text = Some(v);
                                            continue;
                                        }
                                        _ => break
                                    }
                                }

                                if let Some(text) = text {
                                    m.names.push(Names {
                                        function_name: Vec::from(text.as_bytes()),
                                        local_names: Vec::new(),
                                    });
                                } else {
                                    m.names.push(Names {
                                        function_name: Vec::new(),
                                        local_names: Vec::new(),
                                    });
                                }

                                if let Some(name) = name {
                                    function_names.insert(name.as_str(), function_index);
                                }
                                function_index += 1;
                            };
                            _ => {}
                        );
                    }

                    for s in it {
                        sexpr_match!(s;
                            (func *it) => {
                                let mut it = it.iter();

                                let mut ctx = FunctionContext {
                                    func: FunctionBuilder::new(),
                                    local_names: HashMap::new(),
                                    function_names: &function_names
                                };

                                while let Some(s) = it.next() {
                                    match s {
                                        &Sexpr::Variable(_) => continue,
                                        &Sexpr::String(_) => continue,
                                        _ => {}
                                    }
                                    sexpr_match!(s;
                                        (param *args) => {
                                            let mut last_var = false;
                                            for a in args {
                                                match a {
                                                    &Sexpr::Identifier(ref v) => {
                                                        ctx.func.ty.param_types.push(parse_type(v.as_str()).to_u8());
                                                        last_var = false;
                                                    }
                                                    &Sexpr::Variable(ref v) => {
                                                        ctx.local_names.insert(v.as_str(), ctx.func.ty.param_types.len());
                                                        last_var = true;
                                                    }
                                                    _ => panic!()
                                                }
                                            }
                                            assert!(!last_var);
                                        };
                                        (result &ty) => {
                                            if let &Sexpr::Identifier(ref v) = ty {
                                                ctx.func.ty.return_type = Some(parse_type(v.as_str()));
                                            } else {
                                                panic!("3");
                                            }
                                        };
                                        (local &id &ty) => {
                                            if let &Sexpr::Variable(ref v) = id {
                                                ctx.local_names.insert(v.as_str(), ctx.func.ty.param_types.len() + ctx.func.local_types.len());
                                            } else {
                                                panic!("4");
                                            }
                                            if let &Sexpr::Identifier(ref v) = ty {
                                                ctx.func.local_types.push(parse_type(v.as_str()));
                                            } else {
                                                panic!("5");
                                            }
                                        };
                                        (local *args) => {
                                            for ty in args {
                                                println!("gotit {:?}", text);
                                                if let &Sexpr::Identifier(ref v) = ty {
                                                    ctx.func.local_types.push(parse_type(v.as_str()));
                                                } else {
                                                    panic!("6");
                                                }
                                            }
                                        };
                                        _ => {
                                            ctx.parse_op(s);
                                        }
                                    );
                                }

                                m.functions.push(ctx.func.ty.clone());
                                m.code.push(ctx.func.build());
                            };
                            (export &name &id) => {
                                match id {
                                    &Sexpr::Variable(ref id) => {
                                        match name {
                                            &Sexpr::String(ref name) => {
                                                m.exports.push(Export {
                                                    function_index: FunctionIndex(*function_names.get(id.as_str()).unwrap()),
                                                    function_name: Vec::from(name.as_bytes())
                                                });
                                            }
                                            _ => panic!("6")
                                        }
                                    }
                                    _ => panic!("7")
                                }
                            };
                            (import &module &name &ty) => {
                                // println!("found import!");
                            };
                            (import &id &module &name &ty) => {
                                // println!("found import!");
                            };
                            (type &id &ty) => {
                                // println!("found type!");
                            };
                            (type &ty) => {
                                // println!("found type!");
                            };
                            (memory *args) => {
                                // m.memory_info.initial_64k_pages = parse_int(initial);
                                // m.memory_info.maximum_64k_pages = parse_int(max);
                                //
                                // assert!(m.memory_info.maximum_64k_pages >= m.memory_info.initial_64k_pages);
                                //
                                // for s in segments {
                                //     sexpr_match!(s;
                                //         (segment &offset &data) => {
                                //             m.memory_chunks.push(MemoryChunk {
                                //                 offset: parse_int(offset),
                                //                 data: parse_bin_string(data),
                                //             })
                                //         };
                                //         _ => panic!("a")
                                //     );
                                // }
                            };
                            (table *items) => {
                                // println!("found table!");
                            };
                            (start &id) => {
                                // println!("found start!");
                            };
                            _ => {
                                panic!("unhandled inner: {}", s);
                            }
                        );
                    }
                    modules.push((m, Vec::new()));
                };
                (assert_invalid &module &text) => {
                    // TODO
                    // panic!("8");
                };
                (assert_return &invoke &result) => {
                    modules.last_mut().unwrap().1.push(Assert::Return(parse_invoke(invoke), parse_const(result)));
                };
                (assert_return_nan &invoke) => {
                    modules.last_mut().unwrap().1.push(Assert::ReturnNan(parse_invoke(invoke)));
                };
                (assert_trap &invoke &text) => {
                    modules.last_mut().unwrap().1.push(Assert::Trap(parse_invoke(invoke)));
                };
                (invoke &ident *args) => {
                    panic!("10");
                };
                _ => {
                    panic!("unhandled: {}", s);
                }
            );
        }

        TestCase {
            modules: modules
        }
    }

    pub fn run_all(&self) {
        for m in &self.modules {
            let mut instance = Instance::new(&m.0);
            for assert in &m.1 {
                assert.run(&mut instance);
            }
        }
    }
}

struct FunctionContext<'a> {
    local_names: HashMap<&'a str, usize>,
    func: FunctionBuilder,
    function_names: &'a HashMap<&'a str, usize>
}

impl<'a> FunctionContext<'a> {
    fn read_local(&self, expr: &Sexpr) -> usize {
        match expr {
            &Sexpr::Variable(ref name) => *self.local_names.get(name.as_str()).unwrap(),
            &Sexpr::Identifier(ref num) => usize::from_str(num).unwrap(),
            _ => panic!("no local named {}", expr)
        }
    }

    fn read_function(&self, expr: &Sexpr) -> usize {
        match expr {
            &Sexpr::Variable(ref name) => *self.function_names.get(name.as_str()).unwrap(),
            &Sexpr::Identifier(ref num) => usize::from_str(num).unwrap(),
            _ => panic!("no function named {}", expr)
        }
    }

    fn parse_ops(&mut self, exprs: &[Sexpr]) -> usize {
        let mut num = 0;
        for s in exprs {
            self.parse_op(s);
            num += 1;
        }
        num
    }

    fn push(&mut self, op: NormalOp<'static>) {
        self.func.ops.push(LinearOp::Normal(op));
    }

    fn parse_op(&mut self, s: &Sexpr) {
        sexpr_match!(s;
            (ident:&op *args) => {
                match op.as_str() {
                    "nop" => {self.push(NormalOp::Nop);},
                    // "block" => NormalOp::Nop,
                    // "loop" => NormalOp::Nop,
                    "if" => {
                        assert!(args.len() == 2 || args.len() == 3);
                        self.parse_op(&args[0]);
                        self.func.ops.push(LinearOp::If);
                        self.parse_op(&args[1]);
                        if args.len() == 3 {
                            self.func.ops.push(LinearOp::Else);
                            self.parse_op(&args[2]);
                        }
                        self.func.ops.push(LinearOp::End);
                    }
                    // "else" => NormalOp::Nop,
                    // "select" => NormalOp::Nop,
                    // "br" => NormalOp::Nop,
                    // "brif" => NormalOp::Nop,
                    // "brtable" => NormalOp::Nop,
                    "return" => {
                        let num = self.parse_ops(args);
                        assert!(num == 0 || num == 1);
                        self.push(NormalOp::Return{has_arg: num == 1});
                    }
                    // "unreachable" => { self.push(NormalOp::Nop); }
                    // "drop" => { self.push(NormalOp::Nop); }
                    // "end" => { self.push(NormalOp::Nop); }
                    "i32.const" |
                    "i64.const" |
                    "f64.const" |
                    "f32.const" => {
                        self.push(NormalOp::Const(parse_const(s)));
                    }
                    "get_local" => {
                        assert_eq!(args.len(), 1);
                        let local = self.read_local(&args[0]);
                        self.push(NormalOp::GetLocal(local));
                    }
                    "set_local" => {
                        assert_eq!(self.parse_ops(&args[1..]), 1);
                        assert_eq!(args.len(), 2);
                        let local = self.read_local(&args[0]);
                        self.push(NormalOp::SetLocal(local));
                    }
                    "tee_local" => {
                        assert_eq!(self.parse_ops(&args[1..]), 1);
                        assert_eq!(args.len(), 2);
                        let local = self.read_local(&args[0]);
                        self.push(NormalOp::TeeLocal(local));
                    }
                    "call" => {
                        let index = self.read_function(&args[0]);
                        let num = self.parse_ops(&args[1..]);
                        self.push(NormalOp::Call{argument_count: num as u32, index: FunctionIndex(index)});
                    }
                    // "callindirect" => { self.push(NormalOp::Nop); }
                    // "callimport" => { self.push(NormalOp::Nop); }
                    // "i32.load8s" => { self.push(NormalOp::Nop); }
                    // "i32.load8u" => { self.push(NormalOp::Nop); }
                    // "i32.load16s" => { self.push(NormalOp::Nop); }
                    // "i32.load16u" => { self.push(NormalOp::Nop); }
                    // "i64.load8s" => { self.push(NormalOp::Nop); }
                    // "i64.load8u" => { self.push(NormalOp::Nop); }
                    // "i64.load16s" => { self.push(NormalOp::Nop); }
                    // "i64.load16u" => { self.push(NormalOp::Nop); }
                    // "i64.load32s" => { self.push(NormalOp::Nop); }
                    // "i64.load32u" => { self.push(NormalOp::Nop); }
                    // "i32.load" => { self.push(NormalOp::Nop); }
                    // "i64.load" => { self.push(NormalOp::Nop); }
                    // "f32.load" => { self.push(NormalOp::Nop); }
                    // "f64.load" => { self.push(NormalOp::Nop); }
                    // "i32.store8" => { self.push(NormalOp::Nop); }
                    // "i32.store16" => { self.push(NormalOp::Nop); }
                    // "i64.store8" => { self.push(NormalOp::Nop); }
                    // "i64.store16" => { self.push(NormalOp::Nop); }
                    // "i64.store32" => { self.push(NormalOp::Nop); }
                    // "i32.store" => { self.push(NormalOp::Nop); }
                    // "i64.store" => { self.push(NormalOp::Nop); }
                    // "f32.store" => { self.push(NormalOp::Nop); }
                    // "f64.store" => { self.push(NormalOp::Nop); }
                    // "current_memory" => { self.push(NormalOp::Nop); }
                    // "grow_memory" => { self.push(NormalOp::Nop); }
                    "i32.add" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::Add));
                    }
                    "i32.sub" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::Sub));
                    }
                    "i32.mul" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::Mul));
                    }
                    "i32.div_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::DivS));
                    }
                    "i32.div_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::DivU));
                    }
                    "i32.rem_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::RemS));
                    }
                    "i32.rem_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::RemU));
                    }
                    "i32.and" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::And));
                    }
                    "i32.or" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::Or));
                    }
                    "i32.xor" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::Xor));
                    }
                    "i32.shl" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::Shl));
                    }
                    "i32.shr_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::ShrU));
                    }
                    "i32.shr_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::ShrS));
                    }
                    "i32.rotr" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::Rotr));
                    }
                    "i32.rotl" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::Rotl));
                    }
                    "i32.eq" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int32, IntCmpOp::Eq));
                    }
                    "i32.ne" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int32, IntCmpOp::Ne));
                    }
                    "i32.lt_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int32, IntCmpOp::LtS));
                    }
                    "i32.le_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int32, IntCmpOp::LeS));
                    }
                    "i32.lt_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int32, IntCmpOp::LtU));
                    }
                    "i32.le_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int32, IntCmpOp::LeU));
                    }
                    "i32.gt_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int32, IntCmpOp::GtS));
                    }
                    "i32.ge_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int32, IntCmpOp::GeS));
                    }
                    "i32.gt_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int32, IntCmpOp::GtU));
                    }
                    "i32.ge_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int32, IntCmpOp::GeU));
                    }
                    "i32.clz" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntUn(IntType::Int32, IntUnOp::Clz));
                    }
                    "i32.ctz" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntUn(IntType::Int32, IntUnOp::Ctz));
                    }
                    "i32.popcnt" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntUn(IntType::Int32, IntUnOp::Popcnt));
                    }
                    "i32.eqz" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntEqz(IntType::Int32));
                    }
                    "i64.add" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::Add));
                    }
                    "i64.sub" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::Sub));
                    }
                    "i64.mul" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::Mul));
                    }
                    "i64.div_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::DivS));
                    }
                    "i64.div_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::DivU));
                    }
                    "i64.rem_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::RemS));
                    }
                    "i64.rem_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::RemU));
                    }
                    "i64.and" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::And));
                    }
                    "i64.or" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::Or));
                    }
                    "i64.xor" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::Xor));
                    }
                    "i64.shl" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::Shl));
                    }
                    "i64.shr_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::ShrU));
                    }
                    "i64.shr_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::ShrS));
                    }
                    "i64.rotr" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::Rotr));
                    }
                    "i64.rotl" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::Rotl));
                    }
                    "i64.eq" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int64, IntCmpOp::Eq));
                    }
                    "i64.ne" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int64, IntCmpOp::Ne));
                    }
                    "i64.lt_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int64, IntCmpOp::LtS));
                    }
                    "i64.le_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int64, IntCmpOp::LeS));
                    }
                    "i64.lt_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int64, IntCmpOp::LtU));
                    }
                    "i64.le_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int64, IntCmpOp::LeU));
                    }
                    "i64.gt_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int64, IntCmpOp::GtS));
                    }
                    "i64.ge_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int64, IntCmpOp::GeS));
                    }
                    "i64.gt_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int64, IntCmpOp::GtU));
                    }
                    "i64.ge_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int64, IntCmpOp::GeU));
                    }
                    "i64.clz" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntUn(IntType::Int64, IntUnOp::Clz));
                    }
                    "i64.ctz" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntUn(IntType::Int64, IntUnOp::Ctz));
                    }
                    "i64.popcnt" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntUn(IntType::Int64, IntUnOp::Popcnt));
                    }
                    "i64.eqz" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntEqz(IntType::Int64));
                    }
                    "f32.add" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Add));
                    }
                    "f32.sub" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Sub));
                    }
                    "f32.mul" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Mul));
                    }
                    "f32.div" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Div));
                    }
                    "f32.min" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Min));
                    }
                    "f32.max" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Max));
                    }
                    "f32.copysign" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Copysign));
                    }
                    "f32.abs" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Abs));
                    }
                    "f32.neg" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Neg));
                    }
                    "f32.ceil" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Ceil));
                    }
                    "f32.floor" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Floor));
                    }
                    "f32.trunc" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Trunc));
                    }
                    "f32.nearest" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Nearest));
                    }
                    "f32.sqrt" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Sqrt));
                    }
                    "f32.eq" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Eq));
                    }
                    "f32.ne" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Ne));
                    }
                    "f32.lt" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Lt));
                    }
                    "f32.le" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Le));
                    }
                    "f32.gt" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Gt));
                    }
                    "f32.ge" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Ge));
                    }
                    "f64.add" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Add));
                    }
                    "f64.sub" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Sub));
                    }
                    "f64.mul" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Mul));
                    }
                    "f64.div" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Div));
                    }
                    "f64.min" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Min));
                    }
                    "f64.max" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Max));
                    }
                    "f64.copysign" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Copysign));
                    }
                    "f64.abs" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Abs));
                    }
                    "f64.neg" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Neg));
                    }
                    "f64.ceil" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Ceil));
                    }
                    "f64.floor" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Floor));
                    }
                    "f64.trunc" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Trunc));
                    }
                    "f64.nearest" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Nearest));
                    }
                    "f64.sqrt" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Sqrt));
                    }
                    "f64.eq" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Eq));
                    }
                    "f64.ne" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Ne));
                    }
                    "f64.lt" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Lt));
                    }
                    "f64.le" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Le));
                    }
                    "f64.gt" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Gt));
                    }
                    "f64.ge" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Ge));
                    }
                    "i32.trunc_s/f32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatToInt(FloatType::Float32, IntType::Int32, Sign::Signed));
                    }
                    "i32.trunc_s/f64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatToInt(FloatType::Float64, IntType::Int32, Sign::Signed));
                    }
                    "i32.trunc_u/f32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatToInt(FloatType::Float32, IntType::Int32, Sign::Unsigned));
                    }
                    "i32.trunc_u/f64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatToInt(FloatType::Float64, IntType::Int32, Sign::Unsigned));
                    }
                    "i32.wrap/i64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntTruncate);
                    }
                    "i64.trunc_s/f32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatToInt(FloatType::Float32, IntType::Int64, Sign::Signed));
                    }
                    "i64.trunc_s/f64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatToInt(FloatType::Float64, IntType::Int64, Sign::Signed));
                    }
                    "i64.trunc_u/f32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatToInt(FloatType::Float32, IntType::Int64, Sign::Unsigned));
                    }
                    "i64.trunc_u/f64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatToInt(FloatType::Float64, IntType::Int64, Sign::Unsigned));
                    }
                    "i64.extend_s/i32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntExtend(Sign::Signed));
                    }
                    "i64.extend_u/i32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntExtend(Sign::Unsigned));
                    }
                    "f32.convert_s/i32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntToFloat(IntType::Int32, Sign::Signed, FloatType::Float32));
                    }
                    "f32.convert_u/i32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntToFloat(IntType::Int32, Sign::Unsigned, FloatType::Float32));
                    }
                    "f32.convert_s/i64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntToFloat(IntType::Int64, Sign::Signed, FloatType::Float32));
                    }
                    "f32.convert_u/i64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntToFloat(IntType::Int64, Sign::Unsigned, FloatType::Float32));
                    }
                    "f32.demote/f64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatConvert(FloatType::Float32));
                    }
                    "f32.reinterpret/i32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::Reinterpret(Type::Int32, Type::Float32));
                    }
                    "f64.convert_s/i32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntToFloat(IntType::Int32, Sign::Signed, FloatType::Float64));
                    }
                    "f64.convert_u/i32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntToFloat(IntType::Int32, Sign::Unsigned, FloatType::Float64));
                    }
                    "f64.convert_s/i64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntToFloat(IntType::Int64, Sign::Signed, FloatType::Float64));
                    }
                    "f64.convert_u/i64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntToFloat(IntType::Int64, Sign::Unsigned, FloatType::Float64));
                    }
                    "f64.promote/f32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatConvert(FloatType::Float64));
                    }
                    "f64.reinterpret/i64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::Reinterpret(Type::Float64, Type::Int64));
                    }
                    "i32.reinterpret/f32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::Reinterpret(Type::Float32, Type::Int32));
                    }
                    "i64.reinterpret/f64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::Reinterpret(Type::Float64, Type::Int64));
                    }
                    _ => panic!("unexpected instr: {}", op)
                };
            };
            _ => panic!("unexpected instr: {}", s)
        );
    }
}

fn parse_int(node: &Sexpr, ty: IntType) -> Dynamic {
    match node {
        &Sexpr::Identifier(ref text) => {
            println!("parsing int {}", text);
            match ty {
                IntType::Int32 => {
                    if text.starts_with("-0x") {
                        Dynamic::Int32(!Wrapping(u32::from_str_radix(&text[3..], 16).unwrap()) + Wrapping(1))
                    } else if text.starts_with("-") {
                        Dynamic::from_i32(i32::from_str_radix(text, 10).unwrap())
                    } else if text.starts_with("0x") {
                        Dynamic::from_u32(u32::from_str_radix(&text[2..], 16).unwrap())
                    } else {
                        Dynamic::from_u32(u32::from_str_radix(text, 10).unwrap())
                    }
                }
                IntType::Int64 => {
                    if text.starts_with("-0x") {
                        Dynamic::Int64(!Wrapping(u64::from_str_radix(&text[3..], 16).unwrap()) + Wrapping(1))
                    } else if text.starts_with("-") {
                        Dynamic::from_i64(i64::from_str_radix(text, 10).unwrap())
                    } else if text.starts_with("0x") {
                        Dynamic::from_u64(u64::from_str_radix(&text[2..], 16).unwrap())
                    } else {
                        Dynamic::from_u64(u64::from_str_radix(text, 10).unwrap())
                    }
                }
            }
        }
        _ => panic!("expected number id: {}", node)
    }
}

fn parse_float(node: &Sexpr, ty: FloatType) -> Dynamic {
    match node {
        &Sexpr::Identifier(ref text) => {
            println!("parsing {}", text);

            let mut text = text.as_str();
            let neg = if text.starts_with("-") {
                text = &text[1..];
                true
            } else {
                false
            };

            let mut res = if text.starts_with("0x") {
                unsafe { mem::transmute(hexfloat::parse_bits_64(text)) }
            } else if text == "infinity" {
                f64::INFINITY
            } else if text == "nan" {
                f64::NAN
            } else {
                f64::from_str(text).unwrap()
            };

            if neg {
                res = -res;
            }

            match ty {
                FloatType::Float32 => Dynamic::Float32(res as f32),
                FloatType::Float64 => Dynamic::Float64(res),
            }
        }
        _ => panic!("expected number id: {}", node)
    }
}

fn parse_bin_string(node: &Sexpr) -> Vec<u8> {
    match node {
        &Sexpr::String(ref text) => {
            let text = text.as_bytes();
            let mut res = Vec::new();

            assert!(text[0] == b'"');

            let mut pos = 1;

            while pos < text.len() {
                match text[pos] {
                    b'\\' => {
                        assert!(pos + 2 < text.len());
                        res.push(u8::from_str_radix(str::from_utf8(&text[pos + 1..pos + 2]).unwrap(), 16).unwrap());
                    }
                    b'"' => break,
                    ch => res.push(ch)
                }
                pos += 1;
            }

            res
        }
        _ => panic!()
    }
}
