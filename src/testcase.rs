use std::str::{self, FromStr};
use std::{mem, f32, f64, fmt};
use std::collections::HashMap;
use std::num::Wrapping;

use sexpr::Sexpr;
use module::{AsBytes, Module, MemoryInfo, FunctionBuilder,
    Export, FunctionIndex, ImportIndex, Names, MemoryChunk,
    Import, FunctionType, ExportIndex};
use types::{Type, Dynamic, IntType, FloatType, Sign, Size};
use ops::{LinearOp, NormalOp, IntBinOp, IntUnOp, IntCmpOp,
    FloatBinOp, FloatUnOp, FloatCmpOp, MemImm};
use interp::{Instance, InterpResult, BoundInstance};
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
                if name.as_slice() == stringify!($ident).as_bytes() {
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
    function_name: Vec<u8>,
    arguments: Vec<Dynamic>,
}

impl fmt::Display for Invoke {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        try!(write!(f, "{}(", str::from_utf8(&self.function_name).unwrap_or("<bad_utf8>")));
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
    Return(Invoke, Option<Dynamic>),
    ReturnNan(Invoke),
    Trap(Invoke),
    NoTrap(Invoke),
}

impl Assert {
    fn run<'a, B: AsBytes>(&self, instance: &mut Instance<'a, B>) {
        match self {
            &Assert::Return(ref invoke, result) => {
                println!("testing {} => {:?}", invoke, result);
                let a = invoke.run(instance);
                match (a, result) {
                    (InterpResult::Value(Some(Dynamic::Int32(a))), Some(Dynamic::Int32(b))) => assert_eq!(a, b),
                    (InterpResult::Value(Some(Dynamic::Int64(a))), Some(Dynamic::Int64(b))) => assert_eq!(a, b),
                    (InterpResult::Value(Some(Dynamic::Float32(a))), Some(Dynamic::Float32(b))) => {
                        println!("{} {}", a, b);
                        assert!(a == b ||
                            (a.is_nan() && b.is_nan() && a.is_sign_negative() == b.is_sign_negative()));
                    }
                    (InterpResult::Value(Some(Dynamic::Float64(a))), Some(Dynamic::Float64(b))) => {
                        println!("{} {}", a, b);
                        assert!(a == b ||
                            (a.is_nan() && b.is_nan() && a.is_sign_negative() == b.is_sign_negative()));
                    }
                    (InterpResult::Value(None), None) => {}
                    _ => panic!("no match: {:?} vs {:?}", a, result)
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
            &Assert::NoTrap(ref invoke) => {
                println!("testing {} doesn't trap", invoke);
                assert!(invoke.run(instance) != InterpResult::Trap);
            }
        }
    }
}

pub struct TestCase {
    modules: Vec<(Module<Vec<u8>>, Vec<Assert>)>
}

fn parse_type(text: &[u8]) -> Type {
    match text {
        b"i32" => Type::Int32,
        b"i64" => Type::Int64,
        b"f32" => Type::Float32,
        b"f64" => Type::Float64,
        _ => panic!()
    }
}

fn parse_type_expr(s: &Sexpr) -> Type {
    match s {
        &Sexpr::Identifier(ref text) => parse_type(text.as_slice()),
        _ => panic!(),
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
            return match ty.as_slice() {
                b"i32.const" => parse_int(value, IntType::Int32),
                b"i64.const" => parse_int(value, IntType::Int64),
                b"f32.const" => parse_float(value, FloatType::Float32),
                b"f64.const" => parse_float(value, FloatType::Float64),
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
                    let mut import_names = HashMap::new();

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
                                    function_names.insert(name.as_slice(), function_index);
                                }
                                function_index += 1;
                            };
                            (import &module &name &ty) => {
                                m.imports.push(Import {
                                    function_type: parse_function_ty(ty),
                                    module_name: parse_name(module),
                                    function_name: parse_name(name),
                                });
                            };
                            (import &id &module &name &ty) => {
                                import_names.insert(parse_var_id(id), m.imports.len());
                                m.imports.push(Import {
                                    function_type: parse_function_ty(ty),
                                    module_name: parse_name(module),
                                    function_name: parse_name(name),
                                });
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
                                    function_names: &function_names,
                                    import_names: &import_names,
                                    label_names: Vec::new()
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
                                                        ctx.func.ty.param_types.push(parse_type(v.as_slice()).to_u8());
                                                        last_var = false;
                                                    }
                                                    &Sexpr::Variable(ref v) => {
                                                        ctx.local_names.insert(v.as_slice(), ctx.func.ty.param_types.len());
                                                        last_var = true;
                                                    }
                                                    _ => panic!()
                                                }
                                            }
                                            assert!(!last_var);
                                        };
                                        (result &ty) => {
                                            if let &Sexpr::Identifier(ref v) = ty {
                                                ctx.func.ty.return_type = Some(parse_type(v.as_slice()));
                                            } else {
                                                panic!("3");
                                            }
                                        };
                                        (local &id &ty) => {
                                            if let &Sexpr::Variable(ref v) = id {
                                                ctx.local_names.insert(v.as_slice(), ctx.func.ty.param_types.len() + ctx.func.local_types.len());
                                            } else {
                                                panic!("4");
                                            }
                                            if let &Sexpr::Identifier(ref v) = ty {
                                                ctx.func.local_types.push(parse_type(v.as_slice()));
                                            } else {
                                                panic!("5");
                                            }
                                        };
                                        (local *args) => {
                                            for ty in args {
                                                if let &Sexpr::Identifier(ref v) = ty {
                                                    ctx.func.local_types.push(parse_type(v.as_slice()));
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
                                match name {
                                    &Sexpr::String(ref name) => {
                                        let index = match id {
                                            &Sexpr::Variable(ref id) => {
                                                FunctionIndex(*function_names.get(id.as_slice()).unwrap())
                                            }
                                            &Sexpr::Identifier(ref id) => {
                                                FunctionIndex(usize::from_str(str::from_utf8(id.as_slice()).unwrap()).unwrap())
                                            }
                                            _ => panic!("6")
                                        };
                                        m.exports.push(Export {
                                            function_index: index,
                                            function_name: Vec::from(name.as_bytes())
                                        });
                                    }
                                    _ => panic!("7")
                                }
                            };
                            (import *args) => {
                                // already handled
                            };
                            (type &id &ty) => {
                                // println!("found type!");
                            };
                            (type &ty) => {
                                // println!("found type!");
                            };
                            (memory *args) => {
                                let i = 0;
                                let i = if i < args.len() {
                                    match &args[i] {
                                        &Sexpr::Identifier(ref val) => {
                                            m.memory_info.initial_64k_pages = usize::from_str(str::from_utf8(val.as_slice()).unwrap()).unwrap();
                                            i + 1
                                        }
                                        _ => {
                                            m.memory_info.initial_64k_pages = 1;
                                            i
                                        }
                                    }
                                } else {
                                    m.memory_info.initial_64k_pages = 1;
                                    i
                                };
                                let i = if i < args.len() {
                                    match &args[i] {
                                        &Sexpr::Identifier(ref val) => {
                                            m.memory_info.maximum_64k_pages = usize::from_str(str::from_utf8(val.as_slice()).unwrap()).unwrap();
                                            i + 1
                                        }
                                        _ => {
                                            m.memory_info.maximum_64k_pages = 65536;
                                            i
                                        }
                                    }
                                } else {
                                    m.memory_info.maximum_64k_pages = 65536;
                                    i
                                };

                                assert!(m.memory_info.maximum_64k_pages >= m.memory_info.initial_64k_pages);

                                for s in &args[i..] {
                                    sexpr_match!(s;
                                        (segment &offset &data) => {
                                            m.memory_chunks.push(MemoryChunk {
                                                offset:parse_int(offset, IntType::Int32).to_u32() as usize,
                                                data: parse_bin_string(data),
                                            })
                                        };
                                        _ => panic!("a")
                                    );
                                }
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
                (assert_return &invoke) => {
                    modules.last_mut().unwrap().1.push(Assert::Return(parse_invoke(invoke), None));
                };
                (assert_return &invoke &result) => {
                    modules.last_mut().unwrap().1.push(Assert::Return(parse_invoke(invoke), Some(parse_const(result))));
                };
                (assert_return_nan &invoke) => {
                    modules.last_mut().unwrap().1.push(Assert::ReturnNan(parse_invoke(invoke)));
                };
                (assert_trap &invoke &text) => {
                    modules.last_mut().unwrap().1.push(Assert::Trap(parse_invoke(invoke)));
                };
                (invoke *args) => {
                    modules.last_mut().unwrap().1.push(Assert::NoTrap(parse_invoke(s)));
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
            let mut import_table = HashMap::new();
            import_table.insert(&b"spectest"[..], Box::new(SpecTestModule) as Box<BoundInstance>);
            let mut instance = Instance::new(&m.0, import_table);
            for assert in &m.1 {
                assert.run(&mut instance);
            }
        }
    }
}

struct SpecTestModule;

impl BoundInstance for SpecTestModule {
    fn invoke_export(&mut self, func: ExportIndex, args: &[Dynamic]) -> InterpResult {
        assert_eq!(func.0, 0);
        println!("print: {}", args[0].to_u32());
        InterpResult::Value(None)
    }
    fn export_by_name(&self, name: &[u8]) -> ExportIndex {
        assert_eq!(name, b"print");
        ExportIndex(0)
    }
}

struct FunctionContext<'a> {
    local_names: HashMap<&'a [u8], usize>,
    func: FunctionBuilder,
    function_names: &'a HashMap<&'a [u8], usize>,
    import_names: &'a HashMap<&'a [u8], usize>,
    label_names: Vec<&'a [u8]>
}

const EMPTY_DATA: &'static [u8] = &[];

fn log2(data: u32) -> u32 {
    assert!(data.count_ones() == 1);
    data.trailing_zeros()
}

impl<'a> FunctionContext<'a> {
    fn read_local(&self, expr: &Sexpr) -> usize {
        match expr {
            &Sexpr::Variable(ref name) => *self.local_names.get(name.as_bytes()).unwrap(),
            &Sexpr::Identifier(ref num) => usize::from_str(str::from_utf8(num).unwrap()).unwrap(),
            _ => panic!("no local named {}", expr)
        }
    }

    fn read_function(&self, expr: &Sexpr) -> usize {
        match expr {
            &Sexpr::Variable(ref name) => *self.function_names.get(name.as_bytes()).unwrap(),
            &Sexpr::Identifier(ref num) => usize::from_str(str::from_utf8(num).unwrap()).unwrap(),
            _ => panic!("no function named {}", expr)
        }
    }

    fn read_import(&self, expr: &Sexpr) -> usize {
        match expr {
            &Sexpr::Variable(ref name) => *self.import_names.get(name.as_bytes()).unwrap(),
            &Sexpr::Identifier(ref num) => usize::from_str(str::from_utf8(num).unwrap()).unwrap(),
            _ => panic!("no import named {}", expr)
        }
    }

    fn read_label(&self, expr: &Sexpr) -> usize {
        match expr {
            &Sexpr::Variable(ref name) => {
                for i in (0..self.label_names.len() - 1).rev() {
                    if self.label_names[i] == name.as_slice() {
                        return i;
                    }
                }
                panic!("no label named {}", expr)
            }
            &Sexpr::Identifier(ref num) => usize::from_str(str::from_utf8(num).unwrap()).unwrap(),
            _ => panic!("no label named {}", expr)
        }
    }

    fn parse_ops(&mut self, exprs: &'a [Sexpr]) -> usize {
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

    fn parse_mem_imm(&mut self, exprs: &'a [Sexpr], count: usize) -> MemImm {
        let (i, log_of_alignment) = match &exprs[0] {
            &Sexpr::Identifier(ref text) if text.starts_with(b"align=") =>
                (1, log2(u32::from_str(str::from_utf8(&text[b"align=".len()..]).unwrap()).unwrap())),
            _ => (0, 1),
        };
        assert_eq!(self.parse_ops(&exprs[i..]), count);
        MemImm {
            log_of_alignment: log_of_alignment,
            offset: 0
        }
    }

    fn parse_op(&mut self, s: &'a Sexpr) {
        sexpr_match!(s;
            (ident:&op *args) => {
                match op.as_slice() {
                    b"nop" => {self.push(NormalOp::Nop);},
                    b"block" => {
                        let (index, label_name) = if args.len() > 0 {
                            match &args[0] {
                                &Sexpr::Variable(ref v) => (1, Some(v.as_slice())),
                                _ => (0, None)
                            }
                        } else {
                            (0, None)
                        };

                        if let Some(label) = label_name {
                            self.label_names.push(label);
                        }
                        self.func.ops.push(LinearOp::Block);
                        self.parse_ops(&args[index..]);
                        self.func.ops.push(LinearOp::End);
                        if let Some(_) = label_name {
                            self.label_names.pop().unwrap();
                        }
                    }
                    b"loop" => {
                        let (index, label_name_begin) = if args.len() > 0 {
                            match &args[0] {
                                &Sexpr::Variable(ref v) => (1, Some(v.as_slice())),
                                _ => (0, None)
                            }
                        } else {
                            (0, None)
                        };

                        let (index, label_name_end) = if index + 1 < args.len() {
                            match &args[0] {
                                &Sexpr::Variable(ref v) => (index + 1, Some(v.as_slice())),
                                _ => (index, None)
                            }
                        } else {
                            (index, None)
                        };

                        if let Some(label) = label_name_begin {
                            self.label_names.push(label);
                        }

                        if let Some(label) = label_name_end {
                            self.label_names.push(label);
                        }
                        self.func.ops.push(LinearOp::Loop);
                        self.parse_ops(&args[index..]);
                        self.func.ops.push(LinearOp::End);
                        if let Some(_) = label_name_begin {
                            self.label_names.pop().unwrap();
                        }
                        if let Some(_) = label_name_end {
                            self.label_names.pop().unwrap();
                        }
                    }
                    b"if" => {
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
                    b"select" => {
                        assert!(self.parse_ops(args) == 3);
                        self.push(NormalOp::Select);
                    }
                    b"br" => {
                        let relative_depth = self.read_label(&args[0]);

                        if args.len() > 1 {
                            self.parse_op(&args[1]);
                            self.push(NormalOp::Br{has_arg: true, relative_depth: relative_depth as u32});
                        } else {
                            self.push(NormalOp::Br{has_arg: false, relative_depth: relative_depth as u32});
                        }
                    }
                    b"br_if" => {
                        let relative_depth = self.read_label(&args[0]);
                        self.parse_op(&args[1]);

                        if args.len() > 2 {
                            self.parse_op(&args[2]);
                            self.push(NormalOp::BrIf{has_arg: true, relative_depth: relative_depth as u32});
                        } else {
                            self.push(NormalOp::BrIf{has_arg: false, relative_depth: relative_depth as u32});
                        }
                    }
                    b"br_table" => {
                        let relative_depth = self.read_label(&args[0]);

                        let mut i = 1;

                        loop {
                            match &args[i] {
                                &Sexpr::Variable(_) => {},
                                &Sexpr::Identifier(_) => {},
                                _ => break,
                            }
                            i += 1;
                        }

                        self.parse_op(&args[i]);
                        if i + 1 < args.len() {
                            self.parse_op(&args[i + 1]);
                            self.push(NormalOp::BrTable{has_arg: true, target_data: &EMPTY_DATA, default: relative_depth as u32});
                        } else {
                            self.push(NormalOp::BrTable{has_arg: false, target_data: &EMPTY_DATA, default: relative_depth as u32});
                        }
                    }
                    b"return" => {
                        let num = self.parse_ops(args);
                        assert!(num == 0 || num == 1);
                        self.push(NormalOp::Return{has_arg: num == 1});
                    }
                    b"unreachable" => {
                        self.push(NormalOp::Unreachable);
                    }
                    // "drop" => { self.push(NormalOp::Nop); }
                    // "end" => { self.push(NormalOp::Nop); }
                    b"i32.const" |
                    b"i64.const" |
                    b"f64.const" |
                    b"f32.const" => {
                        self.push(NormalOp::Const(parse_const(s)));
                    }
                    b"get_local" => {
                        assert_eq!(args.len(), 1);
                        let local = self.read_local(&args[0]);
                        self.push(NormalOp::GetLocal(local));
                    }
                    b"set_local" => {
                        assert_eq!(self.parse_ops(&args[1..]), 1);
                        assert_eq!(args.len(), 2);
                        let local = self.read_local(&args[0]);
                        self.push(NormalOp::SetLocal(local));
                    }
                    b"tee_local" => {
                        assert_eq!(self.parse_ops(&args[1..]), 1);
                        assert_eq!(args.len(), 2);
                        let local = self.read_local(&args[0]);
                        self.push(NormalOp::TeeLocal(local));
                    }
                    b"call" => {
                        let index = self.read_function(&args[0]);
                        let num = self.parse_ops(&args[1..]);
                        self.push(NormalOp::Call{argument_count: num as u32, index: FunctionIndex(index)});
                    }
                    // "callindirect" => { self.push(NormalOp::Nop); }
                    b"call_import" => {
                        let index = self.read_import(&args[0]);
                        let num = self.parse_ops(&args[1..]);
                        self.push(NormalOp::CallImport{argument_count: num as u32, index: ImportIndex(index)});
                    }
                    b"i32.load8_s" => {
                        let memimm = self.parse_mem_imm(args, 1);
                        self.push(NormalOp::IntLoad(IntType::Int32, Sign::Signed, Size::I8, memimm));
                    }
                    b"i32.load8_u" => {
                        let memimm = self.parse_mem_imm(args, 1);
                        self.push(NormalOp::IntLoad(IntType::Int32, Sign::Unsigned, Size::I8, memimm));
                    }
                    b"i32.load16_s" => {
                        let memimm = self.parse_mem_imm(args, 1);
                        self.push(NormalOp::IntLoad(IntType::Int32, Sign::Signed, Size::I16, memimm));
                    }
                    b"i32.load16_u" => {
                        let memimm = self.parse_mem_imm(args, 1);
                        self.push(NormalOp::IntLoad(IntType::Int32, Sign::Unsigned, Size::I16, memimm));
                    }
                    b"i64.load8_s" => {
                        let memimm = self.parse_mem_imm(args, 1);
                        self.push(NormalOp::IntLoad(IntType::Int64, Sign::Signed, Size::I8, memimm));
                    }
                    b"i64.load8_u" => {
                        let memimm = self.parse_mem_imm(args, 1);
                        self.push(NormalOp::IntLoad(IntType::Int64, Sign::Unsigned, Size::I8, memimm));
                    }
                    b"i64.load16_s" => {
                        let memimm = self.parse_mem_imm(args, 1);
                        self.push(NormalOp::IntLoad(IntType::Int64, Sign::Signed, Size::I16, memimm));
                    }
                    b"i64.load16_u" => {
                        let memimm = self.parse_mem_imm(args, 1);
                        self.push(NormalOp::IntLoad(IntType::Int64, Sign::Unsigned, Size::I16, memimm));
                    }
                    b"i64.load32_s" => {
                        let memimm = self.parse_mem_imm(args, 1);
                        self.push(NormalOp::IntLoad(IntType::Int64, Sign::Signed, Size::I32, memimm));
                    }
                    b"i64.load32_u" => {
                        let memimm = self.parse_mem_imm(args, 1);
                        self.push(NormalOp::IntLoad(IntType::Int64, Sign::Unsigned, Size::I32, memimm));
                    }
                    b"i32.load" => {
                        let memimm = self.parse_mem_imm(args, 1);
                        self.push(NormalOp::IntLoad(IntType::Int32, Sign::Unsigned, Size::I32, memimm));
                    }
                    b"i64.load" => {
                        let memimm = self.parse_mem_imm(args, 1);
                        self.push(NormalOp::IntLoad(IntType::Int64, Sign::Unsigned, Size::I64, memimm));
                    }
                    b"f32.load" => {
                        let memimm = self.parse_mem_imm(args, 1);
                        self.push(NormalOp::FloatLoad(FloatType::Float32, memimm));
                    }
                    b"f64.load" => {
                        let memimm = self.parse_mem_imm(args, 1);
                        self.push(NormalOp::FloatLoad(FloatType::Float64, memimm));
                    }
                    b"i32.store8" => {
                        let memimm = self.parse_mem_imm(args, 2);
                        self.push(NormalOp::IntStore(IntType::Int32, Size::I8, memimm));
                    }
                    b"i32.store16" => {
                        let memimm = self.parse_mem_imm(args, 2);
                        self.push(NormalOp::IntStore(IntType::Int32, Size::I16, memimm));
                    }
                    b"i64.store8" => {
                        let memimm = self.parse_mem_imm(args, 2);
                        self.push(NormalOp::IntStore(IntType::Int64, Size::I8, memimm));
                    }
                    b"i64.store16" => {
                        let memimm = self.parse_mem_imm(args, 2);
                        self.push(NormalOp::IntStore(IntType::Int64, Size::I16, memimm));
                    }
                    b"i64.store32" => {
                        let memimm = self.parse_mem_imm(args, 2);
                        self.push(NormalOp::IntStore(IntType::Int64, Size::I32, memimm));
                    }
                    b"i32.store" => {
                        let memimm = self.parse_mem_imm(args, 2);
                        self.push(NormalOp::IntStore(IntType::Int32, Size::I32, memimm));
                    }
                    b"i64.store" => {
                        let memimm = self.parse_mem_imm(args, 2);
                        self.push(NormalOp::IntStore(IntType::Int64, Size::I64, memimm));
                    }
                    b"f32.store" => {
                        let memimm = self.parse_mem_imm(args, 2);
                        self.push(NormalOp::FloatStore(FloatType::Float32, memimm));
                    }
                    b"f64.store" => {
                        let memimm = self.parse_mem_imm(args, 2);
                        self.push(NormalOp::FloatStore(FloatType::Float64, memimm));
                    }
                    // "current_memory" => { self.push(NormalOp::Nop); }
                    // "grow_memory" => { self.push(NormalOp::Nop); }
                    b"i32.add" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::Add));
                    }
                    b"i32.sub" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::Sub));
                    }
                    b"i32.mul" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::Mul));
                    }
                    b"i32.div_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::DivS));
                    }
                    b"i32.div_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::DivU));
                    }
                    b"i32.rem_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::RemS));
                    }
                    b"i32.rem_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::RemU));
                    }
                    b"i32.and" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::And));
                    }
                    b"i32.or" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::Or));
                    }
                    b"i32.xor" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::Xor));
                    }
                    b"i32.shl" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::Shl));
                    }
                    b"i32.shr_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::ShrU));
                    }
                    b"i32.shr_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::ShrS));
                    }
                    b"i32.rotr" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::Rotr));
                    }
                    b"i32.rotl" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int32, IntBinOp::Rotl));
                    }
                    b"i32.eq" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int32, IntCmpOp::Eq));
                    }
                    b"i32.ne" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int32, IntCmpOp::Ne));
                    }
                    b"i32.lt_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int32, IntCmpOp::LtS));
                    }
                    b"i32.le_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int32, IntCmpOp::LeS));
                    }
                    b"i32.lt_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int32, IntCmpOp::LtU));
                    }
                    b"i32.le_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int32, IntCmpOp::LeU));
                    }
                    b"i32.gt_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int32, IntCmpOp::GtS));
                    }
                    b"i32.ge_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int32, IntCmpOp::GeS));
                    }
                    b"i32.gt_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int32, IntCmpOp::GtU));
                    }
                    b"i32.ge_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int32, IntCmpOp::GeU));
                    }
                    b"i32.clz" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntUn(IntType::Int32, IntUnOp::Clz));
                    }
                    b"i32.ctz" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntUn(IntType::Int32, IntUnOp::Ctz));
                    }
                    b"i32.popcnt" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntUn(IntType::Int32, IntUnOp::Popcnt));
                    }
                    b"i32.eqz" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntEqz(IntType::Int32));
                    }
                    b"i64.add" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::Add));
                    }
                    b"i64.sub" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::Sub));
                    }
                    b"i64.mul" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::Mul));
                    }
                    b"i64.div_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::DivS));
                    }
                    b"i64.div_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::DivU));
                    }
                    b"i64.rem_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::RemS));
                    }
                    b"i64.rem_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::RemU));
                    }
                    b"i64.and" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::And));
                    }
                    b"i64.or" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::Or));
                    }
                    b"i64.xor" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::Xor));
                    }
                    b"i64.shl" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::Shl));
                    }
                    b"i64.shr_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::ShrU));
                    }
                    b"i64.shr_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::ShrS));
                    }
                    b"i64.rotr" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::Rotr));
                    }
                    b"i64.rotl" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntBin(IntType::Int64, IntBinOp::Rotl));
                    }
                    b"i64.eq" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int64, IntCmpOp::Eq));
                    }
                    b"i64.ne" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int64, IntCmpOp::Ne));
                    }
                    b"i64.lt_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int64, IntCmpOp::LtS));
                    }
                    b"i64.le_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int64, IntCmpOp::LeS));
                    }
                    b"i64.lt_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int64, IntCmpOp::LtU));
                    }
                    b"i64.le_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int64, IntCmpOp::LeU));
                    }
                    b"i64.gt_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int64, IntCmpOp::GtS));
                    }
                    b"i64.ge_s" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int64, IntCmpOp::GeS));
                    }
                    b"i64.gt_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int64, IntCmpOp::GtU));
                    }
                    b"i64.ge_u" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::IntCmp(IntType::Int64, IntCmpOp::GeU));
                    }
                    b"i64.clz" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntUn(IntType::Int64, IntUnOp::Clz));
                    }
                    b"i64.ctz" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntUn(IntType::Int64, IntUnOp::Ctz));
                    }
                    b"i64.popcnt" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntUn(IntType::Int64, IntUnOp::Popcnt));
                    }
                    b"i64.eqz" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntEqz(IntType::Int64));
                    }
                    b"f32.add" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Add));
                    }
                    b"f32.sub" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Sub));
                    }
                    b"f32.mul" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Mul));
                    }
                    b"f32.div" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Div));
                    }
                    b"f32.min" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Min));
                    }
                    b"f32.max" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Max));
                    }
                    b"f32.copysign" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float32, FloatBinOp::Copysign));
                    }
                    b"f32.abs" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Abs));
                    }
                    b"f32.neg" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Neg));
                    }
                    b"f32.ceil" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Ceil));
                    }
                    b"f32.floor" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Floor));
                    }
                    b"f32.trunc" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Trunc));
                    }
                    b"f32.nearest" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Nearest));
                    }
                    b"f32.sqrt" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float32, FloatUnOp::Sqrt));
                    }
                    b"f32.eq" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Eq));
                    }
                    b"f32.ne" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Ne));
                    }
                    b"f32.lt" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Lt));
                    }
                    b"f32.le" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Le));
                    }
                    b"f32.gt" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Gt));
                    }
                    b"f32.ge" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float32, FloatCmpOp::Ge));
                    }
                    b"f64.add" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Add));
                    }
                    b"f64.sub" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Sub));
                    }
                    b"f64.mul" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Mul));
                    }
                    b"f64.div" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Div));
                    }
                    b"f64.min" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Min));
                    }
                    b"f64.max" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Max));
                    }
                    b"f64.copysign" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatBin(FloatType::Float64, FloatBinOp::Copysign));
                    }
                    b"f64.abs" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Abs));
                    }
                    b"f64.neg" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Neg));
                    }
                    b"f64.ceil" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Ceil));
                    }
                    b"f64.floor" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Floor));
                    }
                    b"f64.trunc" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Trunc));
                    }
                    b"f64.nearest" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Nearest));
                    }
                    b"f64.sqrt" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatUn(FloatType::Float64, FloatUnOp::Sqrt));
                    }
                    b"f64.eq" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Eq));
                    }
                    b"f64.ne" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Ne));
                    }
                    b"f64.lt" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Lt));
                    }
                    b"f64.le" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Le));
                    }
                    b"f64.gt" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Gt));
                    }
                    b"f64.ge" => {
                        assert_eq!(self.parse_ops(args), 2);
                        self.push(NormalOp::FloatCmp(FloatType::Float64, FloatCmpOp::Ge));
                    }
                    b"i32.trunc_s/f32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatToInt(FloatType::Float32, IntType::Int32, Sign::Signed));
                    }
                    b"i32.trunc_s/f64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatToInt(FloatType::Float64, IntType::Int32, Sign::Signed));
                    }
                    b"i32.trunc_u/f32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatToInt(FloatType::Float32, IntType::Int32, Sign::Unsigned));
                    }
                    b"i32.trunc_u/f64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatToInt(FloatType::Float64, IntType::Int32, Sign::Unsigned));
                    }
                    b"i32.wrap/i64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntTruncate);
                    }
                    b"i64.trunc_s/f32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatToInt(FloatType::Float32, IntType::Int64, Sign::Signed));
                    }
                    b"i64.trunc_s/f64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatToInt(FloatType::Float64, IntType::Int64, Sign::Signed));
                    }
                    b"i64.trunc_u/f32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatToInt(FloatType::Float32, IntType::Int64, Sign::Unsigned));
                    }
                    b"i64.trunc_u/f64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatToInt(FloatType::Float64, IntType::Int64, Sign::Unsigned));
                    }
                    b"i64.extend_s/i32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntExtend(Sign::Signed));
                    }
                    b"i64.extend_u/i32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntExtend(Sign::Unsigned));
                    }
                    b"f32.convert_s/i32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntToFloat(IntType::Int32, Sign::Signed, FloatType::Float32));
                    }
                    b"f32.convert_u/i32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntToFloat(IntType::Int32, Sign::Unsigned, FloatType::Float32));
                    }
                    b"f32.convert_s/i64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntToFloat(IntType::Int64, Sign::Signed, FloatType::Float32));
                    }
                    b"f32.convert_u/i64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntToFloat(IntType::Int64, Sign::Unsigned, FloatType::Float32));
                    }
                    b"f32.demote/f64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatConvert(FloatType::Float32));
                    }
                    b"f32.reinterpret/i32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::Reinterpret(Type::Int32, Type::Float32));
                    }
                    b"f64.convert_s/i32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntToFloat(IntType::Int32, Sign::Signed, FloatType::Float64));
                    }
                    b"f64.convert_u/i32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntToFloat(IntType::Int32, Sign::Unsigned, FloatType::Float64));
                    }
                    b"f64.convert_s/i64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntToFloat(IntType::Int64, Sign::Signed, FloatType::Float64));
                    }
                    b"f64.convert_u/i64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::IntToFloat(IntType::Int64, Sign::Unsigned, FloatType::Float64));
                    }
                    b"f64.promote/f32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::FloatConvert(FloatType::Float64));
                    }
                    b"f64.reinterpret/i64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::Reinterpret(Type::Int64, Type::Float64));
                    }
                    b"i32.reinterpret/f32" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::Reinterpret(Type::Int32, Type::Float32));
                    }
                    b"i64.reinterpret/f64" => {
                        assert_eq!(self.parse_ops(args), 1);
                        self.push(NormalOp::Reinterpret(Type::Int64, Type::Float64));
                    }
                    _ => panic!("unexpected instr: {}", s)
                };
            };
            _ => panic!("unexpected instr: {}", s)
        );
    }
}

fn parse_int(node: &Sexpr, ty: IntType) -> Dynamic {
    match node {
        &Sexpr::Identifier(ref text) => {
            // println!("parsing int {}", text);
            match ty {
                IntType::Int32 => {
                    if text.starts_with(b"-0x") {
                        Dynamic::Int32(!Wrapping(u32::from_str_radix(str::from_utf8(&text[3..]).unwrap(), 16).unwrap()) + Wrapping(1))
                    } else if text.starts_with(b"-") {
                        Dynamic::from_i32(i32::from_str_radix(str::from_utf8(text).unwrap(), 10).unwrap())
                    } else if text.starts_with(b"0x") {
                        Dynamic::from_u32(u32::from_str_radix(str::from_utf8(&text[2..]).unwrap(), 16).unwrap())
                    } else {
                        Dynamic::from_u32(u32::from_str_radix(str::from_utf8(text).unwrap(), 10).unwrap())
                    }
                }
                IntType::Int64 => {
                    if text.starts_with(b"-0x") {
                        Dynamic::Int64(!Wrapping(u64::from_str_radix(str::from_utf8(&text[3..]).unwrap(), 16).unwrap()) + Wrapping(1))
                    } else if text.starts_with(b"-") {
                        Dynamic::from_i64(i64::from_str_radix(str::from_utf8(text).unwrap(), 10).unwrap())
                    } else if text.starts_with(b"0x") {
                        Dynamic::from_u64(u64::from_str_radix(str::from_utf8(&text[2..]).unwrap(), 16).unwrap())
                    } else {
                        Dynamic::from_u64(u64::from_str_radix(str::from_utf8(text).unwrap(), 10).unwrap())
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
            // println!("parsing {}", text);

            let mut text = text.as_slice();
            let neg = if text.starts_with(b"-") {
                text = &text[1..];
                true
            } else {
                false
            };

            let mut res = if text.starts_with(b"0x") {
                unsafe { mem::transmute(hexfloat::parse_bits_64(str::from_utf8(text).unwrap())) }
            } else if text == b"infinity" {
                f64::INFINITY
            } else if text == b"nan" {
                f64::NAN
            } else {
                f64::from_str(str::from_utf8(text).unwrap()).unwrap()
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
        &Sexpr::String(ref text) => Vec::from(text.as_bytes()),
        _ => panic!()
    }
}

fn parse_name(node: &Sexpr) -> Vec<u8> {
    match node {
        &Sexpr::String(ref text) => text.clone(),
        _ => panic!("bad name: {:?}", node),
    }
}

fn parse_var_id(node: &Sexpr) -> &[u8] {
    match node {
        &Sexpr::Variable(ref text) => text.as_slice(),
        _ => panic!("bad var: {:?}", node),
    }
}

fn parse_function_ty(node: &Sexpr) -> FunctionType<Vec<u8>> {
    let mut ty = FunctionType {
        param_types: Vec::new(),
        return_type: None
    };

    sexpr_match!(node;
        (param *params) => {
            for p in params {
                ty.param_types.push(parse_type_expr(p).to_u8());
            }
        };
        _ => panic!()
    );
    ty
}
