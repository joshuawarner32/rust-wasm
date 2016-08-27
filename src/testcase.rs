use std::str;
use std::collections::HashMap;

use sexpr::Sexpr;
use module::{AsBytes, Module, MemoryInfo, FunctionBuilder, Export, FunctionIndex};
use types::{Type, Dynamic, IntType, FloatType};
use ops::{NormalOp, IntBinOp};
use interp::Instance;

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

impl Invoke {
    fn run<'a, B: AsBytes>(&self, instance: &mut Instance<'a, B>) -> Option<Dynamic> {
        let func = instance.module.find(self.function_name.as_bytes()).unwrap();
        instance.invoke(func, &self.arguments)
    }
}

pub enum Assert {
    Return(Invoke, Dynamic),
    Trap(Invoke)
}

impl Assert {
    fn run<'a, B: AsBytes>(&self, instance: &mut Instance<'a, B>) {
        match self {
            &Assert::Return(ref invoke, result) => {
                assert_eq!(invoke.run(instance), Some(result));
            }
            &Assert::Trap(ref invoke) => {
                panic!();
            }
        }
    }
}

pub struct TestCase {
    module: Module<Vec<u8>>,
    asserts: Vec<Assert>
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
                "i32.const" => {
                    Dynamic::from_i32(parse_int(value) as i32)
                }
                "i64.const" => {
                    Dynamic::from_i64(parse_int(value) as i64)
                }
                // &Sexpr::Identifier("i32.const") => {
                //     Dynamic::from_i32(parse_int(it[1]))
                // }
                // &Sexpr::Identifier("i32.const") => {
                //     Dynamic::from_i32(parse_int(it[1]))
                // }
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

        let mut asserts = Vec::new();
        let mut module = None;

        for s in &exprs {
            sexpr_match!(s;
                (module *it) => {
                    let mut m = Module::<Vec<u8>>::new();

                    let mut function_names = HashMap::new();

                    for s in it {
                        sexpr_match!(s;
                            (func *it) => {
                                let mut it = it.iter();

                                let name = if let Some(&Sexpr::Variable(ref v)) = it.next() {
                                    Some(v)
                                } else {
                                    None
                                };

                                let mut func = FunctionBuilder::new();

                                let mut local_names = HashMap::new();

                                while let Some(s) = it.next() {
                                    sexpr_match!(s;
                                        (param &id &ty) => {
                                            if let &Sexpr::Variable(ref v) = id {
                                                local_names.insert(v.as_str(), func.ty.param_types.len());
                                            } else {
                                                panic!();
                                            }
                                            if let &Sexpr::Identifier(ref v) = ty {
                                                func.ty.param_types.push(parse_type(v.as_str()).to_u8());
                                            } else {
                                                panic!();
                                            }
                                        };
                                        (result &ty) => {
                                            if let &Sexpr::Identifier(ref v) = ty {
                                                func.ty.return_type = Some(parse_type(v.as_str()));
                                            } else {
                                                panic!();
                                            }
                                        };
                                        (local &id &ty) => {
                                            if let &Sexpr::Variable(ref v) = id {
                                                local_names.insert(v.as_str(), func.ty.param_types.len() + func.local_types.len());
                                            } else {
                                                panic!();
                                            }
                                            if let &Sexpr::Identifier(ref v) = ty {
                                                func.local_types.push(parse_type(v.as_str()));
                                            } else {
                                                panic!();
                                            }
                                        };
                                        _ => {
                                            parse_op(s, &mut func.ops, &local_names);
                                        }
                                    );
                                }

                                if let Some(name) = name {
                                    function_names.insert(name.as_str(), m.functions.len());
                                }

                                m.functions.push(func.ty.clone());
                                m.code.push(func.build());
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
                                            _ => panic!()
                                        }
                                    }
                                    _ => panic!()
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
                    module = Some(m)
                };
                (assert_invalid &module &text) => {
                    panic!();
                };
                (assert_return &invoke &result) => {
                    asserts.push(Assert::Return(parse_invoke(invoke), parse_const(result)));
                };
                (assert_return_nan &invoke) => {
                    panic!();
                };
                (assert_trap &invoke &text) => {
                    asserts.push(Assert::Trap(parse_invoke(invoke)));
                };
                (invoke &ident *args) => {
                    panic!();
                };
                _ => {
                    panic!("unhandled: {}", s);
                }
            );
        }

        TestCase {
            module: module.unwrap(),
            asserts: asserts
        }
    }

    pub fn run_all(&self) {
        let mut instance = Instance::new(&self.module);
        for assert in &self.asserts {
            assert.run(&mut instance);
        }
    }
}

fn read_local(exprs: &[Sexpr], local_names: &HashMap<&str, usize>) -> usize {
    assert!(exprs.len() == 1);
    match &exprs[0] {
        &Sexpr::Variable(ref name) => *local_names.get(name.as_str()).unwrap(),
        _ => panic!()
    }
}

fn parse_ops(exprs: &[Sexpr], ops: &mut Vec<NormalOp>, local_names: &HashMap<&str, usize>) -> usize {
    let mut num = 0;
    for s in exprs {
        parse_op(s, ops, local_names);
        num += 1;
    }
    num
}

fn parse_op(s: &Sexpr, ops: &mut Vec<NormalOp>, local_names: &HashMap<&str, usize>) {
    sexpr_match!(s;
        (ident:&op *args) => {
            match op.as_str() {
                "nop" => {ops.push(NormalOp::Nop);},
                // "block" => NormalOp::Nop,
                // "loop" => NormalOp::Nop,
                // "if" => NormalOp::Nop,
                // "else" => NormalOp::Nop,
                // "select" => NormalOp::Nop,
                // "br" => NormalOp::Nop,
                // "brif" => NormalOp::Nop,
                // "brtable" => NormalOp::Nop,
                "return" => {
                    let num = parse_ops(args, ops, local_names);
                    assert!(num == 0 || num == 1);
                    ops.push(NormalOp::Return{has_arg: num == 1});
                }
                "unreachable" => { ops.push(NormalOp::Nop); }
                "drop" => { ops.push(NormalOp::Nop); }
                "end" => { ops.push(NormalOp::Nop); }
                "i32.const" => { ops.push(NormalOp::Nop); }
                "i64.const" => { ops.push(NormalOp::Nop); }
                "f64.const" => { ops.push(NormalOp::Nop); }
                "f32.const" => { ops.push(NormalOp::Nop); }
                "get_local" => {
                    ops.push(NormalOp::GetLocal(read_local(args, local_names)));
                }
                "set_local" => {
                    ops.push(NormalOp::SetLocal(read_local(args, local_names)));
                }
                "tee_local" => {
                    ops.push(NormalOp::TeeLocal(read_local(args, local_names)));
                }
                "call" => { ops.push(NormalOp::Nop); }
                "callindirect" => { ops.push(NormalOp::Nop); }
                "callimport" => { ops.push(NormalOp::Nop); }
                "i32.load8s" => { ops.push(NormalOp::Nop); }
                "i32.load8u" => { ops.push(NormalOp::Nop); }
                "i32.load16s" => { ops.push(NormalOp::Nop); }
                "i32.load16u" => { ops.push(NormalOp::Nop); }
                "i64.load8s" => { ops.push(NormalOp::Nop); }
                "i64.load8u" => { ops.push(NormalOp::Nop); }
                "i64.load16s" => { ops.push(NormalOp::Nop); }
                "i64.load16u" => { ops.push(NormalOp::Nop); }
                "i64.load32s" => { ops.push(NormalOp::Nop); }
                "i64.load32u" => { ops.push(NormalOp::Nop); }
                "i32.load" => { ops.push(NormalOp::Nop); }
                "i64.load" => { ops.push(NormalOp::Nop); }
                "f32.load" => { ops.push(NormalOp::Nop); }
                "f64.load" => { ops.push(NormalOp::Nop); }
                "i32.store8" => { ops.push(NormalOp::Nop); }
                "i32.store16" => { ops.push(NormalOp::Nop); }
                "i64.store8" => { ops.push(NormalOp::Nop); }
                "i64.store16" => { ops.push(NormalOp::Nop); }
                "i64.store32" => { ops.push(NormalOp::Nop); }
                "i32.store" => { ops.push(NormalOp::Nop); }
                "i64.store" => { ops.push(NormalOp::Nop); }
                "f32.store" => { ops.push(NormalOp::Nop); }
                "f64.store" => { ops.push(NormalOp::Nop); }
                "current_memory" => { ops.push(NormalOp::Nop); }
                "grow_memory" => { ops.push(NormalOp::Nop); }
                "i32.add" => {
                    assert_eq!(parse_ops(args, ops, local_names), 2);
                    ops.push(NormalOp::IntBin(IntType::Int32, IntBinOp::Add));
                }
                "i32.sub" => { ops.push(NormalOp::Nop); }
                "i32.mul" => { ops.push(NormalOp::Nop); }
                "i32.div_s" => { ops.push(NormalOp::Nop); }
                "i32.div_u" => { ops.push(NormalOp::Nop); }
                "i32.rem_s" => { ops.push(NormalOp::Nop); }
                "i32.rem_u" => { ops.push(NormalOp::Nop); }
                "i32.and" => { ops.push(NormalOp::Nop); }
                "i32.or" => { ops.push(NormalOp::Nop); }
                "i32.xor" => { ops.push(NormalOp::Nop); }
                "i32.shl" => { ops.push(NormalOp::Nop); }
                "i32.shr_u" => { ops.push(NormalOp::Nop); }
                "i32.shr_s" => { ops.push(NormalOp::Nop); }
                "i32.rotr" => { ops.push(NormalOp::Nop); }
                "i32.rotl" => { ops.push(NormalOp::Nop); }
                "i32.eq" => { ops.push(NormalOp::Nop); }
                "i32.ne" => { ops.push(NormalOp::Nop); }
                "i32.lt_s" => { ops.push(NormalOp::Nop); }
                "i32.le_s" => { ops.push(NormalOp::Nop); }
                "i32.lt_u" => { ops.push(NormalOp::Nop); }
                "i32.le_u" => { ops.push(NormalOp::Nop); }
                "i32.gt_s" => { ops.push(NormalOp::Nop); }
                "i32.ge_s" => { ops.push(NormalOp::Nop); }
                "i32.gt_u" => { ops.push(NormalOp::Nop); }
                "i32.ge_u" => { ops.push(NormalOp::Nop); }
                "i32.clz" => { ops.push(NormalOp::Nop); }
                "i32.ctz" => { ops.push(NormalOp::Nop); }
                "i32.popcnt" => { ops.push(NormalOp::Nop); }
                "i32.eqz" => { ops.push(NormalOp::Nop); }
                "i64.add" => { ops.push(NormalOp::Nop); }
                "i64.sub" => { ops.push(NormalOp::Nop); }
                "i64.mul" => { ops.push(NormalOp::Nop); }
                "i64.divs" => { ops.push(NormalOp::Nop); }
                "i64.divu" => { ops.push(NormalOp::Nop); }
                "i64.rems" => { ops.push(NormalOp::Nop); }
                "i64.remu" => { ops.push(NormalOp::Nop); }
                "i64.and" => { ops.push(NormalOp::Nop); }
                "i64.or" => { ops.push(NormalOp::Nop); }
                "i64.xor" => { ops.push(NormalOp::Nop); }
                "i64.shl" => { ops.push(NormalOp::Nop); }
                "i64.shru" => { ops.push(NormalOp::Nop); }
                "i64.shrs" => { ops.push(NormalOp::Nop); }
                "i64.rotr" => { ops.push(NormalOp::Nop); }
                "i64.rotl" => { ops.push(NormalOp::Nop); }
                "i64.eq" => { ops.push(NormalOp::Nop); }
                "i64.ne" => { ops.push(NormalOp::Nop); }
                "i64.lts" => { ops.push(NormalOp::Nop); }
                "i64.les" => { ops.push(NormalOp::Nop); }
                "i64.ltu" => { ops.push(NormalOp::Nop); }
                "i64.leu" => { ops.push(NormalOp::Nop); }
                "i64.gts" => { ops.push(NormalOp::Nop); }
                "i64.ges" => { ops.push(NormalOp::Nop); }
                "i64.gtu" => { ops.push(NormalOp::Nop); }
                "i64.geu" => { ops.push(NormalOp::Nop); }
                "i64.clz" => { ops.push(NormalOp::Nop); }
                "i64.ctz" => { ops.push(NormalOp::Nop); }
                "i64.popcnt" => { ops.push(NormalOp::Nop); }
                "i64.eqz" => { ops.push(NormalOp::Nop); }
                "f32.add" => { ops.push(NormalOp::Nop); }
                "f32.sub" => { ops.push(NormalOp::Nop); }
                "f32.mul" => { ops.push(NormalOp::Nop); }
                "f32.div" => { ops.push(NormalOp::Nop); }
                "f32.min" => { ops.push(NormalOp::Nop); }
                "f32.max" => { ops.push(NormalOp::Nop); }
                "f32.abs" => { ops.push(NormalOp::Nop); }
                "f32.neg" => { ops.push(NormalOp::Nop); }
                "f32.copysign" => { ops.push(NormalOp::Nop); }
                "f32.ceil" => { ops.push(NormalOp::Nop); }
                "f32.floor" => { ops.push(NormalOp::Nop); }
                "f32.trunc" => { ops.push(NormalOp::Nop); }
                "f32.nearest" => { ops.push(NormalOp::Nop); }
                "f32.sqrt" => { ops.push(NormalOp::Nop); }
                "f32.eq" => { ops.push(NormalOp::Nop); }
                "f32.ne" => { ops.push(NormalOp::Nop); }
                "f32.lt" => { ops.push(NormalOp::Nop); }
                "f32.le" => { ops.push(NormalOp::Nop); }
                "f32.gt" => { ops.push(NormalOp::Nop); }
                "f32.ge" => { ops.push(NormalOp::Nop); }
                "f64.add" => { ops.push(NormalOp::Nop); }
                "f64.sub" => { ops.push(NormalOp::Nop); }
                "f64.mul" => { ops.push(NormalOp::Nop); }
                "f64.div" => { ops.push(NormalOp::Nop); }
                "f64.min" => { ops.push(NormalOp::Nop); }
                "f64.max" => { ops.push(NormalOp::Nop); }
                "f64.abs" => { ops.push(NormalOp::Nop); }
                "f64.neg" => { ops.push(NormalOp::Nop); }
                "f64.copysign" => { ops.push(NormalOp::Nop); }
                "f64.ceil" => { ops.push(NormalOp::Nop); }
                "f64.floor" => { ops.push(NormalOp::Nop); }
                "f64.trunc" => { ops.push(NormalOp::Nop); }
                "f64.nearest" => { ops.push(NormalOp::Nop); }
                "f64.sqrt" => { ops.push(NormalOp::Nop); }
                "f64.eq" => { ops.push(NormalOp::Nop); }
                "f64.ne" => { ops.push(NormalOp::Nop); }
                "f64.lt" => { ops.push(NormalOp::Nop); }
                "f64.le" => { ops.push(NormalOp::Nop); }
                "f64.gt" => { ops.push(NormalOp::Nop); }
                "f64.ge" => { ops.push(NormalOp::Nop); }
                "i32.trunc_s/f32" => { ops.push(NormalOp::Nop); }
                "i32.trunc_s/f64" => { ops.push(NormalOp::Nop); }
                "i32.trunc_u/f32" => { ops.push(NormalOp::Nop); }
                "i32.trunc_u/f64" => { ops.push(NormalOp::Nop); }
                "i32.wrap/i64" => { ops.push(NormalOp::Nop); }
                "i64.trunc_s/f32" => { ops.push(NormalOp::Nop); }
                "i64.trunc_s/f64" => { ops.push(NormalOp::Nop); }
                "i64.trunc_u/f32" => { ops.push(NormalOp::Nop); }
                "i64.trunc_u/f64" => { ops.push(NormalOp::Nop); }
                "i64.extend_s/i32" => { ops.push(NormalOp::Nop); }
                "i64.extend_u/i32" => { ops.push(NormalOp::Nop); }
                "f32.convert_s/i32" => { ops.push(NormalOp::Nop); }
                "f32.convert_u/i32" => { ops.push(NormalOp::Nop); }
                "f32.convert_s/i64" => { ops.push(NormalOp::Nop); }
                "f32.convert_u/i64" => { ops.push(NormalOp::Nop); }
                "f32.demote/f64" => { ops.push(NormalOp::Nop); }
                "f32.reinterpret/i32" => { ops.push(NormalOp::Nop); }
                "f64.convert_s/i32" => { ops.push(NormalOp::Nop); }
                "f64.convert_u/i32" => { ops.push(NormalOp::Nop); }
                "f64.convert_s/i64" => { ops.push(NormalOp::Nop); }
                "f64.convert_u/i64" => { ops.push(NormalOp::Nop); }
                "f64.promote/f32" => { ops.push(NormalOp::Nop); }
                "f64.reinterpret/i64" => { ops.push(NormalOp::Nop); }
                "i32.reinterpret/f32" => { ops.push(NormalOp::Nop); }
                "i64.reinterpret/f64" => { ops.push(NormalOp::Nop); }
                _ => panic!("unexpected instr: {}", op)
            };
        };
        _ => panic!("unexpected instr: {}", s)
    );
}

fn parse_int(node: &Sexpr) -> i64 {
    match node {
        &Sexpr::Identifier(ref text) => {
            if text.starts_with("0x") {
                i64::from_str_radix(&text[2..], 16).unwrap()
            } else {
                str::parse(text).unwrap()
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