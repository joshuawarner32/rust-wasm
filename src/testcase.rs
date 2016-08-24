use std::str;
use std::collections::HashMap;

use sexpr::Sexpr;
use module::{Module, ModuleBuilder, MemoryInfo};
use types::Type;
use ops::NormalOp;

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

pub struct TestCase {

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

impl TestCase {
    pub fn parse(bytes: &[u8]) -> TestCase {
        let text = str::from_utf8(bytes).unwrap();
        let exprs = Sexpr::parse(text);

        for s in &exprs {
            sexpr_match!(s;
                (module *it) => {
                    let mut m = ModuleBuilder::new();

                    let mut function_names = HashMap::new();
                    let mut functions = Vec::new();

                    for s in it {
                        sexpr_match!(s;
                            (func *it) => {
                                let mut it = it.iter();

                                let name = if let Some(&Sexpr::Variable(ref v)) = it.next() {
                                    Some(v)
                                } else {
                                    None
                                };

                                let mut param_types = Vec::new();

                                let mut param_names = HashMap::new();

                                let mut result_type = None;

                                let mut local_types = Vec::new();
                                let mut local_names = HashMap::new();

                                let mut ops = Vec::new();

                                while let Some(s) = it.next() {
                                    sexpr_match!(s;
                                        (param &id &ty) => {
                                            if let &Sexpr::Variable(ref v) = id {
                                                param_names.insert(v, param_types.len());
                                            } else {
                                                panic!();
                                            }
                                            if let &Sexpr::Identifier(ref v) = ty {
                                                param_types.push(parse_type(v.as_str()));
                                            } else {
                                                panic!();
                                            }
                                        };
                                        (result &ty) => {
                                            if let &Sexpr::Identifier(ref v) = ty {
                                                result_type = Some(parse_type(v.as_str()));
                                            } else {
                                                panic!();
                                            }
                                        };
                                        (local &id &ty) => {
                                            if let &Sexpr::Variable(ref v) = id {
                                                local_names.insert(v, local_types.len());
                                            } else {
                                                panic!();
                                            }
                                            if let &Sexpr::Identifier(ref v) = ty {
                                                local_types.push(parse_type(v.as_str()));
                                            } else {
                                                panic!();
                                            }
                                        };
                                        _ => {
                                            ops.push(parse_op(s));
                                        }
                                    );
                                }

                                if let Some(name) = name {
                                    function_names.insert(name, functions.len());
                                }

                                functions.push(0);
                            };
                            (export &name &id) => {
                                // println!("found export!");
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
                };
                (assert_invalid &module &text) => {
                    // println!("found assert_invalid");
                };
                (assert_return &module) => {
                    // println!("found assert_return (no text)");
                };
                (assert_return &module &text) => {
                    // println!("found assert_return (text)");
                };
                (assert_return_nan &module) => {
                    // println!("found assert_return_nan");
                };
                (assert_trap &module &text) => {
                    // println!("found assert_trap");
                };
                (invoke &ident *args) => {
                    // println!("found invoke");
                };
                _ => {
                    panic!("unhandled: {}", s);
                }
            );
        }

        TestCase {
        }
    }

    pub fn run_all(&self) {

    }
}

fn parse_op(s: &Sexpr) -> NormalOp {
    sexpr_match!(s;
        (ident:&op *args) => {
            return match op.as_str() {
                "nop" => NormalOp::Nop,
                "block" => NormalOp::Nop,
                "loop" => NormalOp::Nop,
                "if" => NormalOp::Nop,
                "else" => NormalOp::Nop,
                "select" => NormalOp::Nop,
                "br" => NormalOp::Nop,
                "brif" => NormalOp::Nop,
                "brtable" => NormalOp::Nop,
                "return" => NormalOp::Nop,
                "unreachable" => NormalOp::Nop,
                "drop" => NormalOp::Nop,
                "end" => NormalOp::Nop,
                "i32.const" => NormalOp::Nop,
                "i64.const" => NormalOp::Nop,
                "f64.const" => NormalOp::Nop,
                "f32.const" => NormalOp::Nop,
                "get_local" => NormalOp::Nop,
                "set_local" => NormalOp::Nop,
                "tee_local" => NormalOp::Nop,
                "call" => NormalOp::Nop,
                "callindirect" => NormalOp::Nop,
                "callimport" => NormalOp::Nop,
                "i32.load8s" => NormalOp::Nop,
                "i32.load8u" => NormalOp::Nop,
                "i32.load16s" => NormalOp::Nop,
                "i32.load16u" => NormalOp::Nop,
                "i64.load8s" => NormalOp::Nop,
                "i64.load8u" => NormalOp::Nop,
                "i64.load16s" => NormalOp::Nop,
                "i64.load16u" => NormalOp::Nop,
                "i64.load32s" => NormalOp::Nop,
                "i64.load32u" => NormalOp::Nop,
                "i32.load" => NormalOp::Nop,
                "i64.load" => NormalOp::Nop,
                "f32.load" => NormalOp::Nop,
                "f64.load" => NormalOp::Nop,
                "i32.store8" => NormalOp::Nop,
                "i32.store16" => NormalOp::Nop,
                "i64.store8" => NormalOp::Nop,
                "i64.store16" => NormalOp::Nop,
                "i64.store32" => NormalOp::Nop,
                "i32.store" => NormalOp::Nop,
                "i64.store" => NormalOp::Nop,
                "f32.store" => NormalOp::Nop,
                "f64.store" => NormalOp::Nop,
                "current_memory" => NormalOp::Nop,
                "grow_memory" => NormalOp::Nop,
                "i32.add" => NormalOp::Nop,
                "i32.sub" => NormalOp::Nop,
                "i32.mul" => NormalOp::Nop,
                "i32.div_s" => NormalOp::Nop,
                "i32.div_u" => NormalOp::Nop,
                "i32.rem_s" => NormalOp::Nop,
                "i32.rem_u" => NormalOp::Nop,
                "i32.and" => NormalOp::Nop,
                "i32.or" => NormalOp::Nop,
                "i32.xor" => NormalOp::Nop,
                "i32.shl" => NormalOp::Nop,
                "i32.shr_u" => NormalOp::Nop,
                "i32.shr_s" => NormalOp::Nop,
                "i32.rotr" => NormalOp::Nop,
                "i32.rotl" => NormalOp::Nop,
                "i32.eq" => NormalOp::Nop,
                "i32.ne" => NormalOp::Nop,
                "i32.lt_s" => NormalOp::Nop,
                "i32.le_s" => NormalOp::Nop,
                "i32.lt_u" => NormalOp::Nop,
                "i32.le_u" => NormalOp::Nop,
                "i32.gt_s" => NormalOp::Nop,
                "i32.ge_s" => NormalOp::Nop,
                "i32.gt_u" => NormalOp::Nop,
                "i32.ge_u" => NormalOp::Nop,
                "i32.clz" => NormalOp::Nop,
                "i32.ctz" => NormalOp::Nop,
                "i32.popcnt" => NormalOp::Nop,
                "i32.eqz" => NormalOp::Nop,
                "i64.add" => NormalOp::Nop,
                "i64.sub" => NormalOp::Nop,
                "i64.mul" => NormalOp::Nop,
                "i64.divs" => NormalOp::Nop,
                "i64.divu" => NormalOp::Nop,
                "i64.rems" => NormalOp::Nop,
                "i64.remu" => NormalOp::Nop,
                "i64.and" => NormalOp::Nop,
                "i64.or" => NormalOp::Nop,
                "i64.xor" => NormalOp::Nop,
                "i64.shl" => NormalOp::Nop,
                "i64.shru" => NormalOp::Nop,
                "i64.shrs" => NormalOp::Nop,
                "i64.rotr" => NormalOp::Nop,
                "i64.rotl" => NormalOp::Nop,
                "i64.eq" => NormalOp::Nop,
                "i64.ne" => NormalOp::Nop,
                "i64.lts" => NormalOp::Nop,
                "i64.les" => NormalOp::Nop,
                "i64.ltu" => NormalOp::Nop,
                "i64.leu" => NormalOp::Nop,
                "i64.gts" => NormalOp::Nop,
                "i64.ges" => NormalOp::Nop,
                "i64.gtu" => NormalOp::Nop,
                "i64.geu" => NormalOp::Nop,
                "i64.clz" => NormalOp::Nop,
                "i64.ctz" => NormalOp::Nop,
                "i64.popcnt" => NormalOp::Nop,
                "i64.eqz" => NormalOp::Nop,
                "f32.add" => NormalOp::Nop,
                "f32.sub" => NormalOp::Nop,
                "f32.mul" => NormalOp::Nop,
                "f32.div" => NormalOp::Nop,
                "f32.min" => NormalOp::Nop,
                "f32.max" => NormalOp::Nop,
                "f32.abs" => NormalOp::Nop,
                "f32.neg" => NormalOp::Nop,
                "f32.copysign" => NormalOp::Nop,
                "f32.ceil" => NormalOp::Nop,
                "f32.floor" => NormalOp::Nop,
                "f32.trunc" => NormalOp::Nop,
                "f32.nearest" => NormalOp::Nop,
                "f32.sqrt" => NormalOp::Nop,
                "f32.eq" => NormalOp::Nop,
                "f32.ne" => NormalOp::Nop,
                "f32.lt" => NormalOp::Nop,
                "f32.le" => NormalOp::Nop,
                "f32.gt" => NormalOp::Nop,
                "f32.ge" => NormalOp::Nop,
                "f64.add" => NormalOp::Nop,
                "f64.sub" => NormalOp::Nop,
                "f64.mul" => NormalOp::Nop,
                "f64.div" => NormalOp::Nop,
                "f64.min" => NormalOp::Nop,
                "f64.max" => NormalOp::Nop,
                "f64.abs" => NormalOp::Nop,
                "f64.neg" => NormalOp::Nop,
                "f64.copysign" => NormalOp::Nop,
                "f64.ceil" => NormalOp::Nop,
                "f64.floor" => NormalOp::Nop,
                "f64.trunc" => NormalOp::Nop,
                "f64.nearest" => NormalOp::Nop,
                "f64.sqrt" => NormalOp::Nop,
                "f64.eq" => NormalOp::Nop,
                "f64.ne" => NormalOp::Nop,
                "f64.lt" => NormalOp::Nop,
                "f64.le" => NormalOp::Nop,
                "f64.gt" => NormalOp::Nop,
                "f64.ge" => NormalOp::Nop,
                "i32.trunc_s/f32" => NormalOp::Nop,
                "i32.trunc_s/f64" => NormalOp::Nop,
                "i32.trunc_u/f32" => NormalOp::Nop,
                "i32.trunc_u/f64" => NormalOp::Nop,
                "i32.wrap/i64" => NormalOp::Nop,
                "i64.trunc_s/f32" => NormalOp::Nop,
                "i64.trunc_s/f64" => NormalOp::Nop,
                "i64.trunc_u/f32" => NormalOp::Nop,
                "i64.trunc_u/f64" => NormalOp::Nop,
                "i64.extend_s/i32" => NormalOp::Nop,
                "i64.extend_u/i32" => NormalOp::Nop,
                "f32.convert_s/i32" => NormalOp::Nop,
                "f32.convert_u/i32" => NormalOp::Nop,
                "f32.convert_s/i64" => NormalOp::Nop,
                "f32.convert_u/i64" => NormalOp::Nop,
                "f32.demote/f64" => NormalOp::Nop,
                "f32.reinterpret/i32" => NormalOp::Nop,
                "f64.convert_s/i32" => NormalOp::Nop,
                "f64.convert_u/i32" => NormalOp::Nop,
                "f64.convert_s/i64" => NormalOp::Nop,
                "f64.convert_u/i64" => NormalOp::Nop,
                "f64.promote/f32" => NormalOp::Nop,
                "f64.reinterpret/i64" => NormalOp::Nop,
                "i32.reinterpret/f32" => NormalOp::Nop,
                "i64.reinterpret/f64" => NormalOp::Nop,
                _ => panic!("unexpected instr: {}", op)
            };
        };
        _ => panic!("unexpected instr: {}", s)
    );
    panic!();
}

fn parse_int(node: &Sexpr) -> usize {
    match node {
        &Sexpr::Identifier(ref text) => {
            str::parse(text).unwrap()
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
