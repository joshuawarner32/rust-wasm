use std::str;

use sexpr::Sexpr;
use module::{Module, ModuleBuilder, MemoryInfo};

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
            if let &Sexpr::Identifier(ref name) = &$val[0] {
                let $ident = &$val[0];
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

impl TestCase {
    pub fn parse(bytes: &[u8]) -> TestCase {
        let text = str::from_utf8(bytes).unwrap();
        let exprs = Sexpr::parse(text);

        for s in &exprs {
            sexpr_match!(s;
                (module *it) => {
                    let mut m = ModuleBuilder::new();

                    for s in it {
                        sexpr_match!(s;
                            (func) => {};
                            (func &name *it) => {
                                for s in it {
                                    sexpr_match!(s;
                                        (param &id &ty) => {
                                            println!("found param!");
                                        };
                                        (result &ty) => {
                                            println!("found result!");
                                        };
                                        (local &id &ty) => {
                                            println!("found local!");
                                        };
                                        _ => {
                                            parse_instr(s);
                                        }
                                    );
                                }
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

fn parse_instr(s: &Sexpr) {
    sexpr_match!(s;
        (&instr *args) => {
        };
        _ => panic!("unexpected instr: {}", s)
    );
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
