use std::{fmt, str};

#[derive(Debug)]
pub enum Sexpr {
    String(Vec<u8>),
    Identifier(Vec<u8>),
    Variable(Vec<u8>),
    List(Vec<Sexpr>)
}

struct Parser<'a> {
    text: &'a [u8],
    pos: usize
}

fn is_ws_char(ch: u8) -> bool {
    ch == b' ' || ch == b'\n' || ch == b'\t'
}

fn is_sep_char(ch: u8) -> bool {
    ch == b'(' || ch == b')' || is_ws_char(ch)
}

fn parse_hex_digit(ch: u8) -> u8 {
    match ch {
        b'0'...b'9' => ch - b'0',
        b'a'...b'f' => ch - b'a' + 10,
        b'A'...b'F' => ch - b'A' + 10,
        _ => panic!(),
    }
}

fn unescape(text: &[u8]) -> Vec<u8> {
    let mut res = Vec::new();

    enum State {
        Start,
        Esc,
        One(u8),
    }

    let mut state = State::Start;

    for ch in text {
        match state {
            State::Start => match *ch {
                b'\\' => state = State::Esc,
                _ => res.push(*ch),
            },
            State::Esc => match *ch {
                b'0'...b'9' | b'a'...b'f' | b'A'...b'F' =>
                    state = State::One(parse_hex_digit(*ch)),
                b'n' => {
                    res.push(b'\n');
                    state = State::Start;
                }
                b'\\' => {
                    res.push(b'\\');
                    state = State::Start;
                }
                b'\"' => {
                    res.push(b'\"');
                    state = State::Start;
                }
                _ => panic!("unexpected escape {}", *ch),
            },
            State::One(val) => match *ch {
                b'0'...b'9' | b'a'...b'f' | b'A'...b'F' => {
                    let d = parse_hex_digit(*ch);
                    res.push(d + (val << 4));
                    state = State::Start;
                }
                _ => panic!("unexpected second escape {}", *ch),
            },
        }
    }

    res
}

impl<'a> Parser<'a> {
    fn new(text: &'a str) -> Parser {
        Parser {
            text: text.as_bytes(),
            pos: 0
        }
    }

    fn skip_ws(&mut self) {
        loop {
            while self.pos < self.text.len() && is_ws_char(self.text[self.pos]) {
                self.pos += 1;
            }

            if self.pos + 1 < self.text.len() &&
                self.text[self.pos] == ';' as u8 &&
                self.text[self.pos + 1] == ';' as u8 {
                while self.pos < self.text.len() && self.text[self.pos] != b'\n' {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn parse_item(&mut self) -> Sexpr {
        self.skip_ws();

        if self.pos >= self.text.len() {
            panic!();
        }

        let res = match self.text[self.pos] {
            b'(' => {
                let mut res = Vec::new();

                self.pos += 1;

                self.skip_ws();

                while self.text[self.pos] != b')' {
                    res.push(self.parse_item());
                }

                self.pos += 1;

                Sexpr::List(res)
            }
            b'"' => {
                let mut len = 1;

                loop {
                    if self.pos + len >= self.text.len() {
                        panic!();
                    }

                    if self.text[self.pos + len] == b'\\' {
                        len += 1;
                        if len == self.text.len() {
                            panic!();
                        } else {
                            len += 1;
                        }
                    }

                    if self.text[self.pos + len] == b'"' {
                        len += 1;
                        break;
                    }
                    len += 1;
                }
                let offset = self.pos + 1;
                self.pos += len;

                Sexpr::String(unescape(&self.text[offset..self.pos - 1]))
            }
            // x @ b'0'...b'9' => {
            //     let mut len = 1;
            //     loop {
            //         if self.pos + len >= self.text.len() || is_sep_char(self.text[self.pos + len]) {
            //             break;
            //         } else {
            //             match self.text[self.pos + len] {
            //                 b'0'...b'9' => {},
            //                 _ => break;
            //             }
            //             panic!();
            //         }
            //         len += 1;
            //     }
            //     let offset = self.pos;
            //     self.pos += len;

            //     Sexpr::Number(::std::str::from_utf8(&self.text[offset..self.pos]).unwrap().to_owned())
            // }
            b'$' => {
                let mut len = 1;
                loop {
                    if self.pos + len >= self.text.len() || is_sep_char(self.text[self.pos + len]) {
                        break;
                    } else if self.text[self.pos + len] == b'"' {
                        panic!();
                    }
                    len += 1;
                }
                let offset = self.pos + 1;
                self.pos += len;

                Sexpr::Variable(Vec::from(&self.text[offset..self.pos]))
            }
            x if !is_sep_char(x) => {
                let mut len = 1;
                loop {
                    if self.pos + len >= self.text.len() || is_sep_char(self.text[self.pos + len]) {
                        break;
                    }
                    len += 1;
                }
                let offset = self.pos;
                self.pos += len;

                Sexpr::Identifier(Vec::from(&self.text[offset..self.pos]))
            }
            _ => panic!()
        };

        self.skip_ws();

        res
    }

    fn at_end(&self) -> bool {
        self.pos >= self.text.len()
    }
}

impl Sexpr {
    pub fn parse(text: &str) -> Vec<Sexpr> {
        let mut res = Vec::new();
        let mut p = Parser::new(text);
        while !p.at_end() {
            res.push(p.parse_item());
        }
        res
    }
}


impl fmt::Display for Sexpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            &Sexpr::String(ref text) |
            &Sexpr::Identifier(ref text) |
            &Sexpr::Variable(ref text) => write!(f, "{}", str::from_utf8(text).unwrap_or("<bad utf8>")),
            &Sexpr::List(ref items) => {
                try!(write!(f, "("));
                for (i, s) in items.iter().enumerate() {
                    if i != 0 {
                        try!(write!(f, " "));
                    }
                    try!(write!(f, "{}", s));
                }
                write!(f, ")")
            }
        }
    }
}
