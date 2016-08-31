use std::str::FromStr;
use std::cmp::min;

pub enum Expanded {
    Normal {  // (-1)^negative * 0.left_aligned_mantissa * 2^exponent
        negative: bool,
        mantissa: u64,
        exponent: i32
    },
    Zero {
        negative: bool,
    },
    Infinite {
        negative: bool,
    },
    Nan {
        negative: bool,
    }
}

fn parse_hex_digit(ch: char) -> usize {
    match ch {
        '0'...'9' => (ch as usize) - ('0' as usize),
        'a'...'f' => (ch as usize) - ('a' as usize) + 10,
        'A'...'F' => (ch as usize) - ('A' as usize) + 10,
        _ => panic!(),
    }
}

pub fn parse_expanded(text: &str) -> Expanded {

    let mut text = text;
    let mut neg = false;

    if text.starts_with("-") {
        text = &text[1..];
        neg = true;
    }

    assert!(text.starts_with("0x"));
    let text = &text[2..];

    let (d, f, e) = match text.find(|c| c == '.') {
        None => match text.find(|c| c == 'p') {
            None => (text, "", ""),
            Some(pindex) => (&text[..pindex], "", &text[pindex + 1..]),
        },
        Some(dindex) => match text[dindex + 1..].find(|c| c == 'p') {
            None => (&text[..dindex], &text[dindex + 1..], ""),
            Some(pindex) => (&text[..dindex], &text[dindex + 1..dindex + pindex + 1], &text[dindex + pindex + 2..]),
        }
    };

    let mut exp = if e.len() > 0 { i32::from_str(e).unwrap() } else { 0 };
    assert!(exp >= -1022 && exp <= 1023);

    let mut bits = 0u64;

    let mut shift = 64 - 4;
    for ch in d.chars() {
        bits |= (parse_hex_digit(ch) as u64) << shift;
        shift -= 4;
        exp += 4;
    }

    for ch in f.chars() {
        bits |= (parse_hex_digit(ch) as u64) << shift;
        shift -= 4;
    }

    if bits == 0 {
        Expanded::Zero{negative: neg}
    } else {
        let leading = bits.leading_zeros();
        bits <<= leading;
        exp -= leading as i32;

        Expanded::Normal {
            negative: neg,
            mantissa: bits,
            exponent: exp
        }
    }
}

pub fn parse_bits_32(text: &str) -> u32 {
    let mut bits = 0u32;
    match parse_expanded(text) {
        Expanded::Normal {negative, mantissa, exponent} => {
            if negative {
                bits |= 0x8000_0000;
            }

            let exponent = exponent - 1;

            let mut mant = mantissa;
            if exponent > 127 {
                bits |= 0xff << 23;
            } else if exponent < -126 {
                let shift = 32 + 9 + min(24, -126 - exponent) - 1;
                if shift < 64 {
                    mant >>= shift;
                } else {
                    mant = 0;
                }
                bits |= mant as u32;
            } else {
                mant <<= 1;
                let leftover = mant << (32 - 9);
                mant >>= 32 + 9;
                bits |= mant as u32;
                if leftover > 0x8000_0000_0000_0000 ||
                    (leftover == 0x8000_0000_0000_0000 && (bits & 1) == 1) {
                    // round up
                    bits += 1;
                }
                bits |= ((exponent + 127) as u32) << 23;
            }
        }
        Expanded::Zero {negative} => {
            if negative {
                bits |= 0x8000_0000;
            }
        }
        Expanded::Infinite {negative} => {
            if negative {
                bits |= 0x8000_0000;
            }
            bits |= 0xff << 23;
        }
        Expanded::Nan {negative} => {
            if negative {
                bits |= 0x8000_0000;
            }
            bits |= 0xff << 23;
            bits |= 1;
        }
    }

    bits
}

pub fn parse_bits_64(text: &str) -> u64 {
    let mut bits = 0u64;
    match parse_expanded(text) {
        Expanded::Normal {negative, mantissa, exponent} => {
            if negative {
                bits |= 0x8000_0000_0000_0000;
            }

            let exponent = exponent - 1;

            let mut mant = mantissa;
            if exponent > 1023 {
                bits |= 0xff << 23;
            } else if exponent < -1022 {
                mant >>= 12 + min(52, -1022 - exponent) - 1;
                bits |= mant;
            } else {
                mant <<= 1;
                mant >>= 12;
                bits |= mant;
                bits |= ((exponent + 1023) as u64) << 52;
            }
        }
        Expanded::Zero {negative} => {
            if negative {
                bits |= 0x8000_0000_0000_0000;
            }
        }
        Expanded::Infinite {negative} => {
            if negative {
                bits |= 0x8000_0000_0000_0000;
            }
            bits |= 0xff << 52;
        }
        Expanded::Nan {negative} => {
            if negative {
                bits |= 0x8000_0000_0000_0000;
            }
            bits |= 0xff << 52;
            bits |= 1;
        }
    }

    bits
}

macro_rules! assert_bin_eq {
    ($a:expr, $b:expr) => {
        let a = $a;
        let b = $b;
        if a != b {
            panic!("assertion failed: `(left == right)` (left: `0b{:b}`, right: `0b{:b}`)", a, b);
        }
    }
}

#[cfg(test)]
mod tests {
    use hexfloat::parse_bits_32;
    use hexfloat::parse_bits_64;

    #[test]
    fn test_parse() {
        assert_bin_eq!(parse_bits_32("-0x0p+0"), 0x80000000);
        assert_bin_eq!(parse_bits_64("-0x0p+0"), 0x8000000000000000);
        assert_bin_eq!(parse_bits_32("-0x1.000002p-126"), 0x80800001);
        assert_bin_eq!(parse_bits_64("-0x1.000002p-126"), 0xb810000020000000);
        assert_bin_eq!(parse_bits_32("-0x1.3bd3cep+5"), 0xc21de9e7);
        assert_bin_eq!(parse_bits_64("-0x1.3bd3cep+5"), 0xc043bd3ce0000000);
        assert_bin_eq!(parse_bits_32("-0x1.45f304p+125"), 0xfe22f982);
        assert_bin_eq!(parse_bits_64("-0x1.45f304p+125"), 0xc7c45f3040000000);
        assert_bin_eq!(parse_bits_32("-0x1.45f306p-3"), 0xbe22f983);
        assert_bin_eq!(parse_bits_64("-0x1.45f306p-3"), 0xbfc45f3060000000);
        assert_bin_eq!(parse_bits_32("-0x1.45f306p-4"), 0xbda2f983);
        assert_bin_eq!(parse_bits_64("-0x1.45f306p-4"), 0xbfb45f3060000000);
        assert_bin_eq!(parse_bits_32("-0x1.45f3p-129"), 0x80145f30);
        assert_bin_eq!(parse_bits_64("-0x1.45f3p-129"), 0xb7e45f3000000000);
        assert_bin_eq!(parse_bits_32("-0x1.521fb6p+2"), 0xc0a90fdb);
        assert_bin_eq!(parse_bits_64("-0x1.521fb6p+2"), 0xc01521fb60000000);
        assert_bin_eq!(parse_bits_32("-0x1.721fb6p+2"), 0xc0b90fdb);
        assert_bin_eq!(parse_bits_64("-0x1.721fb6p+2"), 0xc01721fb60000000);
        assert_bin_eq!(parse_bits_32("-0x1.8p+0"), 0xbfc00000);
        assert_bin_eq!(parse_bits_64("-0x1.8p+0"), 0xbff8000000000000);
        assert_bin_eq!(parse_bits_32("-0x1.8p+2"), 0xc0c00000);
        assert_bin_eq!(parse_bits_64("-0x1.8p+2"), 0xc018000000000000);
        assert_bin_eq!(parse_bits_32("-0x1.8p-147"), 0x80000006);
        assert_bin_eq!(parse_bits_64("-0x1.8p-147"), 0xb6c8000000000000);
        assert_bin_eq!(parse_bits_32("-0x1.921fb6p+1"), 0xc0490fdb);
        assert_bin_eq!(parse_bits_64("-0x1.921fb6p+1"), 0xc00921fb60000000);
        assert_bin_eq!(parse_bits_32("-0x1.921fb6p+2"), 0xc0c90fdb);
        assert_bin_eq!(parse_bits_64("-0x1.921fb6p+2"), 0xc01921fb60000000);
        assert_bin_eq!(parse_bits_32("-0x1.921fb6p+3"), 0xc1490fdb);
        assert_bin_eq!(parse_bits_64("-0x1.921fb6p+3"), 0xc02921fb60000000);
        assert_bin_eq!(parse_bits_32("-0x1.921fb6p-124"), 0x81c90fdb);
        assert_bin_eq!(parse_bits_64("-0x1.921fb6p-124"), 0xb83921fb60000000);
        assert_bin_eq!(parse_bits_32("-0x1.921fb8p-126"), 0x80c90fdc);
        assert_bin_eq!(parse_bits_64("-0x1.921fb8p-126"), 0xb81921fb80000000);
        assert_bin_eq!(parse_bits_32("-0x1.b21fb6p+2"), 0xc0d90fdb);
        assert_bin_eq!(parse_bits_64("-0x1.b21fb6p+2"), 0xc01b21fb60000000);
        assert_bin_eq!(parse_bits_32("-0x1.cp+2"), 0xc0e00000);
        assert_bin_eq!(parse_bits_64("-0x1.cp+2"), 0xc01c000000000000);
        assert_bin_eq!(parse_bits_32("-0x1.d21fb6p+2"), 0xc0e90fdb);
        assert_bin_eq!(parse_bits_64("-0x1.d21fb6p+2"), 0xc01d21fb60000000);
        assert_bin_eq!(parse_bits_32("-0x1.fffffcp-127"), 0x807fffff);
        assert_bin_eq!(parse_bits_64("-0x1.fffffcp-127"), 0xb80fffffc0000000);
        assert_bin_eq!(parse_bits_32("-0x1.fffffep+1"), 0xc07fffff);
        assert_bin_eq!(parse_bits_64("-0x1.fffffep+1"), 0xc00fffffe0000000);
        assert_bin_eq!(parse_bits_32("-0x1.fffffep+126"), 0xfeffffff);
        assert_bin_eq!(parse_bits_64("-0x1.fffffep+126"), 0xc7dfffffe0000000);
        assert_bin_eq!(parse_bits_32("-0x1.fffffep+127"), 0xff7fffff);
        assert_bin_eq!(parse_bits_64("-0x1.fffffep+127"), 0xc7efffffe0000000);
        assert_bin_eq!(parse_bits_32("-0x1.fffffep-22"), 0xb4ffffff);
        assert_bin_eq!(parse_bits_64("-0x1.fffffep-22"), 0xbe9fffffe0000000);
        assert_bin_eq!(parse_bits_32("-0x1p+0"), 0xbf800000);
        assert_bin_eq!(parse_bits_64("-0x1p+0"), 0xbff0000000000000);
        assert_bin_eq!(parse_bits_32("-0x1p+1"), 0xc0000000);
        assert_bin_eq!(parse_bits_64("-0x1p+1"), 0xc000000000000000);
        assert_bin_eq!(parse_bits_32("-0x1p+125"), 0xfe000000);
        assert_bin_eq!(parse_bits_64("-0x1p+125"), 0xc7c0000000000000);
        assert_bin_eq!(parse_bits_32("-0x1p+126"), 0xfe800000);
        assert_bin_eq!(parse_bits_64("-0x1p+126"), 0xc7d0000000000000);
        assert_bin_eq!(parse_bits_32("-0x1p+23"), 0xcb000000);
        assert_bin_eq!(parse_bits_64("-0x1p+23"), 0xc160000000000000);
        assert_bin_eq!(parse_bits_32("-0x1p-1"), 0xbf000000);
        assert_bin_eq!(parse_bits_64("-0x1p-1"), 0xbfe0000000000000);
        assert_bin_eq!(parse_bits_32("-0x1p-125"), 0x81000000);
        assert_bin_eq!(parse_bits_64("-0x1p-125"), 0xb820000000000000);
        assert_bin_eq!(parse_bits_32("-0x1p-126"), 0x80800000);
        assert_bin_eq!(parse_bits_64("-0x1p-126"), 0xb810000000000000);
        assert_bin_eq!(parse_bits_32("-0x1p-127"), 0x80400000);
        assert_bin_eq!(parse_bits_64("-0x1p-127"), 0xb800000000000000);
        assert_bin_eq!(parse_bits_32("-0x1p-128"), 0x80200000);
        assert_bin_eq!(parse_bits_64("-0x1p-128"), 0xb7f0000000000000);
        assert_bin_eq!(parse_bits_32("-0x1p-129"), 0x80100000);
        assert_bin_eq!(parse_bits_64("-0x1p-129"), 0xb7e0000000000000);
        assert_bin_eq!(parse_bits_32("-0x1p-148"), 0x80000002);
        assert_bin_eq!(parse_bits_64("-0x1p-148"), 0xb6b0000000000000);
        assert_bin_eq!(parse_bits_32("-0x1p-149"), 0x80000001);
        assert_bin_eq!(parse_bits_64("-0x1p-149"), 0xb6a0000000000000);
        assert_bin_eq!(parse_bits_32("-0x1p-2"), 0xbe800000);
        assert_bin_eq!(parse_bits_64("-0x1p-2"), 0xbfd0000000000000);
        assert_bin_eq!(parse_bits_32("-0x1p-23"), 0xb4000000);
        assert_bin_eq!(parse_bits_64("-0x1p-23"), 0xbe80000000000000);
        assert_bin_eq!(parse_bits_32("0x0p+0"), 0x0);
        assert_bin_eq!(parse_bits_64("0x0p+0"), 0x0);
        assert_bin_eq!(parse_bits_32("0x1.000002p-126"), 0x800001);
        assert_bin_eq!(parse_bits_64("0x1.000002p-126"), 0x3810000020000000);
        assert_bin_eq!(parse_bits_32("0x1.3bd3cep+5"), 0x421de9e7);
        assert_bin_eq!(parse_bits_64("0x1.3bd3cep+5"), 0x4043bd3ce0000000);
        assert_bin_eq!(parse_bits_32("0x1.40d932p+1"), 0x40206c99);
        assert_bin_eq!(parse_bits_64("0x1.40d932p+1"), 0x40040d9320000000);
        assert_bin_eq!(parse_bits_32("0x1.45f304p+125"), 0x7e22f982);
        assert_bin_eq!(parse_bits_64("0x1.45f304p+125"), 0x47c45f3040000000);
        assert_bin_eq!(parse_bits_32("0x1.45f306p-3"), 0x3e22f983);
        assert_bin_eq!(parse_bits_64("0x1.45f306p-3"), 0x3fc45f3060000000);
        assert_bin_eq!(parse_bits_32("0x1.45f306p-4"), 0x3da2f983);
        assert_bin_eq!(parse_bits_64("0x1.45f306p-4"), 0x3fb45f3060000000);
        assert_bin_eq!(parse_bits_32("0x1.45f3p-129"), 0x145f30);
        assert_bin_eq!(parse_bits_64("0x1.45f3p-129"), 0x37e45f3000000000);
        assert_bin_eq!(parse_bits_32("0x1.521fb6p+2"), 0x40a90fdb);
        assert_bin_eq!(parse_bits_64("0x1.521fb6p+2"), 0x401521fb60000000);
        assert_bin_eq!(parse_bits_32("0x1.6a09e6p-1"), 0x3f3504f3);
        assert_bin_eq!(parse_bits_64("0x1.6a09e6p-1"), 0x3fe6a09e60000000);
        assert_bin_eq!(parse_bits_32("0x1.6a09e6p-75"), 0x1a3504f3);
        assert_bin_eq!(parse_bits_64("0x1.6a09e6p-75"), 0x3b46a09e60000000);
        assert_bin_eq!(parse_bits_32("0x1.721fb6p+2"), 0x40b90fdb);
        assert_bin_eq!(parse_bits_64("0x1.721fb6p+2"), 0x401721fb60000000);
        assert_bin_eq!(parse_bits_32("0x1.8p+0"), 0x3fc00000);
        assert_bin_eq!(parse_bits_64("0x1.8p+0"), 0x3ff8000000000000);
        assert_bin_eq!(parse_bits_32("0x1.8p+2"), 0x40c00000);
        assert_bin_eq!(parse_bits_64("0x1.8p+2"), 0x4018000000000000);
        assert_bin_eq!(parse_bits_32("0x1.8p-147"), 0x6);
        assert_bin_eq!(parse_bits_64("0x1.8p-147"), 0x36c8000000000000);
        assert_bin_eq!(parse_bits_32("0x1.921fb6p+1"), 0x40490fdb);
        assert_bin_eq!(parse_bits_64("0x1.921fb6p+1"), 0x400921fb60000000);
        assert_bin_eq!(parse_bits_32("0x1.921fb6p+2"), 0x40c90fdb);
        assert_bin_eq!(parse_bits_64("0x1.921fb6p+2"), 0x401921fb60000000);
        assert_bin_eq!(parse_bits_32("0x1.921fb6p+3"), 0x41490fdb);
        assert_bin_eq!(parse_bits_64("0x1.921fb6p+3"), 0x402921fb60000000);
        assert_bin_eq!(parse_bits_32("0x1.921fb6p-124"), 0x1c90fdb);
        assert_bin_eq!(parse_bits_64("0x1.921fb6p-124"), 0x383921fb60000000);
        assert_bin_eq!(parse_bits_32("0x1.921fb8p-126"), 0xc90fdc);
        assert_bin_eq!(parse_bits_64("0x1.921fb8p-126"), 0x381921fb80000000);
        assert_bin_eq!(parse_bits_32("0x1.b21fb6p+2"), 0x40d90fdb);
        assert_bin_eq!(parse_bits_64("0x1.b21fb6p+2"), 0x401b21fb60000000);
        assert_bin_eq!(parse_bits_32("0x1.cp+2"), 0x40e00000);
        assert_bin_eq!(parse_bits_64("0x1.cp+2"), 0x401c000000000000);
        assert_bin_eq!(parse_bits_32("0x1.d21fb6p+2"), 0x40e90fdb);
        assert_bin_eq!(parse_bits_64("0x1.d21fb6p+2"), 0x401d21fb60000000);
        assert_bin_eq!(parse_bits_32("0x1.fffffcp-127"), 0x7fffff);
        assert_bin_eq!(parse_bits_64("0x1.fffffcp-127"), 0x380fffffc0000000);
        assert_bin_eq!(parse_bits_32("0x1.fffffep+1"), 0x407fffff);
        assert_bin_eq!(parse_bits_64("0x1.fffffep+1"), 0x400fffffe0000000);
        assert_bin_eq!(parse_bits_32("0x1.fffffep+126"), 0x7effffff);
        assert_bin_eq!(parse_bits_64("0x1.fffffep+126"), 0x47dfffffe0000000);
        assert_bin_eq!(parse_bits_32("0x1.fffffep+127"), 0x7f7fffff);
        assert_bin_eq!(parse_bits_64("0x1.fffffep+127"), 0x47efffffe0000000);
        assert_bin_eq!(parse_bits_32("0x1.fffffep+63"), 0x5f7fffff);
        assert_bin_eq!(parse_bits_64("0x1.fffffep+63"), 0x43efffffe0000000);
        assert_bin_eq!(parse_bits_32("0x1.fffffep-22"), 0x34ffffff);
        assert_bin_eq!(parse_bits_64("0x1.fffffep-22"), 0x3e9fffffe0000000);
        assert_bin_eq!(parse_bits_32("0x1p+0"), 0x3f800000);
        assert_bin_eq!(parse_bits_64("0x1p+0"), 0x3ff0000000000000);
        assert_bin_eq!(parse_bits_32("0x1p+1"), 0x40000000);
        assert_bin_eq!(parse_bits_64("0x1p+1"), 0x4000000000000000);
        assert_bin_eq!(parse_bits_32("0x1p+125"), 0x7e000000);
        assert_bin_eq!(parse_bits_64("0x1p+125"), 0x47c0000000000000);
        assert_bin_eq!(parse_bits_32("0x1p+126"), 0x7e800000);
        assert_bin_eq!(parse_bits_64("0x1p+126"), 0x47d0000000000000);
        assert_bin_eq!(parse_bits_32("0x1p+23"), 0x4b000000);
        assert_bin_eq!(parse_bits_64("0x1p+23"), 0x4160000000000000);
        assert_bin_eq!(parse_bits_32("0x1p-1"), 0x3f000000);
        assert_bin_eq!(parse_bits_64("0x1p-1"), 0x3fe0000000000000);
        assert_bin_eq!(parse_bits_32("0x1p-125"), 0x1000000);
        assert_bin_eq!(parse_bits_64("0x1p-125"), 0x3820000000000000);
        assert_bin_eq!(parse_bits_32("0x1p-126"), 0x800000);
        assert_bin_eq!(parse_bits_64("0x1p-126"), 0x3810000000000000);
        assert_bin_eq!(parse_bits_32("0x1p-127"), 0x400000);
        assert_bin_eq!(parse_bits_64("0x1p-127"), 0x3800000000000000);
        assert_bin_eq!(parse_bits_32("0x1p-128"), 0x200000);
        assert_bin_eq!(parse_bits_64("0x1p-128"), 0x37f0000000000000);
        assert_bin_eq!(parse_bits_32("0x1p-129"), 0x100000);
        assert_bin_eq!(parse_bits_64("0x1p-129"), 0x37e0000000000000);
        assert_bin_eq!(parse_bits_32("0x1p-148"), 0x2);
        assert_bin_eq!(parse_bits_64("0x1p-148"), 0x36b0000000000000);
        assert_bin_eq!(parse_bits_32("0x1p-149"), 0x1);
        assert_bin_eq!(parse_bits_64("0x1p-149"), 0x36a0000000000000);
        assert_bin_eq!(parse_bits_32("0x1p-2"), 0x3e800000);
        assert_bin_eq!(parse_bits_64("0x1p-2"), 0x3fd0000000000000);
        assert_bin_eq!(parse_bits_32("0x1p-23"), 0x34000000);
        assert_bin_eq!(parse_bits_64("0x1p-23"), 0x3e80000000000000);
        assert_bin_eq!(parse_bits_32("0x1p-63"), 0x20000000);
        assert_bin_eq!(parse_bits_64("0x1p-63"), 0x3c00000000000000);
        assert_bin_eq!(parse_bits_32("-0x0.0000000000001p-1022"), 0x80000000);
        assert_bin_eq!(parse_bits_64("-0x0.0000000000001p-1022"), 0x8000000000000001);
        assert_bin_eq!(parse_bits_32("-0x0.0000000000002p-1022"), 0x80000000);
        assert_bin_eq!(parse_bits_64("-0x0.0000000000002p-1022"), 0x8000000000000002);
        assert_bin_eq!(parse_bits_32("-0x0.0000000000006p-1022"), 0x80000000);
        assert_bin_eq!(parse_bits_64("-0x0.0000000000006p-1022"), 0x8000000000000006);
        assert_bin_eq!(parse_bits_32("-0x0.28be60db9391p-1022"), 0x80000000);
        assert_bin_eq!(parse_bits_64("-0x0.28be60db9391p-1022"), 0x80028be60db93910);
        assert_bin_eq!(parse_bits_32("-0x0.2p-1022"), 0x80000000);
        assert_bin_eq!(parse_bits_64("-0x0.2p-1022"), 0x8002000000000000);
        assert_bin_eq!(parse_bits_32("-0x0.4p-1022"), 0x80000000);
        assert_bin_eq!(parse_bits_64("-0x0.4p-1022"), 0x8004000000000000);
        assert_bin_eq!(parse_bits_32("-0x0.8p-1022"), 0x80000000);
        assert_bin_eq!(parse_bits_64("-0x0.8p-1022"), 0x8008000000000000);
        assert_bin_eq!(parse_bits_32("-0x0.fffffffffffffp-1022"), 0x80000000);
        assert_bin_eq!(parse_bits_64("-0x0.fffffffffffffp-1022"), 0x800fffffffffffff);
        assert_bin_eq!(parse_bits_32("-0x0p+0"), 0x80000000);
        assert_bin_eq!(parse_bits_64("-0x0p+0"), 0x8000000000000000);
        assert_bin_eq!(parse_bits_32("-0x1.0000000000001p-1022"), 0x80000000);
        assert_bin_eq!(parse_bits_64("-0x1.0000000000001p-1022"), 0x8010000000000001);
        assert_bin_eq!(parse_bits_32("-0x1.3bd3cc9be45dep+5"), 0xc21de9e6);
        assert_bin_eq!(parse_bits_64("-0x1.3bd3cc9be45dep+5"), 0xc043bd3cc9be45de);
        assert_bin_eq!(parse_bits_32("-0x1.45f306dc9c882p+1021"), 0xff800000);
        assert_bin_eq!(parse_bits_64("-0x1.45f306dc9c882p+1021"), 0xffc45f306dc9c882);
        assert_bin_eq!(parse_bits_32("-0x1.45f306dc9c883p-3"), 0xbe22f983);
        assert_bin_eq!(parse_bits_64("-0x1.45f306dc9c883p-3"), 0xbfc45f306dc9c883);
        assert_bin_eq!(parse_bits_32("-0x1.45f306dc9c883p-4"), 0xbda2f983);
        assert_bin_eq!(parse_bits_64("-0x1.45f306dc9c883p-4"), 0xbfb45f306dc9c883);
        assert_bin_eq!(parse_bits_32("-0x1.521fb54442d18p+2"), 0xc0a90fdb);
        assert_bin_eq!(parse_bits_64("-0x1.521fb54442d18p+2"), 0xc01521fb54442d18);
        assert_bin_eq!(parse_bits_32("-0x1.721fb54442d18p+2"), 0xc0b90fdb);
        assert_bin_eq!(parse_bits_64("-0x1.721fb54442d18p+2"), 0xc01721fb54442d18);
        assert_bin_eq!(parse_bits_32("-0x1.8p+0"), 0xbfc00000);
        assert_bin_eq!(parse_bits_64("-0x1.8p+0"), 0xbff8000000000000);
        assert_bin_eq!(parse_bits_32("-0x1.8p+2"), 0xc0c00000);
        assert_bin_eq!(parse_bits_64("-0x1.8p+2"), 0xc018000000000000);
        assert_bin_eq!(parse_bits_32("-0x1.921fb54442d18p+1"), 0xc0490fdb);
        assert_bin_eq!(parse_bits_64("-0x1.921fb54442d18p+1"), 0xc00921fb54442d18);
        assert_bin_eq!(parse_bits_32("-0x1.921fb54442d18p+2"), 0xc0c90fdb);
        assert_bin_eq!(parse_bits_64("-0x1.921fb54442d18p+2"), 0xc01921fb54442d18);
        assert_bin_eq!(parse_bits_32("-0x1.921fb54442d18p+3"), 0xc1490fdb);
        assert_bin_eq!(parse_bits_64("-0x1.921fb54442d18p+3"), 0xc02921fb54442d18);
        assert_bin_eq!(parse_bits_32("-0x1.921fb54442d18p-1020"), 0x80000000);
        assert_bin_eq!(parse_bits_64("-0x1.921fb54442d18p-1020"), 0x803921fb54442d18);
        assert_bin_eq!(parse_bits_32("-0x1.921fb54442d19p-1022"), 0x80000000);
        assert_bin_eq!(parse_bits_64("-0x1.921fb54442d19p-1022"), 0x801921fb54442d19);
        assert_bin_eq!(parse_bits_32("-0x1.b21fb54442d18p+2"), 0xc0d90fdb);
        assert_bin_eq!(parse_bits_64("-0x1.b21fb54442d18p+2"), 0xc01b21fb54442d18);
        assert_bin_eq!(parse_bits_32("-0x1.cp+2"), 0xc0e00000);
        assert_bin_eq!(parse_bits_64("-0x1.cp+2"), 0xc01c000000000000);
        assert_bin_eq!(parse_bits_32("-0x1.d21fb54442d18p+2"), 0xc0e90fdb);
        assert_bin_eq!(parse_bits_64("-0x1.d21fb54442d18p+2"), 0xc01d21fb54442d18);
        assert_bin_eq!(parse_bits_32("-0x1.fffffffffffffp+1"), 0xc0800000);
        assert_bin_eq!(parse_bits_64("-0x1.fffffffffffffp+1"), 0xc00fffffffffffff);
        assert_bin_eq!(parse_bits_32("-0x1.fffffffffffffp+1022"), 0xff800000);
        assert_bin_eq!(parse_bits_64("-0x1.fffffffffffffp+1022"), 0xffdfffffffffffff);
        assert_bin_eq!(parse_bits_32("-0x1.fffffffffffffp+1023"), 0xff800000);
        assert_bin_eq!(parse_bits_64("-0x1.fffffffffffffp+1023"), 0xffefffffffffffff);
        assert_bin_eq!(parse_bits_32("-0x1.fffffffffffffp-51"), 0xa6800000);
        assert_bin_eq!(parse_bits_64("-0x1.fffffffffffffp-51"), 0xbccfffffffffffff);
        assert_bin_eq!(parse_bits_32("-0x1p+0"), 0xbf800000);
        assert_bin_eq!(parse_bits_64("-0x1p+0"), 0xbff0000000000000);
        assert_bin_eq!(parse_bits_32("-0x1p+1"), 0xc0000000);
        assert_bin_eq!(parse_bits_64("-0x1p+1"), 0xc000000000000000);
        assert_bin_eq!(parse_bits_32("-0x1p+1021"), 0xff800000);
        assert_bin_eq!(parse_bits_64("-0x1p+1021"), 0xffc0000000000000);
        assert_bin_eq!(parse_bits_32("-0x1p+1022"), 0xff800000);
        assert_bin_eq!(parse_bits_64("-0x1p+1022"), 0xffd0000000000000);
        assert_bin_eq!(parse_bits_32("-0x1p+52"), 0xd9800000);
        assert_bin_eq!(parse_bits_64("-0x1p+52"), 0xc330000000000000);
        assert_bin_eq!(parse_bits_32("-0x1p-1"), 0xbf000000);
        assert_bin_eq!(parse_bits_64("-0x1p-1"), 0xbfe0000000000000);
        assert_bin_eq!(parse_bits_32("-0x1p-1021"), 0x80000000);
        assert_bin_eq!(parse_bits_64("-0x1p-1021"), 0x8020000000000000);
        assert_bin_eq!(parse_bits_32("-0x1p-1022"), 0x80000000);
        assert_bin_eq!(parse_bits_64("-0x1p-1022"), 0x8010000000000000);
        assert_bin_eq!(parse_bits_32("-0x1p-2"), 0xbe800000);
        assert_bin_eq!(parse_bits_64("-0x1p-2"), 0xbfd0000000000000);
        assert_bin_eq!(parse_bits_32("-0x1p-52"), 0xa5800000);
        assert_bin_eq!(parse_bits_64("-0x1p-52"), 0xbcb0000000000000);
        assert_bin_eq!(parse_bits_32("0x0.0000000000001p-1022"), 0x0);
        assert_bin_eq!(parse_bits_64("0x0.0000000000001p-1022"), 0x1);
        assert_bin_eq!(parse_bits_32("0x0.0000000000002p-1022"), 0x0);
        assert_bin_eq!(parse_bits_64("0x0.0000000000002p-1022"), 0x2);
        assert_bin_eq!(parse_bits_32("0x0.0000000000006p-1022"), 0x0);
        assert_bin_eq!(parse_bits_64("0x0.0000000000006p-1022"), 0x6);
        assert_bin_eq!(parse_bits_32("0x0.28be60db9391p-1022"), 0x0);
        assert_bin_eq!(parse_bits_64("0x0.28be60db9391p-1022"), 0x28be60db93910);
        assert_bin_eq!(parse_bits_32("0x0.2p-1022"), 0x0);
        assert_bin_eq!(parse_bits_64("0x0.2p-1022"), 0x2000000000000);
        assert_bin_eq!(parse_bits_32("0x0.4p-1022"), 0x0);
        assert_bin_eq!(parse_bits_64("0x0.4p-1022"), 0x4000000000000);
        assert_bin_eq!(parse_bits_32("0x0.8p-1022"), 0x0);
        assert_bin_eq!(parse_bits_64("0x0.8p-1022"), 0x8000000000000);
        assert_bin_eq!(parse_bits_32("0x0.fffffffffffffp-1022"), 0x0);
        assert_bin_eq!(parse_bits_64("0x0.fffffffffffffp-1022"), 0xfffffffffffff);
        assert_bin_eq!(parse_bits_32("0x0p+0"), 0x0);
        assert_bin_eq!(parse_bits_64("0x0p+0"), 0x0);
        assert_bin_eq!(parse_bits_32("0x1.0000000000001p-1022"), 0x0);
        assert_bin_eq!(parse_bits_64("0x1.0000000000001p-1022"), 0x10000000000001);
        assert_bin_eq!(parse_bits_32("0x1.3bd3cc9be45dep+5"), 0x421de9e6);
        assert_bin_eq!(parse_bits_64("0x1.3bd3cc9be45dep+5"), 0x4043bd3cc9be45de);
        assert_bin_eq!(parse_bits_32("0x1.40d931ff62705p+1"), 0x40206c99);
        assert_bin_eq!(parse_bits_64("0x1.40d931ff62705p+1"), 0x40040d931ff62705);
        assert_bin_eq!(parse_bits_32("0x1.45f306dc9c882p+1021"), 0x7f800000);
        assert_bin_eq!(parse_bits_64("0x1.45f306dc9c882p+1021"), 0x7fc45f306dc9c882);
        assert_bin_eq!(parse_bits_32("0x1.45f306dc9c883p-3"), 0x3e22f983);
        assert_bin_eq!(parse_bits_64("0x1.45f306dc9c883p-3"), 0x3fc45f306dc9c883);
        assert_bin_eq!(parse_bits_32("0x1.45f306dc9c883p-4"), 0x3da2f983);
        assert_bin_eq!(parse_bits_64("0x1.45f306dc9c883p-4"), 0x3fb45f306dc9c883);
        assert_bin_eq!(parse_bits_32("0x1.521fb54442d18p+2"), 0x40a90fdb);
        assert_bin_eq!(parse_bits_64("0x1.521fb54442d18p+2"), 0x401521fb54442d18);
        assert_bin_eq!(parse_bits_32("0x1.6a09e667f3bcdp-1"), 0x3f3504f3);
        assert_bin_eq!(parse_bits_64("0x1.6a09e667f3bcdp-1"), 0x3fe6a09e667f3bcd);
        assert_bin_eq!(parse_bits_32("0x1.721fb54442d18p+2"), 0x40b90fdb);
        assert_bin_eq!(parse_bits_64("0x1.721fb54442d18p+2"), 0x401721fb54442d18);
        assert_bin_eq!(parse_bits_32("0x1.8p+0"), 0x3fc00000);
        assert_bin_eq!(parse_bits_64("0x1.8p+0"), 0x3ff8000000000000);
        assert_bin_eq!(parse_bits_32("0x1.8p+2"), 0x40c00000);
        assert_bin_eq!(parse_bits_64("0x1.8p+2"), 0x4018000000000000);
        assert_bin_eq!(parse_bits_32("0x1.921fb54442d18p+1"), 0x40490fdb);
        assert_bin_eq!(parse_bits_64("0x1.921fb54442d18p+1"), 0x400921fb54442d18);
        assert_bin_eq!(parse_bits_32("0x1.921fb54442d18p+2"), 0x40c90fdb);
        assert_bin_eq!(parse_bits_64("0x1.921fb54442d18p+2"), 0x401921fb54442d18);
        assert_bin_eq!(parse_bits_32("0x1.921fb54442d18p+3"), 0x41490fdb);
        assert_bin_eq!(parse_bits_64("0x1.921fb54442d18p+3"), 0x402921fb54442d18);
        assert_bin_eq!(parse_bits_32("0x1.921fb54442d18p-1020"), 0x0);
        assert_bin_eq!(parse_bits_64("0x1.921fb54442d18p-1020"), 0x3921fb54442d18);
        assert_bin_eq!(parse_bits_32("0x1.921fb54442d19p-1022"), 0x0);
        assert_bin_eq!(parse_bits_64("0x1.921fb54442d19p-1022"), 0x1921fb54442d19);
        assert_bin_eq!(parse_bits_32("0x1.b21fb54442d18p+2"), 0x40d90fdb);
        assert_bin_eq!(parse_bits_64("0x1.b21fb54442d18p+2"), 0x401b21fb54442d18);
        assert_bin_eq!(parse_bits_32("0x1.cp+2"), 0x40e00000);
        assert_bin_eq!(parse_bits_64("0x1.cp+2"), 0x401c000000000000);
        assert_bin_eq!(parse_bits_32("0x1.d21fb54442d18p+2"), 0x40e90fdb);
        assert_bin_eq!(parse_bits_64("0x1.d21fb54442d18p+2"), 0x401d21fb54442d18);
        assert_bin_eq!(parse_bits_32("0x1.fffffffffffffp+1"), 0x40800000);
        assert_bin_eq!(parse_bits_64("0x1.fffffffffffffp+1"), 0x400fffffffffffff);
        assert_bin_eq!(parse_bits_32("0x1.fffffffffffffp+1022"), 0x7f800000);
        assert_bin_eq!(parse_bits_64("0x1.fffffffffffffp+1022"), 0x7fdfffffffffffff);
        assert_bin_eq!(parse_bits_32("0x1.fffffffffffffp+1023"), 0x7f800000);
        assert_bin_eq!(parse_bits_64("0x1.fffffffffffffp+1023"), 0x7fefffffffffffff);
        assert_bin_eq!(parse_bits_32("0x1.fffffffffffffp+511"), 0x7f800000);
        assert_bin_eq!(parse_bits_64("0x1.fffffffffffffp+511"), 0x5fefffffffffffff);
        assert_bin_eq!(parse_bits_32("0x1.fffffffffffffp-51"), 0x26800000);
        assert_bin_eq!(parse_bits_64("0x1.fffffffffffffp-51"), 0x3ccfffffffffffff);
        assert_bin_eq!(parse_bits_32("0x1p+0"), 0x3f800000);
        assert_bin_eq!(parse_bits_64("0x1p+0"), 0x3ff0000000000000);
        assert_bin_eq!(parse_bits_32("0x1p+1"), 0x40000000);
        assert_bin_eq!(parse_bits_64("0x1p+1"), 0x4000000000000000);
        assert_bin_eq!(parse_bits_32("0x1p+1021"), 0x7f800000);
        assert_bin_eq!(parse_bits_64("0x1p+1021"), 0x7fc0000000000000);
        assert_bin_eq!(parse_bits_32("0x1p+1022"), 0x7f800000);
        assert_bin_eq!(parse_bits_64("0x1p+1022"), 0x7fd0000000000000);
        assert_bin_eq!(parse_bits_32("0x1p+52"), 0x59800000);
        assert_bin_eq!(parse_bits_64("0x1p+52"), 0x4330000000000000);
        assert_bin_eq!(parse_bits_32("0x1p-1"), 0x3f000000);
        assert_bin_eq!(parse_bits_64("0x1p-1"), 0x3fe0000000000000);
        assert_bin_eq!(parse_bits_32("0x1p-1021"), 0x0);
        assert_bin_eq!(parse_bits_64("0x1p-1021"), 0x20000000000000);
        assert_bin_eq!(parse_bits_32("0x1p-1022"), 0x0);
        assert_bin_eq!(parse_bits_64("0x1p-1022"), 0x10000000000000);
        assert_bin_eq!(parse_bits_32("0x1p-2"), 0x3e800000);
        assert_bin_eq!(parse_bits_64("0x1p-2"), 0x3fd0000000000000);
        assert_bin_eq!(parse_bits_32("0x1p-511"), 0x0);
        assert_bin_eq!(parse_bits_64("0x1p-511"), 0x2000000000000000);
        assert_bin_eq!(parse_bits_32("0x1p-52"), 0x25800000);
        assert_bin_eq!(parse_bits_64("0x1p-52"), 0x3cb0000000000000);
        assert_bin_eq!(parse_bits_32("0x1p-537"), 0x0);
        assert_bin_eq!(parse_bits_64("0x1p-537"), 0x1e60000000000000);
    }
}
