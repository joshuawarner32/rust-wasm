use std::str;

pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Reader<'a> {
        Reader {
            buf: buf,
            pos: 0
        }
    }

    pub fn skip(&mut self, count: usize) {
        self.pos += count;
        if self.pos > self.buf.len() {
            panic!();
        }
    }

    pub fn peek_u8(&self) -> u8 {
        self.buf[self.pos]
    }

    pub fn read_u8(&mut self) -> u8 {
        let res = self.buf[self.pos];
        self.pos += 1;
        res
    }

    pub fn read_u32(&mut self) -> u32 {
        ((self.read_u8() as u32) << 0*8) +
        ((self.read_u8() as u32) << 1*8) +
        ((self.read_u8() as u32) << 2*8) +
        ((self.read_u8() as u32) << 3*8)
    }

    pub fn read_u64(&mut self) -> u64 {
        ((self.read_u8() as u64) << 0*8) +
        ((self.read_u8() as u64) << 1*8) +
        ((self.read_u8() as u64) << 2*8) +
        ((self.read_u8() as u64) << 3*8) +
        ((self.read_u8() as u64) << (4+0)*8) +
        ((self.read_u8() as u64) << (4+1)*8) +
        ((self.read_u8() as u64) << (4+2)*8) +
        ((self.read_u8() as u64) << (4+3)*8)
    }

    pub fn read_var_u32(&mut self) -> u32 {
        let mut res = 0;
        let mut shift = 0;
        loop {
            let b = self.read_u8() as u32;
            res |= (b & 0x7f) << shift;
            shift += 7;
            if (b >> 7) == 0 {
                break;
            }
        }
        res
    }

    pub fn read_var_u64(&mut self) -> u64 {
        let mut res = 0;
        let mut shift = 0;
        loop {
            let b = self.read_u8() as u64;
            res |= (b & 0x7f) << shift;
            shift += 7;
            if (b >> 7) == 0 {
                break;
            }
        }
        res
    }

    pub fn read_var_i32(&mut self) -> i32 {
        let mut res = 0i32;
        let mut shift = 0;
        loop {
            let b = self.read_u8() as i32;
            res |= (b & 0x7f) << shift;
            shift += 7;
            if (b & 0x80) == 0 {
                if shift < 31 && (b & 0x40) != 0 {
                    res |= -(1 << shift);
                }
                break;
            }
        }
        res
    }

    pub fn read_bytes_with_len(&mut self, data_len: usize) -> &'a [u8] {
        let data = &self.buf[self.pos..self.pos + data_len];
        self.pos += data_len;

        data
    }

    pub fn read_bytes(&mut self) -> &'a [u8] {
        let data_len = self.read_var_u32() as usize;
        self.read_bytes_with_len(data_len)
    }

    pub fn read_str(&mut self) -> &'a str {
        str::from_utf8(self.read_bytes()).unwrap()
    }

    pub fn into_remaining(self) -> &'a [u8] {
        &self.buf[self.pos..]
    }

    pub fn at_eof(&self) -> bool {
        self.pos >= self.buf.len()
    }
}
