
mod reader;

mod types;
mod module;
pub mod ops;
mod interp;
mod testcase;
mod sexpr;
mod hexfloat;

pub use types::Dynamic;
pub use module::{Module, FunctionIndex, ExportIndex, FunctionType};
pub use interp::{Instance, BoundInstance, InterpResult, Memory};
pub use testcase::TestCase;

#[cfg(test)]
mod test {
    use std::fs::{self, File};
    use std::path::Path;
    use std::io::Read;

    // failing due to stack size limits
    // #[test]
    fn run_all_wast_tests() {
        for entry in fs::read_dir(Path::new("test")).unwrap() {
            let entry = entry.unwrap();
            println!("running {:?}", entry.path());

            let mut contents = Vec::new();
            File::open(&entry.path()).expect("readable file").read_to_end(&mut contents).expect("read succeeds");

            let test = ::testcase::TestCase::parse(&contents);

            test.run_all();

            println!("\n\n\n\n");
        }
    }
}
