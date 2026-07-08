use burn_onnx::ModelGen;
use std::{env, fs, path::Path};

// Basename of the artifacts produced by ../../get_qwen_onnx.py.
const SAFE_NAME: &str = "qwen3-0_6b";
const SEQ_LENGTH: usize = 32;
const VOCAB_SIZE: usize = 151936;

fn main() {
    let pkg_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let artifacts = Path::new(&pkg_dir).join("artifacts");
    let onnx_path = artifacts.join(format!("{SAFE_NAME}_opset16.onnx"));
    let test_data_path = artifacts.join(format!("{SAFE_NAME}_test_data.pt"));

    println!("cargo:rerun-if-changed={}", onnx_path.display());
    println!("cargo:rerun-if-changed=build.rs");

    if !onnx_path.exists() {
        panic!(
            "ONNX model not found at {}.\n\nExport it first:\n    ./bench/get_qwen_onnx.py\n",
            onnx_path.display()
        );
    }

    // Generate `$OUT_DIR/model/<name>_opset16.rs` (defines `Model`) and the
    // companion `<name>_opset16.bpk` burnpack weight file.
    ModelGen::new()
        .input(onnx_path.to_str().unwrap())
        .out_dir("model/")
        .run_from_script();

    let out_dir = env::var("OUT_DIR").unwrap();

    // burn-onnx bakes bool constants as `BoolStore::Native`. The MLX backend
    // (our ONNX backend) accepts Native bool, but rewriting the init to U32 is
    // harmless and keeps the generated source compatible with cubecl's U8/U32
    // requirement should the model ever be constructed without loading weights.
    // NOTE: the cubecl Metal/wgpu backends still cannot *load* this model — the
    // bool constant is persisted in the .bpk as Native and cubecl's
    // `bool_from_data` panics on Native at load time. That is why the ONNX case
    // runs on MLX (see README methodology).
    let gen_rs = Path::new(&out_dir)
        .join("model")
        .join(format!("{SAFE_NAME}_opset16.rs"));
    let src = fs::read_to_string(&gen_rs).expect("Failed to read generated model");
    let patched = src.replace("BoolStore::Native", "BoolStore::U32");
    fs::write(&gen_rs, patched).expect("Failed to write patched model");
    let info_path = Path::new(&out_dir).join("model_info.rs");
    fs::write(
        &info_path,
        format!(
            r#"pub const SEQ_LENGTH: usize = {seq};
pub const VOCAB_SIZE: usize = {vocab};
pub const WEIGHTS_PATH: &str = concat!(env!("OUT_DIR"), "/model/{name}_opset16.bpk");
pub const TEST_DATA_PATH: &str = "{test_data}";

pub mod qwen_model {{
    include!(concat!(env!("OUT_DIR"), "/model/{name}_opset16.rs"));
}}
"#,
            seq = SEQ_LENGTH,
            vocab = VOCAB_SIZE,
            name = SAFE_NAME,
            test_data = test_data_path.display(),
        ),
    )
    .expect("Failed to write model_info.rs");
}
