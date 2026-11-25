//! Build script for the `skia_rs` Python extension module.

fn main() {
    // An extension module must not link libpython: the interpreter that loads
    // it provides the `Py*` symbols at import time. Linux tolerates undefined
    // symbols in a cdylib and Windows links the `python3.lib` import library,
    // but macOS `ld` rejects them unless told to resolve them dynamically
    // (`-undefined dynamic_lookup`). This emits exactly that on macOS and is a
    // no-op elsewhere.
    pyo3_build_config::add_extension_module_link_args();
}
