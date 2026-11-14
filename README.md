# glass-shader-rs

Glass effects: host GPU passes, shared parameters and shader libraries.

Extracted from the Civ3Renderer workspace with file history preserved.

Crates:

- `glass`
- `glass-shared`
- `glass-shaders`
- `glass-test-spirv`
- `glass-test-shaders`

Glass effects are independent of the viewer. Tests compile their own tile-and-glass shader module; the application links its production module.
