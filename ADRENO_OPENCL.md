# Experimental Adreno (OpenCL) GPU backend for Android

Yekta's fork of fllama can build llama.cpp's OpenCL backend so mid-range
Qualcomm Adreno phones (the bulk of the Turkish Android market) run inference
on the GPU instead of CPU-only. It is **OFF by default** and gated behind a
CMake option because it needs headers/libraries vendored into the NDK sysroot
and must be validated on real hardware — a silent all-layer offload to a weak
GPU can thrash or crash.

## Status

- llama.cpp already vendors the OpenCL backend
  (`ggml/src/ggml-opencl`, docs at `src/llama.cpp/docs/backend/OPENCL.md`).
- The fork's `src/CMakeLists.txt` exposes `FLLAMA_ANDROID_OPENCL` (default
  OFF). When ON it sets `GGML_OPENCL`, `GGML_OPENCL_EMBED_KERNELS`,
  `GGML_OPENCL_USE_ADRENO_KERNELS`.
- App side: `AppSettings.androidGpu` (Settings → experimental) drives
  `GenerationConfig.androidGpu`; `FllamaEngine._effectiveGpuLayers` then
  offloads the requested layers on Android instead of pinning to 0.
- **Not yet validated on a device.** The remaining work is below.

## Remaining work to ship it

1. **Vendor OpenCL headers + ICD loader** for the Android build:
   - `KhronosGroup/OpenCL-Headers` and `KhronosGroup/OpenCL-ICD-Loader`,
     cross-compiled for `arm64-v8a` (API 24+), installed where the NDK
     toolchain finds them (pass `-DCMAKE_PREFIX_PATH` or add to the sysroot).
   - Only `arm64-v8a` is targeted; other ABIs stay CPU-only.
2. **Build** with the flag, e.g. from the app:
   `flutter build apk -Pfllama.opencl=ON` (wire the Gradle property through to
   the CMake `-DFLLAMA_ANDROID_OPENCL=ON` arg in `android/app/build.gradle`),
   or set it directly in a local CMake invocation.
3. **Validate on Adreno hardware** (verified upstream on Adreno 750 /
   Snapdragon 8 Gen 3 and Adreno 830 / 8 Elite): confirm correctness (no
   degenerate output vs the CPU build) and measure tok/s. Only ship the toggle
   as non-experimental if the gain is real and stable.
4. **Gate by device**: query the GPU (Adreno model / driver) on Android and
   only surface the toggle where OpenCL is actually present, matching the
   "honest device fit" principle — default OFF everywhere until measured.

## Why gated, not default-on

Q4_K_M/Q5/Q8/MXFP4 now work on the OpenCL path (2025 they didn't), but Adreno
driver quality varies wildly across mid-range SoCs. Default-CPU keeps every
device working; the GPU path is opt-in until per-device measurement says it
helps.
