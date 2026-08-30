"""Quick loadability test: hipModuleLoadData on an offline-compiled hsaco."""
import sys, ctypes
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import wmma_gemm_v2 as W  # loads amdhip64 + hiprtc with argtypes

HIP = W.HIP
path = sys.argv[1] if len(sys.argv) > 1 else r"B:\git\cubbyllm-lite\kernels\_v19_rtc.hsaco"
data = open(path, "rb").read()
print(f"{path}: {len(data)} bytes")
buf = ctypes.create_string_buffer(data)
mod = ctypes.c_void_p()
rc_load = HIP.hipModuleLoadData(ctypes.byref(mod), buf)
print(f"hipModuleLoadData rc={rc_load}")
fn = ctypes.c_void_p()
rc_fn = HIP.hipModuleGetFunction(ctypes.byref(fn), mod, b"moe_v19")
print(f"hipModuleGetFunction('moe_v19') rc={rc_fn} handle={fn.value}")
assert rc_load == 0 and rc_fn == 0 and fn.value, "FAILED"
print("PASS: offline hsaco loads and exports moe_v19")
