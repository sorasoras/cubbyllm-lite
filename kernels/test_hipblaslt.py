import sys, os, ctypes
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import numpy as np

SDK = r"B:\git\rocm79\bin"
os.add_dll_directory(SDK)

# 1) load the 7.9 runtime + hipblaslt
hip = ctypes.CDLL(os.path.join(SDK, "amdhip64_7.dll"))
lt = ctypes.CDLL(os.path.join(SDK, "libhipblaslt.dll"))
print("DLLs loaded: amdhip64_7 + libhipblaslt")

# 2) hipblasLtCreate
handle = ctypes.c_void_p()
rc = lt.hipblasLtCreate(ctypes.byref(handle))
print(f"hipblasLtCreate -> {rc} (0 = HIPBLAS_STATUS_SUCCESS)  handle={handle.value}")

# 3) enumerate symbols we care about
for sym in ["hipblasLtMatmul", "hipblasLtMatmulAlgoGetHeuristic", "hipblasLtGetVersion"]:
    print(f"  {sym}: {'OK' if hasattr(lt, sym) else 'MISSING'}")
