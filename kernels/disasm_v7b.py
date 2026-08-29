import sys, struct, subprocess, ctypes
sys.path.insert(0, r"B:\git\cubbyllm-lite\kernels")
import gemm_v6 as G
import wmma_gemm_v2 as W
RTC = W.RTC

buf = ctypes.create_string_buffer(G.SRC.encode())
prog = ctypes.c_void_p()
assert RTC.hiprtcCreateProgram(ctypes.byref(prog), ctypes.cast(buf, ctypes.c_char_p), b"v7b", 0, None, None) == 0
opts = (ctypes.c_char_p * 4)(b"--offload-arch=gfx1201", b"-O3", b"-DNEG_A=1", b"-DNEG_B=1")
assert RTC.hiprtcCompileProgram(prog, 4, opts) == 0
csz = ctypes.c_size_t(); RTC.hiprtcGetCodeSize(prog, ctypes.byref(csz))
code = ctypes.create_string_buffer(csz.value); RTC.hiprtcGetCode(prog, code)
elf = code.raw
open("v7b.hsaco", "wb").write(elf)
print(f"code object: {len(elf)} bytes")

# minimal ELF64 section extraction
e_shoff = struct.unpack_from("<Q", elf, 0x28)[0]
e_shentsize = struct.unpack_from("<H", elf, 0x3A)[0]
e_shnum = struct.unpack_from("<H", elf, 0x3C)[0]
e_shstrndx = struct.unpack_from("<H", elf, 0x3E)[0]
def shdr(i):
    o = e_shoff + i * e_shentsize
    name, typ, flags, addr, off, size = struct.unpack_from("<IIQQQQ", elf, o)
    return name, typ, off, size
strtab_off = shdr(e_shstrndx)[2]
def sname(n):
    end = elf.index(b"\0", strtab_off + n)
    return elf[strtab_off + n:end].decode()
text = None
for i in range(e_shnum):
    name, typ, off, size = shdr(i)
    if sname(name) == ".text":
        text = elf[off:off + size]
        print(f".text: {size} bytes at section {i}")
assert text
open("v7b.text.bin", "wb").write(text)

LLMC = r"B:\git\rocm-venv\Lib\site-packages\_rocm_sdk_core\lib\llvm\bin\llvm-mc.exe"
hexstr = text.hex()
p = subprocess.run([LLMC, "--triple=amdgcn-amd-amdhsa", "-mcpu=gfx1201", "--disassemble"],
                   input=hexstr, capture_output=True, text=True, timeout=120)
asm = p.stdout
open("v7b.asm", "w").write(asm)
print(f"disasm: {len(asm.splitlines())} lines, rc={p.returncode}")
if p.returncode != 0:
    print(p.stderr[:400])
