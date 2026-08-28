"""Zero-forgetting benchmark: frozen (hash) vs learned router, sequential A->B.

Run:  B:\\git\\grilly-venv\\Scripts\\python.exe cubbylite\\benchmark.py
"""
import subprocess, sys, os
os.system(f'"{sys.executable}" "{os.path.dirname(__file__)}/benchmark.py"')
