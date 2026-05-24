import subprocess
import sys
import os

print("Running test_pipeline.py...")
# Use absolute path to avoid cwd issues
script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_pipeline.py"))
result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, encoding="utf-8", errors="replace")

print("Exit code:", result.returncode)
out_log = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_output.log"))
with open(out_log, "w", encoding="utf-8") as f:
    f.write(f"=== EXIT CODE: {result.returncode} ===\n")
    f.write("=== STDOUT ===\n")
    f.write(result.stdout)
    f.write("\n=== STDERR ===\n")
    f.write(result.stderr)
print(f"Done. Saved to {out_log}")
