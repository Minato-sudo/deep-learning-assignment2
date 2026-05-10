import os
import sys

# Wrapper to match standard template
if __name__ == "__main__":
    os.system(f"python repo/original_architecture_reproduction/main.py {' '.join(sys.argv[1:])}")
