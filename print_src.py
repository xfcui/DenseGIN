#!/usr/bin/env python3
import os
import subprocess
import glob
from pathlib import Path

def print_files(pattern="src/**/*.py"):
    """
    Prints all files matching the glob pattern using the default system printer.
    Optimizes layout using standard 'lp' options for columns, landscape, and headers.
    """
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return

    files.sort()
    print(f"Found {len(files)} files to print.")

    for file_path in files:
        if os.path.isfile(file_path):
            if os.path.basename(file_path) == "__init__.py":
                print(f"Skipping: {file_path}")
                continue
            
            print(f"Printing: {file_path}")
            
            # Use native 'lp' options for optimized layout:
            # -o Duplex=DuplexNoTumble: Duplex printing (long-edge)
            # -o prettyprint: Adds headers and line numbers (if supported by the printer driver)
            # -t <title>: Adds a title (often used for the header)
            
            lp_cmd = [
                "lp",
                "-o", "Duplex=DuplexNoTumble",
                "-o", "prettyprint",
                "-t", file_path,
                file_path
            ]
            
            try:
                result = subprocess.run(lp_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error printing {file_path}: {result.stderr}")
                else:
                    print(f"Successfully sent {file_path} to printer.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    print_files()
