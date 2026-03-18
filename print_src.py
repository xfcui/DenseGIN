#!/usr/bin/env python3
import os
import subprocess
import glob
import sys

def print_files(paths=None):
    """
    Print files passed explicitly by user, or fall back to src/**/*.py.
    Supports globs and normal file paths.
    """
    if paths is None or len(paths) == 0:
        paths = ["src/**/*.py"]

    files = []
    for path in paths:
        if glob.has_magic(path):
            files.extend(glob.glob(path, recursive=True))
        elif os.path.isdir(path):
            files.extend(sorted(glob.glob(os.path.join(path, "**/*.py"), recursive=True)))
        else:
            files.append(path)

    files = sorted(set(files))
    
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
            # -o prettyprint: Adds headers and line numbers (if supported by the printer driver)
            # -o number-up=2: Prints in 2 columns
            # -o landscape: Prints in landscape orientation
            # -o Duplex=DuplexNoTumble: Enables two-sided printing (long-edge binding)
            # -t <title>: Adds a title (often used for the header)

            lp_cmd = [
                "lp",
                "-o", "prettyprint",
                "-o", "number-up=2",
                "-o", "landscape",
                "-o", "Duplex=DuplexNoTumble",
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
    print_files(sys.argv[1:])
