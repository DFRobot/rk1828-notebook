# -*- coding: utf-8 -*-
"""Shared helpers for running and displaying the prebuilt C++ demos."""
import glob
import os
import subprocess
import sys
import time


def run_demo(args, cwd, timeout=3600):
    """Run a prebuilt demo inside `cwd` (auto-injects LD_LIBRARY_PATH=lib).

    args: argv list (first element is the executable name).
    Streams stdout/stderr live and prints the elapsed time at the end;
    raises on a non-zero exit code.
    """
    env = dict(os.environ, LD_LIBRARY_PATH="lib")
    t0 = time.time()
    print(f"$ (cd {cwd} && LD_LIBRARY_PATH=lib {' '.join(args)})")
    p = subprocess.run(args, cwd=cwd, env=env, capture_output=True,
                       text=True, timeout=timeout)
    dt = time.time() - t0
    sys.stdout.write(p.stdout)
    if p.stderr:
        sys.stderr.write(p.stderr)
    print(f"\n[elapsed {dt:.1f}s, exit code {p.returncode}]")
    if p.returncode != 0:
        raise RuntimeError(f"demo failed (exit code {p.returncode}), see output above")
    return p


def snapshot(cwd):
    """Record the current set of files, for diffing out demo outputs later."""
    return {f for f in glob.glob(os.path.join(cwd, "**", "*"), recursive=True)
            if os.path.isfile(f)}


def new_files(cwd, before):
    """Return files created after the snapshot."""
    return sorted(set(glob.glob(os.path.join(cwd, "**", "*"), recursive=True))
                  - before)


def show_images(paths, max_width=None):
    """Display images one by one in the notebook."""
    from IPython.display import display
    from PIL import Image
    for f in paths:
        print(f)
        img = Image.open(f)
        if max_width and img.width > max_width:
            img = img.resize((max_width, int(img.height * max_width / img.width)))
        display(img)


def show_audio(path):
    """Display an audio player in the notebook."""
    from IPython.display import Audio, display
    print(path)
    display(Audio(filename=path))


def show_new_images(cwd, before, exts=(".png", ".jpg", ".jpeg")):
    """Display images created after the snapshot; returns the file list."""
    files = [f for f in new_files(cwd, before)
             if f.lower().endswith(exts)]
    if not files:
        print("(no new image files)")
    show_images(files)
    return files
