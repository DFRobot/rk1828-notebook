# -*- coding: utf-8 -*-
"""notebook 共用的 C++ demo 运行/展示辅助。"""
import glob
import os
import subprocess
import sys
import time


def run_demo(args, cwd, timeout=3600):
    """在 cwd 目录下运行预编译 demo（自动注入 LD_LIBRARY_PATH=lib）。

    args: 命令行参数列表（第一个元素是可执行文件名）
    实时打印 stdout/stderr，结束后打印耗时；非零返回码抛异常。
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
    print(f"\n[耗时 {dt:.1f}s, 返回码 {p.returncode}]")
    if p.returncode != 0:
        raise RuntimeError(f"demo 运行失败（返回码 {p.returncode}），见上方输出")
    return p


def snapshot(cwd):
    """记录目录当前文件集合，用于之后 diff 出 demo 生成的文件。"""
    return {f for f in glob.glob(os.path.join(cwd, "**", "*"), recursive=True)
            if os.path.isfile(f)}


def new_files(cwd, before):
    """返回快照之后新增的文件列表。"""
    return sorted(set(glob.glob(os.path.join(cwd, "**", "*"), recursive=True))
                  - before)


def show_images(paths, max_width=None):
    """在 notebook 里依次显示图片。"""
    from IPython.display import display
    from PIL import Image
    for f in paths:
        print(f)
        img = Image.open(f)
        if max_width and img.width > max_width:
            img = img.resize((max_width, int(img.height * max_width / img.width)))
        display(img)


def show_audio(path):
    """在 notebook 里显示音频播放器。"""
    from IPython.display import Audio, display
    print(path)
    display(Audio(filename=path))


def show_new_images(cwd, before, exts=(".png", ".jpg", ".jpeg")):
    """显示快照之后新生成的图片，返回文件列表。"""
    files = [f for f in new_files(cwd, before)
             if f.lower().endswith(exts)]
    if not files:
        print("（没有生成新的图片文件）")
    show_images(files)
    return files
