# -*- coding: utf-8 -*-
"""notebook 共用的模型下载模块。

所有模型托管在 Hugging Face 的 qhddz 账号下，支持两个下载源：

- ``hf``     : https://huggingface.co  （官方，海外/有代理时用）
- ``mirror`` : https://hf-mirror.com   （国内镜像，直连更快）

典型用法（在 notebook 里）::

    from nb_download import source_picker, ensure_files

    w_source = source_picker()          # 下拉框选下载源（cell 1）
    ensure_files(FILES, w_source.value) # 下载缺失文件（cell 2）

``FILES`` 是 (repo, remote_path, local_path) 三元组的列表；
已存在（且非空）的本地文件会直接跳过，不重复下载。
"""
import os
import shutil
import urllib.request

SOURCES = {
    "mirror": "https://hf-mirror.com/qhddz",
    "hf": "https://huggingface.co/qhddz",
}
DEFAULT_SOURCE = "mirror"


def source_picker(default=DEFAULT_SOURCE):
    """返回一个 ipywidgets 下拉框，用于选择下载源。

    在没有 ipywidgets 的环境（如命令行）退化为打印提示并返回
    一个带 ``value`` 属性的简单对象。
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print(f"未安装 ipywidgets，使用默认下载源: {default}")
        return type("W", (), {"value": default})()

    labels = [("HF-Mirror 国内镜像（推荐，直连快）", "mirror"),
              ("Hugging Face 官方（海外/有代理）", "hf")]
    value = default if default in ("mirror", "hf") else "mirror"
    w = widgets.Dropdown(
        options=labels, value=value, description="下载源:",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="420px"))
    display(w)
    print("选好后，继续运行下一个单元格开始下载 ▼")
    return w


def _download(url, dst):
    """带重试的流式下载，先写 .part 再改名，失败不留半截文件。"""
    from urllib.parse import quote
    url = quote(url, safe=":/?&=%")  # 文件名可能含空格等字符
    tmp = dst + ".part"
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    last_err = None
    for attempt in range(3):
        try:
            # hf-mirror 会 403 拦截 Python-urllib 默认 UA，伪装成浏览器
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                                            "AppleWebKit/537.36 Chrome/120 Safari/537.36"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r    {done/1e6:8.1f}/{total/1e6:.1f} MB "
                              f"({done*100//total}%)", end="", flush=True)
            if total and done != total:
                raise IOError(f"文件不完整: 期望 {total} 字节，实收 {done} 字节")
            os.replace(tmp, dst)
            print()
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"\n    第 {attempt + 1} 次下载失败: {e}")
    if os.path.exists(tmp):
        os.remove(tmp)
    raise RuntimeError(f"下载失败（已重试 3 次）: {url}\n{last_err}")


def ensure_files(files, source=DEFAULT_SOURCE, quiet=False):
    """确保 files 里的文件都存在，缺什么下什么。

    files: [(repo, remote_path, local_path), ...]
    source: "mirror" 或 "hf"
    返回 (已存在的数量, 新下载的数量)。
    """
    if source not in SOURCES:
        raise ValueError(f"未知下载源 {source!r}，可选: {list(SOURCES)}")
    base = SOURCES[source]

    missing = [(r, rp, lp) for r, rp, lp in files
               if not (os.path.exists(lp) and os.path.getsize(lp) > 0)]
    n_skip = len(files) - len(missing)
    if not files:
        return 0, 0
    if not quiet:
        print(f"下载源: {base}")
        print(f"共 {len(files)} 个文件: {n_skip} 个已存在跳过，{len(missing)} 个需要下载")
    for repo, remote, local in missing:
        url = f"{base}/{repo}/resolve/main/{remote}"
        print(f"  下载 {remote} -> {local}")
        _download(url, local)
    if not missing and not quiet:
        print("所有文件已就绪，无需下载 ✔")
    return n_skip, len(missing)


def extract_tarball(tar_path, dest_dir, marker="lib"):
    """解压 demo_x86_64.tar.gz 到 dest_dir（已解压过则跳过）。"""
    import tarfile
    if not os.path.exists(tar_path):
        print(f"压缩包不存在，跳过: {tar_path}（如果只下载了部分规格，这是正常的）")
        return
    if os.path.isdir(os.path.join(dest_dir, marker)):
        print(f"已解压，跳过: {dest_dir}")
        return
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(tar_path) as t:
        t.extractall(dest_dir)
    # 如果压缩包里带一层同名顶层目录，则把内容上移一层
    entries = os.listdir(dest_dir)
    if (len(entries) == 1 and os.path.isdir(os.path.join(dest_dir, entries[0]))
            and os.path.isdir(os.path.join(dest_dir, entries[0], marker))):
        top = os.path.join(dest_dir, entries[0])
        for name in os.listdir(top):
            shutil.move(os.path.join(top, name), os.path.join(dest_dir, name))
        os.rmdir(top)
    # demo 程序需要可执行权限
    for name in os.listdir(dest_dir):
        p = os.path.join(dest_dir, name)
        if os.path.isfile(p) and not os.path.splitext(name)[1]:
            os.chmod(p, 0o755)
    print(f"已解压 {tar_path} -> {dest_dir}")
