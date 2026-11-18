# -*- coding: utf-8 -*-
"""Shared model-download helper for the notebooks.

All models are hosted on Hugging Face under the qhddz account. Two mirrors
are supported:

- ``hf``     : https://huggingface.co  (official)
- ``mirror`` : https://hf-mirror.com   (mirror, faster direct link in China)

Typical usage (in a notebook)::

    from nb_download import ensure_files

    ensure_files(FILES, "hf")   # download whatever is missing

``FILES`` is a list of (repo, remote_path, local_path) tuples; local files
that already exist (and are non-empty) are skipped, never re-downloaded.
"""
import os
import shutil
import urllib.request

SOURCES = {
    "hf": "https://huggingface.co/qhddz",
    "mirror": "https://hf-mirror.com/qhddz",
}
DEFAULT_SOURCE = "hf"


def source_picker(default=DEFAULT_SOURCE):
    """Return an ipywidgets dropdown for choosing the download source.

    Falls back to a plain object with a ``value`` attribute when ipywidgets
    is unavailable (e.g. plain command line).
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print(f"ipywidgets not installed, using default source: {default}")
        return type("W", (), {"value": default})()

    labels = [("Hugging Face (official)", "hf"),
              ("HF-Mirror (China mirror)", "mirror")]
    value = default if default in ("hf", "mirror") else "hf"
    w = widgets.Dropdown(
        options=labels, value=value, description="Source:",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="420px"))
    display(w)
    print("Then run the next cell to start downloading ▼")
    return w


def _download(url, dst):
    """Streaming download with retries; writes .part first, then renames."""
    from urllib.parse import quote
    url = quote(url, safe=":/?&=%")  # file names may contain spaces etc.
    tmp = dst + ".part"
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    last_err = None
    for attempt in range(3):
        try:
            # hf-mirror 403-blocks urllib's default UA; pretend to be a browser
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
                raise IOError(f"incomplete file: expected {total} bytes, got {done}")
            os.replace(tmp, dst)
            print()
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"\n    attempt {attempt + 1} failed: {e}")
    if os.path.exists(tmp):
        os.remove(tmp)
    raise RuntimeError(f"download failed (3 retries): {url}\n{last_err}")


def ensure_files(files, source=DEFAULT_SOURCE, quiet=False):
    """Make sure every file in `files` exists, downloading what's missing.

    files: [(repo, remote_path, local_path), ...]
    source: "hf" or "mirror"
    Returns (n_existing, n_downloaded).
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}, choose from: {list(SOURCES)}")
    base = SOURCES[source]

    missing = [(r, rp, lp) for r, rp, lp in files
               if not (os.path.exists(lp) and os.path.getsize(lp) > 0)]
    n_skip = len(files) - len(missing)
    if not files:
        return 0, 0
    if not quiet:
        print(f"Source: {base}")
        print(f"{len(files)} files: {n_skip} already exist, {len(missing)} to download")
    for repo, remote, local in missing:
        url = f"{base}/{repo}/resolve/main/{remote}"
        print(f"  downloading {remote} -> {local}")
        _download(url, local)
    if not missing and not quiet:
        print("All files ready, nothing to download ✔")
    return n_skip, len(missing)


def extract_tarball(tar_path, dest_dir, marker="lib"):
    """Extract demo_x86_64.tar.gz into dest_dir (skips if already extracted)."""
    import tarfile
    if not os.path.exists(tar_path):
        print(f"tarball missing, skipped: {tar_path} "
              f"(expected if you only downloaded part of the variants)")
        return
    if os.path.isdir(os.path.join(dest_dir, marker)):
        print(f"already extracted, skipped: {dest_dir}")
        return
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(tar_path) as t:
        t.extractall(dest_dir)
    # if the tarball nests a single top-level directory, flatten it
    entries = os.listdir(dest_dir)
    if (len(entries) == 1 and os.path.isdir(os.path.join(dest_dir, entries[0]))
            and os.path.isdir(os.path.join(dest_dir, entries[0], marker))):
        top = os.path.join(dest_dir, entries[0])
        for name in os.listdir(top):
            shutil.move(os.path.join(top, name), os.path.join(dest_dir, name))
        os.rmdir(top)
    # demo programs need the executable bit
    for name in os.listdir(dest_dir):
        p = os.path.join(dest_dir, name)
        if os.path.isfile(p) and not os.path.splitext(name)[1]:
            os.chmod(p, 0o755)
    print(f"extracted {tar_path} -> {dest_dir}")
