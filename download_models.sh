#!/bin/bash
# 下载全部 notebook 需要的模型文件（可选下载源）
# 用法: bash download_models.sh [mirror|hf]   默认 mirror（HF-Mirror 国内镜像）
set -e
cd "$(dirname "$0")"
SRC="${1:-mirror}"
case "$SRC" in
  mirror|hf) ;;
  *) echo "用法: bash download_models.sh [mirror|hf]"; exit 1;;
esac

python3 - "$SRC" <<'PYEOF'
import ast
import glob
import json
import re
import sys

sys.path.insert(0, ".")
from nb_download import ensure_files

# 从各 notebook 的 FILES 清单汇总（单一事实来源）
files = []
for nb_path in sorted(glob.glob("*.ipynb")):
    nb = json.load(open(nb_path))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        m = re.search(r"^FILES\s*=\s*(\[.*?\])\s*$", src, re.S | re.M)
        if m:
            files.extend(ast.literal_eval(m.group(1)))

# 按本地路径去重
seen, uniq = set(), []
for f in files:
    if f[2] not in seen:
        seen.add(f[2])
        uniq.append(f)

print(f"共 {len(uniq)} 个文件（已去重）")
ensure_files(uniq, sys.argv[1])
PYEOF

echo "== 全部模型下载完成 =="
