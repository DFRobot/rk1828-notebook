#!/bin/bash
# Download every model file needed by the notebooks (from huggingface.co).
# Usage: bash download_models.sh [hf|mirror]   default: hf (huggingface.co)
set -e
cd "$(dirname "$0")"
SRC="${1:-hf}"
case "$SRC" in
  mirror|hf) ;;
  *) echo "Usage: bash download_models.sh [hf|mirror]"; exit 1;;
esac

python3 - "$SRC" <<'PYEOF'
import ast
import glob
import json
import re
import sys

sys.path.insert(0, ".")
from nb_download import ensure_files

# Aggregate the FILES lists from every notebook (single source of truth)
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

# De-duplicate by local path
seen, uniq = set(), []
for f in files:
    if f[2] not in seen:
        seen.add(f[2])
        uniq.append(f)

print(f"{len(uniq)} files (de-duplicated)")
ensure_files(uniq, sys.argv[1])
PYEOF

echo "== all models downloaded =="
