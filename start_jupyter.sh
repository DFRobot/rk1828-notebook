#!/bin/bash
# 在本仓库目录启动 Jupyter Lab，监听 0.0.0.0:8888（局域网可访问）
# 访问方式：http://<本机IP>:8888/?token=<启动日志里的 token>
cd "$(dirname "$0")"
exec jupyter lab --ip=0.0.0.0 --no-browser "$@"
