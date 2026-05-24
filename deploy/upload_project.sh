#!/bin/bash
###############################################################################
# Upload project files from local machine to remote server
#
# 使用方法:
#   bash upload_project.sh user@server_ip
#
# 可选环境变量:
#   PROJECT_DIR  - 远程服务器上的项目目录（默认 /opt/investment-app）
#   EXCLUDE_GPU  - 设为 1 则排除 DLMethod/models/ 下的 GPU 权重（节省空间）
###############################################################################
set -e

REMOTE="${1:?Usage: bash upload_project.sh user@server_ip}"
PROJECT_DIR="${PROJECT_DIR:-/opt/investment-app}"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Uploading project to $REMOTE:$PROJECT_DIR ..."

# 创建远程目录
ssh "$REMOTE" "mkdir -p $PROJECT_DIR"

# 同步项目文件（排除 venv、.git、缓存等）
rsync -avz --progress \
    --exclude 'InvestmentMatching' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '*.whl' \
    --exclude '.idea' \
    --exclude '*.egg-info' \
    --exclude '.streamlit' \
    "$LOCAL_ROOT/" "$REMOTE:$PROJECT_DIR/"

echo ""
echo "Upload complete!"
echo "Project files are at $REMOTE:$PROJECT_DIR"
