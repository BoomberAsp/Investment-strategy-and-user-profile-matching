#!/bin/bash
###############################################################################
# Investment Strategy App - Server Setup Script
#
# 在裸 Ubuntu 24.04 / 22.04 服务器上一键部署 Streamlit + Nginx
# 使用方法: sudo DOMAIN=oneweblog.cn bash setup_server.sh
#
# 部署前需要先将项目文件上传到服务器（见 README）
###############################################################################
set -e

echo "=========================================="
echo "  Investment Strategy App - Server Setup"
echo "=========================================="

# ──────────────────────────────────────────────
# 参数配置（可通过环境变量覆盖）
# ──────────────────────────────────────────────
DOMAIN="${DOMAIN:-oneweblog.cn}"
PROJECT_DIR="${PROJECT_DIR:-/opt/investment-app}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
PYTHON_VER="${PYTHON_VER:-3.12}"

# ──────────────────────────────────────────────
# Step 0: 检查 root 权限
# ──────────────────────────────────────────────
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: 请使用 sudo 运行此脚本"
    exit 1
fi

# ──────────────────────────────────────────────
# Step 1: 安装系统依赖
# ──────────────────────────────────────────────
echo ""
echo "[1/7] Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive
apt update -qq
apt install -y --no-install-recommends \
    "python${PYTHON_VER}" \
    "python${PYTHON_VER}-venv" \
    "python${PYTHON_VER}-dev" \
    python3-pip \
    nginx \
    curl \
    fonts-wqy-zenhei \
    fonts-noto-cjk \
    supervisor

echo "  -> System dependencies installed."

# ──────────────────────────────────────────────
# Step 2: 中文字体配置
# ──────────────────────────────────────────────
echo ""
echo "[2/7] Setting up Chinese fonts..."
fc-cache -fv 2>/dev/null

# 创建 matplotlib 字体缓存目录（确保中文显示）
mkdir -p /root/.cache/matplotlib
echo "  -> Chinese fonts configured."

# ──────────────────────────────────────────────
# Step 3: 创建项目目录（如果不存在）
# ──────────────────────────────────────────────
echo ""
echo "[3/7] Preparing project directory: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"

# ──────────────────────────────────────────────
# Step 4: 创建 Python 虚拟环境
# ──────────────────────────────────────────────
echo ""
echo "[4/7] Creating Python virtual environment..."
cd "$PROJECT_DIR"

# 如果已有 venv 则跳过
if [ ! -d "venv" ]; then
    "python${PYTHON_VER}" -m venv venv
    echo "  -> Virtual environment created."
else
    echo "  -> Virtual environment already exists, skipping."
fi

source venv/bin/activate
pip install --upgrade pip -q

# ──────────────────────────────────────────────
# Step 5: 安装 Python 依赖
# ──────────────────────────────────────────────
echo ""
echo "[5/7] Installing Python dependencies..."

REQ_FILE="$PROJECT_DIR/requirements.txt"
if [ ! -f "$REQ_FILE" ]; then
    echo "  ERROR: requirements.txt not found at $REQ_FILE"
    echo "  Please copy your project files to $PROJECT_DIR first."
    exit 1
fi

# 检测 GPU 并安装对应版本的 torch
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "  -> GPU detected, installing torch with CUDA..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121 -q
else
    echo "  -> No GPU detected, installing CPU-only torch..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu -q
fi

# 安装其余依赖（跳过已在上面安装的 torch）
grep -v '^torch' "$REQ_FILE" | pip install -r /dev/stdin -q

echo "  -> Python dependencies installed."

# ──────────────────────────────────────────────
# Step 6: Nginx 反向代理配置
# ──────────────────────────────────────────────
echo ""
echo "[6/7] Configuring Nginx reverse proxy..."

NGINX_CONF="/etc/nginx/sites-available/investment-app"
cat > "$NGINX_CONF" <<NGINX_EOF
server {
    listen 80;
    server_name ${DOMAIN};

    # Streamlit 反向代理
    location /app/ {
        proxy_pass http://127.0.0.1:${STREAMLIT_PORT}/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 86400;
    }
}
NGINX_CONF_EOF

# 启用站点配置
ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/investment-app
# 移除默认的站点（避免冲突）
rm -f /etc/nginx/sites-enabled/default

nginx -t && systemctl reload nginx
echo "  -> Nginx configured for ${DOMAIN}"

# ──────────────────────────────────────────────
# Step 7: Supervisor 进程管理
# ──────────────────────────────────────────────
echo ""
echo "[7/7] Configuring supervisor service..."

cat > /etc/supervisor/conf.d/investment-app.conf <<SUPERVISOR_EOF
[program:investment-app]
command=${PROJECT_DIR}/venv/bin/streamlit run ${PROJECT_DIR}/app.py \
    --server.address 0.0.0.0 \
    --server.port ${STREAMLIT_PORT} \
    --server.headless true
directory=${PROJECT_DIR}
autostart=true
autorestart=true
user=root
stdout_logfile=/var/log/investment-app.log
stderr_logfile=/var/log/investment-app-error.log
environment=HOME="${PROJECT_DIR}"
SUPERVISOR_EOF

supervisorctl reread
supervisorctl update
supervisorctl start investment-app 2>/dev/null || true

echo "  -> Supervisor service configured."

# ──────────────────────────────────────────────
# 完成
# ──────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Server setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Copy project files to $PROJECT_DIR:"
echo "     - stats_data/          (strategy data)"
echo "     - DLMethod/models/     (pre-trained model weights)"
echo "     - app/data/            (user data directory)"
echo "     - *.py, *.json files   (app code and config)"
echo ""
echo "  2. Edit $PROJECT_DIR/DLMethod/.env and set DEEPSEEK_API_KEY"
echo ""
echo "  3. Run pipeline (generates precomputed data):"
echo "     cd $PROJECT_DIR && source venv/bin/activate && python pipeline.py"
echo ""
echo "  4. Restart app:"
echo "     supervisorctl restart investment-app"
echo ""
echo "  5. Setup HTTPS (requires domain DNS pointing to this server):"
echo "     certbot --nginx -d ${DOMAIN}"
echo ""
echo "  App will be available at: http://${DOMAIN}/app/"
echo "  Logs: /var/log/investment-app.log"
echo "  Status: supervisorctl status investment-app"
