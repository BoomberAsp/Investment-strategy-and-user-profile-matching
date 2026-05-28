#!/bin/bash
###############################################################################
# 快速部署脚本 — 假设项目 + venv 已上传到服务器
#
# 用法:
#   sudo DOMAIN=app.oneweblog.cn bash setup.sh
#   sudo DOMAIN=app.oneweblog.cn PROJECT_DIR=/root/vdb/investment-app PORT=8502 bash setup.sh
#
# 环境变量（均可选）:
#   DOMAIN       — 域名或子域名（默认 app.oneweblog.cn）
#   PROJECT_DIR  — 项目根目录（默认脚本所在目录的父目录）
#   PORT         — Streamlit 端口（默认 8501）
###############################################################################
set -e

DOMAIN="${DOMAIN:-app.oneweblog.cn}"
PORT="${PORT:-8501}"

# 自动推断项目目录（脚本在 deploy/ 下，父目录即项目根）
if [ -z "$PROJECT_DIR" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
fi

echo "=========================================="
echo "  Investment App — Quick Setup"
echo "  Project : $PROJECT_DIR"
echo "  Domain  : $DOMAIN"
echo "  Port    : $PORT"
echo "=========================================="

# ── 0. 权限检查 ──
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: 请使用 sudo 运行"
    exit 1
fi

# ── 1. 系统依赖 ──
echo ""
echo "[1/4] Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive
apt update -qq
apt install -y --no-install-recommends nginx supervisor fonts-wqy-zenhei fonts-noto-cjk > /dev/null
echo "  -> Done"

# ── 2. 修复 venv shebang ──
echo ""
echo "[2/4] Fixing venv shebang paths..."

# 查找 venv 目录（支持 venv / InvestmentMatching 等命名）
VENV_DIR=""
for d in "$PROJECT_DIR/InvestmentMatching" "$PROJECT_DIR/venv"; do
    if [ -f "$d/bin/streamlit" ]; then
        VENV_DIR="$d"
        break
    fi
done

if [ -z "$VENV_DIR" ]; then
    echo "  ERROR: 未找到虚拟环境 (streamlit 不在 venv/ 或 InvestmentMatching/ 中)"
    exit 1
fi

PYTHON_BIN="$VENV_DIR/bin/python"
echo "  -> 虚拟环境: $VENV_DIR"

# 修复所有指向旧路径的 shebang
BROKEN_COUNT=$(grep -rl '^#!.*/python$' "$VENV_DIR/bin/" 2>/dev/null | while read f; do
    head -1 "$f" | grep -qv "$PYTHON_BIN" && echo "$f"
done | wc -l)

if [ "$BROKEN_COUNT" -gt 0 ]; then
    grep -rl '^#!.*/python$' "$VENV_DIR/bin/" 2>/dev/null | while read f; do
        sed -i "1s|.*|#!${PYTHON_BIN}|" "$f"
    done
    echo "  -> 已修复 $BROKEN_COUNT 个 shebang 路径"
else
    echo "  -> shebang 路径正常，跳过"
fi

# ── 3. Nginx ──
echo ""
echo "[3/4] Configuring Nginx..."

tee /etc/nginx/sites-available/investment-app > /dev/null <<NGINX
server {
    listen 80;
    server_name ${DOMAIN};

    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    location / {
        proxy_pass http://127.0.0.1:${PORT}/;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
        proxy_buffering off;
        proxy_cache off;
    }
}
NGINX

ln -sf /etc/nginx/sites-available/investment-app /etc/nginx/sites-enabled/investment-app
rm -f /etc/nginx/sites-enabled/default
mkdir -p /var/www/certbot

nginx -t && systemctl reload nginx
echo "  -> Nginx 已配置: http://${DOMAIN}"

# ── 4. Streamlit 字体修复 ──
echo ""
echo "[4/4] Configuring Streamlit & Supervisor..."

# 修复 matplotlibrc 中文字体
APP_FILE="$PROJECT_DIR/app.py"
if grep -q "SimHei" "$APP_FILE"; then
    sed -i "s/\['SimHei'\]/['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei']/" "$APP_FILE"
    echo "  -> 字体已适配服务器"
fi

# 清除字体缓存
rm -rf /root/.cache/matplotlib /root/.matplotlib \
       /home/ubuntu/.cache/matplotlib /home/ubuntu/.matplotlib 2>/dev/null || true

# Supervisor 配置
tee /etc/supervisor/conf.d/investment-app.conf > /dev/null <<SUP
[program:investment-app]
command=${VENV_DIR}/bin/streamlit run ${PROJECT_DIR}/app.py \\
    --server.address 0.0.0.0 \\
    --server.port ${PORT} \\
    --server.headless true \\
    --server.enableCORS false \\
    --server.enableXsrfProtection false
directory=${PROJECT_DIR}
autostart=true
autorestart=true
user=root
stdout_logfile=/var/log/investment-app.log
stderr_logfile=/var/log/investment-app-error.log
environment=HOME="${PROJECT_DIR}"
SUP

supervisorctl reread > /dev/null
supervisorctl update > /dev/null
supervisorctl restart investment-app > /dev/null 2>&1 || supervisorctl start investment-app

# ── 健康检查 ──
echo ""
echo "--- 健康检查 ---"
sleep 3
if supervisorctl status investment-app | grep -q RUNNING; then
    echo "  Supervisor : RUNNING"
else
    echo "  Supervisor : FAILED — 查看日志: tail /var/log/investment-app-error.log"
fi

HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    echo "  Streamlit  : HTTP 200"
else
    echo "  Streamlit  : HTTP ${HTTP_CODE}"
fi

HTTP_NGINX=$(curl -s -o /dev/null -w '%{http_code}' -H "Host: ${DOMAIN}" "http://127.0.0.1/" 2>/dev/null || echo "000")
if [ "$HTTP_NGINX" = "200" ] || [ "$HTTP_NGINX" = "301" ] || [ "$HTTP_NGINX" = "302" ]; then
    echo "  Nginx      : HTTP ${HTTP_NGINX}"
else
    echo "  Nginx      : HTTP ${HTTP_NGINX}"
fi

echo ""
echo "=========================================="
echo "  Setup 完成"
echo "=========================================="
echo ""
echo "  访问地址: http://${DOMAIN}/"
echo "  查看状态: sudo supervisorctl status investment-app"
echo "  查看日志: tail -f /var/log/investment-app.log"
echo "  配置 HTTPS: sudo certbot --nginx -d ${DOMAIN}"
