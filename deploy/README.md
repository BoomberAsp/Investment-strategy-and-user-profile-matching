# 部署指南

## 快速部署（两步）

### 第 1 步：上传项目文件到服务器

```bash
bash deploy/upload_project.sh user@your_server_ip
```

例如：
```bash
bash deploy/upload_project.sh root@116.7.234.249
```

### 第 2 步：在服务器上运行安装脚本

SSH 登录服务器后执行：

```bash
sudo DOMAIN=oneweblog.cn bash /opt/investment-app/deploy/setup_server.sh
```

### 第 3 步：配置和启动

```bash
# 1. 编辑环境变量（设置 DeepSeek API Key，可选）
nano /opt/investment-app/DLMethod/.env

# 2. 运行数据管线（生成预计算文件）
cd /opt/investment-app
source venv/bin/activate
python pipeline.py

# 3. 重启应用
sudo supervisorctl restart investment-app

# 4. 检查状态
sudo supervisorctl status investment-app

# 5. 配置 HTTPS（需要域名 DNS 已指向服务器）
sudo certbot --nginx -d oneweblog.cn
```

## 目录结构

```
/opt/investment-app/
├── app.py                    # Streamlit 入口
├── pipeline.py               # 数据管线
├── requirements.txt          # Python 依赖
├── account_details.json      # 账户配置
├── user_accounts_info.json   # 用户信息
├── venv/                     # Python 虚拟环境（自动创建）
├── app/                      # 应用模块
├── DLMethod/                 # 深度学习模块
│   ├── models/               # 预训练模型权重（必须存在）
│   └── .env                  # DeepSeek API Key
├── stats_data/               # 策略源数据（必须存在）
└── deploy/                   # 部署脚本
```

## 管理命令

| 操作 | 命令 |
|------|------|
| 查看状态 | `sudo supervisorctl status investment-app` |
| 启动 | `sudo supervisorctl start investment-app` |
| 停止 | `sudo supervisorctl stop investment-app` |
| 重启 | `sudo supervisorctl restart investment-app` |
| 查看日志 | `tail -f /var/log/investment-app.log` |
| 查看错误日志 | `tail -f /var/log/investment-app-error.log` |

## Nginx 管理

| 操作 | 命令 |
|------|------|
| 测试配置 | `sudo nginx -t` |
| 重载配置 | `sudo systemctl reload nginx` |
| 查看状态 | `sudo systemctl status nginx` |
| 配置文件 | `/etc/nginx/sites-available/investment-app` |

## 故障排查

### 中文显示为方块
```bash
sudo apt install -y fonts-wqy-zenhei fonts-noto-cjk
fc-cache -fv
sudo supervisorctl restart investment-app
```

### 端口被占用
```bash
# 修改端口
sudo STREAMLIT_PORT=8502 DOMAIN=oneweblog.cn bash /opt/investment-app/deploy/setup_server.sh
```

### GPU 服务器切换
如果服务器有 GPU，脚本会自动安装 CUDA 版本的 torch。
无 GPU 则安装 CPU-only 版本。无需手动配置。
