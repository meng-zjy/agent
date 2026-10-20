# 17 封闭 Beta 部署（Ubuntu 26.04 + Docker + Tailscale）

> **目标**：把当前单机 Agent 部署到 Ubuntu 服务器，供你自己的设备和少量受邀测试者通过 Tailscale 私有网络访问。
> 不需要公网域名、不暴露应用端口、不引入多进程或账户系统。

## 1. 部署边界

这是一套**封闭 Beta**方案，不是公开 SaaS：

```text
已加入同一 Tailscale 网络的设备
             ↓ HTTPS（Tailscale Serve）
Ubuntu 主机上的 127.0.0.1:8000
             ↓
Docker 内单个 Uvicorn worker
             ↓
宿主机 ./data（会话 / 知识库 / 日志）
```

- `8000` 只绑定 `127.0.0.1`，云防火墙和 UFW 都不应开放它。
- 必须维持 **一个 Uvicorn worker**。会话目前是 JSON 文件；单进程锁不能协调多个 worker。
- `data/` 是唯一业务数据目录，删除容器不会删除它；删除该目录会丢失会话与日志。
- Beta 模板通过 `MCP_ENABLED=false` 关闭 filesystem MCP，避免模型触发服务器端文件操作。
- Tailscale 账号/ACL 是访问边界。只有你自己使用时，将服务器和电脑加入同一账号即可；邀请测试者前先配置 ACL。

## 2. 首次登录与主机基线

在云控制台启动实例后，以 SSH 密钥登录。不要在聊天记录、仓库或截图中暴露私钥、API Key 或完整公网 IP。

```bash
ssh ubuntu@<服务器公网IP>
sudo apt update
sudo apt upgrade -y
sudo apt install -y ca-certificates curl git ufw
```

先确保当前 SSH 连接不会被防火墙锁在外面，再启用 UFW：

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow OpenSSH
sudo ufw enable
sudo ufw status verbose
```

云厂商安全组也只需要先放行 `22`（SSH）。**不要开放 `8000`**；Tailscale 通过出站连接组网。

4 GiB 内存足够本阶段运行一个 worker。可额外配置 2 GiB swap，降低首次构建或加载本地 embedding 模型时被 OOM 杀掉的风险：

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h
```

## 3. 安装 Docker Engine 与 Compose

按 Docker 官方针对 Ubuntu 26.04 的当前安装指引安装 Docker Engine 和 Compose plugin。安装后执行：

```bash
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
exit
```

重新登录服务器，再验证：

```bash
docker --version
docker compose version
```

优先使用官方 Engine。若服务器到 Docker 官方仓库的 GPG Key / 软件源网络不稳定，Ubuntu 26.04 的系统包是可用降级方案（本项目已实测 Docker 29.1.3、Compose 2.40.3）：

```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2
sudo systemctl enable --now docker
docker --version
docker compose version
```

生产环境建议建立普通部署用户并加入 `docker` 组；本次封闭 Beta 以 root 维护时，仍须限制 SSH 登录并妥善保管密钥。

## 4. 部署 Agent

```bash
git clone <你的Git仓库SSH或HTTPS地址> agent
cd agent
cp .env.production.example .env
chmod 600 .env
nano .env
```

只需把 `DEEPSEEK_API_KEY` 改为真实 Key。保留下面三项，不要为了“功能完整”而在服务器上先打开 MCP：

```dotenv
MCP_ENABLED=false
SESSION_DIR=/app/data/sessions
LOG_DIR=/app/data/logs
TIME_ZONE=Asia/Shanghai
```

`TIME_ZONE` 是网页询问“现在几点 / 今天几号 / 星期几”时的权威时区。服务端会直接返回该时区的时间，不再将这类结果交给模型改写；宿主机仍应保持 NTP 时间同步。

启用 M10 多用户封闭 Beta 时，额外在 `.env` 设置并妥善保管邀请码：

```dotenv
ACCOUNT_ENABLED=true
REGISTRATION_INVITE_CODE=使用密码管理器生成的长随机口令
AUTH_DATABASE=/app/data/auth.db
AUTH_SESSION_HOURS=168
```

升级后的**第一位注册用户**会认领旧会话，所以应由服务器所有者先注册。邀请码、`auth.db` 与 API Key 都属于敏感数据，不能提交 Git 或发到聊天中；升级前先备份 `data/`。

如果 Docker Hub 在服务器网络中无法拉取 Python 镜像，模板已配置 `PYTHON_IMAGE` 使用镜像代理。
它仅影响 Docker 构建，不会传入 Agent。不要删除该行；如果将来服务器能稳定直连 Docker Hub，
可改回 `python:3.12-slim`。

为非 root 容器进程创建可写数据目录，然后构建并启动：

```bash
mkdir -p data/sessions data/knowledge data/logs
sudo chown -R 10001:10001 data
docker compose up -d --build
docker compose ps
curl --fail http://127.0.0.1:8000/health
```

首次启动若启用 RAG，会下载 embedding 模型；这会比普通启动慢。查看启动与应用日志：

```bash
docker compose logs -f agent
tail -f data/logs/agent.log
```

## 5. 安装并配置 Tailscale Serve

在服务器安装 Tailscale 并登录你的 tailnet。安装方式与命令以 Tailscale 官方 Ubuntu 文档为准；登录完成后确认状态：

```bash
sudo tailscale up
tailscale status
```

将 Tailscale 的 HTTPS 入口反向代理到本机 Agent。该命令不把服务发布到公网：

```bash
sudo tailscale serve --bg --https=443 http://127.0.0.1:8000
tailscale serve status
```

`tailscale serve status` 会显示一个 `https://<机器名>.<tailnet>.ts.net` 地址。用已登录同一 tailnet 的本机浏览器打开它，发送一条普通文本消息，确认 SSE 流式输出正常。

若只想先验证网络，也可以从另一台已登录 Tailscale 的设备访问服务器的 Tailscale IP；但正式日常使用优先走上面的 HTTPS 地址。

## 6. 验收清单

1. `docker compose ps` 中 `agent` 为 `healthy`。
2. `curl http://127.0.0.1:8000/health` 返回 `"ok": true`。
3. Tailscale HTTPS 地址能打开页面；未加入 tailnet 的设备不能访问。
4. 新建会话、发送消息、重启容器后，会话仍存在。
5. `data/logs/agent.log` 出现 request id，且不包含消息正文或 API Key。
6. 图片超过 3 MiB、超过 3 张、SVG 或伪造 data URI 都得到 400，而不是进入模型请求。
7. `/health` 和页面显示的图片限制与 `.env` 一致。

重启验证：

```bash
docker compose restart agent
docker compose ps
curl --fail http://127.0.0.1:8000/health
```

## 7. 更新、备份与故障处理

更新代码前先备份 `data/`，它包含用户会话：

```bash
tar -czf "agent-data-$(date +%F).tar.gz" data
git pull --ff-only
docker compose up -d --build
docker compose ps
```

如果容器不能启动，先看 `docker compose logs --tail=200 agent`；如果是权限错误，确认 `data/` 及其子目录属于 UID `10001`。如果 embedding 模型下载失败，可先保留空的 `data/knowledge/`，应用会降级为不带 RAG 的聊天。

## 8. 这套方案不解决什么

- 不支持多人账户、会话归属或审计；受邀 Tailscale 用户看到的是同一个 Agent 数据目录。
- 不支持多 worker 或多台机器共享会话；要扩容前应迁移到数据库并实现跨进程锁。
- 不适合向陌生公网用户开放。公开前需要域名/HTTPS、真实登录、按用户隔离数据、反向代理限流、备份与监控，并重新评审 MCP、RAG 和图片输入。

## 9. 阿里云 Ubuntu 26.04 实际部署记录（2026-10-20）

本节记录这次环境中实际遇到的网络与时区问题。它们不是每台机器都必然出现；只有遇到同类现象时再按对应小节处理。所有命令都在服务器上执行。

### 9.1 GitHub HTTPS 克隆被中断

现象是 `GnuTLS recv error (-110)`、`TLS connection was non-properly terminated` 或 `git pull` 超时。先让 Git 使用 HTTP/1.1，再做一次浅克隆：

```bash
git config --global http.version HTTP/1.1
git clone --depth 1 <你的仓库HTTPS地址> ~/agent
```

网络偶发失败时直接重试 `git pull --ff-only` 即可；不要为此关闭 TLS 校验，也不要在命令中写入 Token。

### 9.2 Docker Hub 能拉小镜像、但拉不到 Python 基础镜像

本次服务器的阿里云 Docker 镜像加速地址可以拉到 `hello-world`，却对 `python:3.12-slim` 返回 `not found`。项目因此把基础镜像地址参数化，并在生产模板中固定为已经验证可用的：

```dotenv
PYTHON_IMAGE=docker.m.daocloud.io/library/python:3.12-slim
```

先单独验证，再构建应用：

```bash
docker pull docker.m.daocloud.io/library/python:3.12-slim
docker compose up -d --build
```

该镜像代理是网络兼容方案，不是业务依赖；将来 Docker Hub 直连稳定后可将 `PYTHON_IMAGE` 改回官方 `python:3.12-slim`。

### 9.3 Tailscale Serve 的 HTTPS 证书未签发

`tailscale serve status` 已显示代理，但浏览器或 `curl` TLS 握手失败时，先看日志：

```bash
journalctl -u tailscaled --since "15 minutes ago" --no-pager | tail -n 100
```

若日志包含 `acme-v02.api.letsencrypt.org` 的 DNS 超时，说明 Tailscale 无法通过 ACME 申请证书。先确认系统可解析该域名：

```bash
tailscale set --accept-dns=false
resolvectl dns eth0 223.5.5.5 223.6.6.6
resolvectl flush-caches
resolvectl query acme-v02.api.letsencrypt.org
```

`eth0` 必须替换为 `resolvectl status` 中实际承载默认路由的网卡名。解析恢复后重建 Serve 配置：

```bash
tailscale serve --https=443 off
tailscale serve --bg --https=443 http://127.0.0.1:8000
tailscale serve status
```

Tailscale 会异步申请证书。必要时可用 Tailscale IP 验证完整链路（把三个占位符替换为 `tailscale serve status` 的主机名和 `tailscale status` 的 IP）：

```bash
curl -vk --resolve <主机名>.ts.net:443:<Tailscale-IP> https://<主机名>.ts.net/
```

拿到 `HTTP/2 200` 且证书颁发者为 Let's Encrypt 后，使用同一 Tailnet 内、已登录 Tailscale 的电脑在浏览器访问该 HTTPS 地址。不要把该地址当作公网网站；`tailscale serve` 默认仅 Tailnet 可见。

### 9.4 时间与时区验收

宿主机的时钟必须由 NTP 同步，应用的 `TIME_ZONE` 则决定中文报时所用的时区。两者分别验证：

```bash
timedatectl set-timezone Asia/Shanghai
timedatectl set-ntp true
date
date -u
timedatectl status
docker compose exec agent date
```

期望结果：宿主机和容器的本地时间均为 `CST (+0800)`，UTC 比本地慢 8 小时，`System clock synchronized: yes`。应用从本版本起会将“现在几点 / 今天几号 / 今天星期几”直接按 `TIME_ZONE` 返回，不再交由模型自由生成；因此升级后可在网页新建会话验证一次。

### 9.5 本次环境的最终验收结果

以下项目均已完成：Docker 容器 `healthy`、`/health` 返回 `ok: true`、应用端口仅绑定 `127.0.0.1:8000`、Tailscale Serve 的 HTTPS 证书已签发、Tailnet 内浏览器能打开网页、宿主机与容器的 NTP/中国时区一致。

API Key 曾出现在本地编辑器/聊天上下文时，应当按“已泄露”处理：去模型供应商控制台撤销旧 Key、创建新 Key、更新 `~/agent/.env`，再执行 `docker compose up -d --force-recreate`。Key 绝不能提交到 Git 仓库、文档或日志。
