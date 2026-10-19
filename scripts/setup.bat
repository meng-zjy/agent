@echo off
REM ============================================================
REM  一键装环境：venv + pip install + .env
REM  适用 Windows（项目当前在 Windows 11 验证，见 docs/11 §4.5）
REM  用法：双击，或命令行执行 scripts\setup.bat
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0.."

echo.
echo [1/4] 检查 Python...
python --version >nul 2>&1 || (echo.  错误：未找到 python。请先安装 Python 3.11+ 并加入 PATH。& exit /b 1)
for /f "tokens=2" %%v in ('python --version') do set PYV=%%v
echo      找到 Python %PYV%

echo.
echo [2/4] 创建虚拟环境 venv...
if exist "venv\Scripts\activate.bat" (
    echo      已存在，跳过
) else (
    python -m venv venv || (echo.  错误：venv 创建失败。& exit /b 1)
    echo      已创建
)

echo.
echo [3/4] 安装依赖（首次较慢，约 1-3 分钟）...
call venv\Scripts\activate.bat || (echo.  错误：venv 激活失败。& exit /b 1)
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt || (echo.  错误：依赖安装失败。& exit /b 1)
echo      完成

echo.
echo [4/4] 准备 .env...
if exist ".env" (
    echo      已存在，跳过
) else (
    copy .env.example .env >nul || (echo.  错误：.env 创建失败。& exit /b 1)
    echo.
    echo  ╔══════════════════════════════════════════════════╗
    echo  ║  请编辑 .env，填入你的 DEEPSEEK_API_KEY           ║
    echo  ║  （其他配置项均有默认值，可不改）                  ║
    echo  ╚══════════════════════════════════════════════════╝
    echo.
)

echo.
echo ============================================================
echo  装好了。两种运行方式：
echo.
echo    Web 界面：  call venv\Scripts\activate.bat
echo                python -m src.api
echo                然后浏览器打开 http://localhost:8000
echo.
echo    命令行：    call venv\Scripts\activate.bat
echo                python -m src.cli
echo.
echo  跑测试：      call venv\Scripts\activate.bat
echo                python -m pytest -q
echo ============================================================
echo.
