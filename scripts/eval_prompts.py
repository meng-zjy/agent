"""真实模型提示词回归（付费）：固定资料检索，不评估 embedding 召回质量。

python scripts/eval_prompts.py --output data/evals/prompts.json
所有对话均为虚构测试数据，MCP 关闭，会话和日志在独立临时目录中。
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import bootstrap
from src.config import settings
from src.core.agent import AgentStatus
from src.core.logger import reset
from src.core.rag import Chunk
from src.prompts import DEFAULT_SYSTEM_PROMPT, PROMPT_VERSION


class FixtureRetriever:
    """始终返回唯一测试资料，隔离检索质量与提示词遵循能力。"""

    def search(self, query, top_k=4):
        return [Chunk(
            text="星桥测试项目的负责人是林知夏，内部验收编号为 QV-7319。资料未记录预算。",
            source="星桥测试资料.md",
            index=0,
        )][:top_k]


def run_evaluation() -> dict:
    """运行固定案例，保留完整合成输入输出及显式检查结果。"""
    service = bootstrap.build_service(FixtureRetriever())
    results = []

    def ask(case_id, session_id, question, check):
        started = time.perf_counter()
        tool_names = []
        completed_tools = []
        pieces = []
        error = None
        registry = service._agent._tools
        execute = registry.execute

        def record_execution(name, arguments, context=None):
            output = execute(name, arguments, context)
            completed_tools.append(name)
            return output

        try:
            with patch.object(registry, "execute", side_effect=record_execution):
                for event in service.send_events(session_id, question):
                    if isinstance(event, AgentStatus):
                        tool_names.append(event.tool_name)
                    else:
                        pieces.append(event)
            checks = check("".join(pieces), completed_tools)
        except Exception as exc:  # 单个请求失败仍保留报告，避免丢掉已付费结果。
            error = type(exc).__name__
            checks = {"request_succeeded": False}
        result = {
            "id": case_id,
            "question": question,
            "answer": "".join(pieces),
            "tools": tool_names,
            "completed_tools": completed_tools,
            "checks": checks,
            "passed": all(checks.values()),
            "error_type": error,
            "seconds": round(time.perf_counter() - started, 2),
        }
        results.append(result)
        print(f"{case_id}: {'PASS' if result['passed'] else 'FAIL'} "
              f"tools={tool_names} checks={checks}", flush=True)

    memory_session = service.new_session().id
    ask("memory_setup", memory_session,
        "请记住：我叫沈小禾，喜欢的编程语言是 Rust。简单确认即可。",
        lambda answer, tools: {"acknowledged": bool(answer), "no_tools": not tools})

    # 重新构建服务，迫使从磁盘恢复会话；不能靠上一个实例的内存回答。
    service = bootstrap.build_service(FixtureRetriever())
    ask("memory_after_reload", memory_session,
        "我叫什么？我喜欢的编程语言是什么？",
        lambda answer, tools: {
            "name": "沈小禾" in answer,
            "language": "rust" in answer.lower(),
            "no_tools": not tools,
        })
    ask("memory_unknown", memory_session,
        "我的生日是几月几号？如果没有告诉过你，请明确说不知道，不要猜。",
        lambda answer, tools: {
            "acknowledges_unknown": any(word in answer for word in (
                "不知道", "不清楚", "没有告诉", "没告诉", "未提供", "没有提供", "不确定",
            )),
            "no_tools": not tools,
        })
    ask("calculator", service.new_session().id,
        "帮我算一下 137 乘以 29 再加 418，告诉我结果。",
        lambda answer, tools: {
            "calculator_called": "calculator" in tools,
            "correct_result": "4391" in answer.replace(",", "").replace("，", ""),
        })
    ask("knowledge_fact", service.new_session().id,
        "星桥测试项目的负责人是谁？内部验收编号是什么？请给出来源文件名。",
        lambda answer, tools: {
            "retrieval_called": "search_knowledge_base" in tools,
            "owner": "林知夏" in answer,
            "code": "QV-7319" in answer,
            "source": "星桥测试资料.md" in answer,
        })
    ask("knowledge_unknown", service.new_session().id,
        "星桥测试项目的预算是多少？请以知识库资料为准，没有记录就明确说没有。",
        lambda answer, tools: {
            "retrieval_called": "search_knowledge_base" in tools,
            "missing_budget": any(word in answer for word in (
                "未记录", "没有记录", "未提及", "没有提及", "未提供", "没有提供", "未记载",
            )),
        })
    ask("ordinary_chat", service.new_session().id,
        "你好，用一句话和我打个招呼。",
        lambda answer, tools: {"answered": bool(answer), "no_tools": not tools})

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_worktree_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()),
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in (
                "scripts/eval_prompts.py", "src/prompts.py", "src/bootstrap.py",
                "src/core/agent.py", "src/core/security.py", "src/core/tools.py",
            )
        },
        "prompt_version": PROMPT_VERSION,
        "provider": bootstrap.provider_label(),
        "system_prompt_sha256": hashlib.sha256(service._system_prompt.encode("utf-8")).hexdigest(),
        "custom_system_prompt": settings.system_prompt != DEFAULT_SYSTEM_PROMPT,
        "retrieval": "fixed_single_document_fixture",
        "limitations": "One sample per case; keyword checks require human review. No embedding, MCP or summary generation evaluation.",
        "passed": sum(result["passed"] for result in results),
        "total": len(results),
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/evals/prompts.json")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    # mkdtemp 的父目录固定在被 Git 忽略的 data 内，不接触真实会话或知识库。
    temporary_root = ROOT / "data/tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prompt_eval_", dir=temporary_root) as directory:
        workspace = Path(directory)
        with patch.object(settings, "session_dir", str(workspace / "sessions")), \
             patch.object(settings, "log_dir", str(workspace / "logs")), \
             patch.object(settings, "summary_keep_turns", 0), \
             patch.object(bootstrap, "build_mcp_tools", return_value=[]):
            reset()
            try:
                report = run_evaluation()
            finally:
                reset()  # Windows 下先释放日志句柄，再清理临时目录。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{report['passed']}/{report['total']} passed; report={args.output}")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
