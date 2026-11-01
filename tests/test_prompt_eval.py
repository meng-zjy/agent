"""评估器失败报告回归，不调用真实模型。"""

from types import SimpleNamespace

from scripts import eval_prompts
from src.core.tools import ToolRegistry


def test_evaluation_continues_after_request_failure_without_recording_exception_text(monkeypatch):
    class UnavailableService:
        _system_prompt = "测试人设"
        _agent = SimpleNamespace(_tools=ToolRegistry())

        def new_session(self):
            return SimpleNamespace(id="isolated")

        def send_events(self, session_id, question):
            raise RuntimeError("private-provider-error-do-not-record")

    monkeypatch.setattr(eval_prompts.bootstrap, "build_service", lambda retriever: UnavailableService())
    report = eval_prompts.run_evaluation()
    assert report["total"] == 7
    assert report["passed"] == 0
    assert all(case["error_type"] == "RuntimeError" for case in report["cases"])
    assert "private-provider-error-do-not-record" not in str(report)
