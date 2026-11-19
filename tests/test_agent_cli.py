"""The public CLI selects actual strategies without online requests."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reclab_agent.__main__ import main
from reclab_agent.core import ModelTurn
from reclab_agent.evaluation import replay_trace


class Replies:
    def __init__(self, texts):
        self.texts = iter(texts)

    def respond(self, **kwargs):
        text = next(self.texts)
        return ModelTurn(output=[{"role": "assistant", "content": text}], output_text=text)


class AgentCliTests(unittest.TestCase):
    def test_old_default_and_selected_strategies_write_replayable_results(self):
        cases = {
            None: (["original answer"], "original answer", 1),
            "simple": (["direct answer"], "direct answer", 1),
            "plan_execute": ([json.dumps({"steps": ["inspect context"]}), "step result", "synthesis"],
                             "synthesis", 3),
            "reflection": (["draft", json.dumps({"approved": False, "feedback": "state uncertainty"}),
                            "revised answer"], "revised answer", 3),
        }
        for kind, (replies, expected, steps) in cases.items():
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
                trace = Path(temp) / "run.jsonl"
                arguments = ["reclab", "--model", "fixture", "--ask", "explain the task",
                             "--trace", str(trace)]
                if kind is not None:
                    arguments += ["--agent", kind]
                output = io.StringIO()
                with patch("sys.argv", arguments), contextlib.redirect_stdout(output), \
                        patch("reclab_agent.__main__.OpenAIResponsesBackend", return_value=Replies(replies)):
                    main()
                result = json.loads(output.getvalue())
                self.assertEqual((result["status"], result["answer"], result["steps"]),
                                 ("completed", expected, steps))
                self.assertTrue(replay_trace(trace, [])["trace_valid"])

    def test_invalid_strategy_is_rejected_before_creating_backend(self):
        with patch("sys.argv", ["reclab", "--agent", "unknown"]), \
                contextlib.redirect_stderr(io.StringIO()), \
                patch("reclab_agent.__main__.OpenAIResponsesBackend") as backend:
            with self.assertRaises(SystemExit) as raised:
                main()
            self.assertEqual(raised.exception.code, 2)
            backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
