"""CPU-only smoke + unit tests for shared utilities.

These run with just the standard library + pytest (no torch/transformers/GPU), so the core harness
can be validated anywhere — including CI and a fresh Colab cell.
"""

from repocoder.common import env
from repocoder.common.config import Config
from repocoder.common.metrics import ast_similarity, code_bleu, rouge_l
from repocoder.common.metrics.execution import execution_accuracy, run_python
from repocoder.common.metrics.pass_k import pass_at_k
from repocoder.common.metrics.retrieval import retrieval_metrics


# --- env -------------------------------------------------------------------
def test_detect_platform_returns_known_value():
    assert env.detect_platform() in {env.COLAB, env.KAGGLE, env.LOCAL}


def test_get_device_is_valid():
    assert env.get_device() in {"cuda", "mps", "cpu"}


def test_summary_has_expected_keys():
    s = env.summary()
    assert {"platform", "device", "supports_4bit", "python"} <= set(s)


# --- config ----------------------------------------------------------------
def test_config_dotted_access_and_get_path():
    cfg = Config({"model": {"id": "x"}, "limit": 5})
    assert cfg.model.id == "x"
    assert cfg.get_path("model.id") == "x"
    assert cfg.get_path("model.missing", "d") == "d"


# --- pass@k ----------------------------------------------------------------
def test_pass_at_k_all_correct_is_one():
    out = pass_at_k([[True], [True]], k=1)
    assert out["pass@1"] == 1.0


def test_pass_at_k_all_wrong_is_zero():
    out = pass_at_k([[False, False]], k=1)
    assert out["pass@1"] == 0.0


def test_pass_at_k_partial():
    # 1 correct of 2 samples -> pass@1 == 0.5
    out = pass_at_k([[True, False]], k=1)
    assert abs(out["pass@1"] - 0.5) < 1e-9


# --- execution sandbox ------------------------------------------------------
def test_run_python_success():
    assert run_python("assert 1 + 1 == 2").passed


def test_run_python_failure():
    assert not run_python("assert False").passed


def test_run_python_timeout():
    res = run_python("while True: pass", timeout_s=1.0)
    assert res.timed_out and not res.passed


def test_execution_accuracy_mixed():
    sols = ["def f():\n    return 2", "def f():\n    return 3"]
    tests = ["assert f() == 2", "assert f() == 2"]
    out = execution_accuracy(sols, tests)
    assert out["passed"] == 1 and out["total"] == 2


# --- retrieval --------------------------------------------------------------
def test_retrieval_metrics_perfect_top1():
    out = retrieval_metrics(retrieved=[["a", "b", "c"]], relevant=[{"a"}], k=3)
    assert out["mrr@3"] == 1.0
    assert abs(out["recall@3"] - 1.0) < 1e-9


def test_retrieval_metrics_second_rank():
    out = retrieval_metrics(retrieved=[["x", "a"]], relevant=[{"a"}], k=2)
    assert abs(out["mrr@2"] - 0.5) < 1e-9


# --- text/code metrics (fallback backends, no heavy deps) -------------------
def test_rouge_l_identical_is_one():
    out = rouge_l(["a b c d"], ["a b c d"])
    assert out["rouge_l"] == 1.0


def test_codebleu_identical_high():
    out = code_bleu(["def f(): return 1"], ["def f(): return 1"], lang="python")
    assert out["codebleu"] > 0.9


def test_ast_similarity_identical_python_is_one():
    code = "def f(x):\n    return x + 1"
    out = ast_similarity([code], [code], lang="python")
    assert out["ast_similarity"] == 1.0
    assert out["backend"] in {"ast", "fallback_tokens"}
