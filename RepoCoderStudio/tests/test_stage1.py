"""Stage 1 smoke tests — exercise data loading + evaluation offline (no model download).

The generation step (which needs torch/transformers + weights) is covered by the notebook and the
CLI on a GPU host; here we verify the data contract and metric wiring on CPU.
"""

from repocoder.common.config import Config
from repocoder.stage1_documentation import data, evaluate
from repocoder.stage1_documentation.run import _clean, build_prompt


def _cfg(**over):
    base = {"dataset": {"name": "synthetic"}, "limit": None}
    base.update(over)
    return Config(base)


def test_synthetic_records_have_contract():
    records = data.load_dataset(_cfg())
    assert len(records) >= 3
    for r in records:
        assert {"id", "code", "reference", "language"} <= set(r)


def test_limit_is_respected():
    records = data.load_dataset(_cfg(limit=2))
    assert len(records) == 2


def test_build_prompt_includes_code():
    record = data.load_dataset(_cfg())[0]
    prompt = build_prompt(record, tokenizer=None)
    assert record["code"].splitlines()[0] in prompt


def test_clean_takes_first_meaningful_line():
    assert _clean("\n# heading\nReturn the sum.\nmore") == "heading"
    assert _clean("Return the sum of a and b.") == "Return the sum of a and b."


def test_evaluate_perfect_predictions_score_high():
    records = data.load_dataset(_cfg())
    refs = [r["reference"] for r in records]
    # Predict the references verbatim -> ROUGE-L should be perfect, CodeBLEU high.
    res = evaluate.evaluate(refs, refs, metrics=("codebleu", "rouge_l"))
    assert res["rouge_l"] == 1.0
    assert res["codebleu"] > 0.9
    assert res["n"] == len(records)


def test_evaluate_skips_codebertscore_without_backend_gracefully():
    records = data.load_dataset(_cfg())
    refs = [r["reference"] for r in records]
    res = evaluate.evaluate(refs, refs, metrics=("rouge_l", "codebertscore"))
    # Either it computed (backend installed) or it recorded a skip — never raised.
    assert ("codebertscore_f1" in res) or ("codebertscore_skipped" in res)
