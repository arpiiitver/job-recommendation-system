"""§21 evaluation gates + metric exactness (plan §21).

Golden data, datasets, and trained artifacts are committed, so these gates
run offline.  Cases that need an unavailable dependency (ranklet not yet
trained in a fresh clone, embedder failure) skip with an explicit reason
instead of passing vacuously — `python ml/train.py` produces everything.
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://x:x@localhost:55432/x")
os.environ.setdefault("EMBEDDING_BACKEND", "minilm")

BACKEND = Path(__file__).resolve().parents[1]
GOLDEN = BACKEND / "ml" / "data" / "golden"
DATASETS = BACKEND / "ml" / "datasets"
ARTIFACTS = BACKEND / "ml" / "artifacts"
REPORTS = BACKEND / "ml" / "reports"


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


# ---------------------------------------------------------------------------
# Metric exactness (hand-computed)
# ---------------------------------------------------------------------------

def test_ndcg_at_k_exact():
    from ml.metrics import ndcg_at_k

    labels = [3, 2, 0, 2]
    dcg = 7 / 1 + 3 / math.log2(3) + 0 + 3 / math.log2(5)
    idcg = 7 / 1 + 3 / math.log2(3) + 3 / math.log2(4) + 0
    assert ndcg_at_k(labels, 4) == pytest.approx(dcg / idcg, rel=1e-9)
    assert ndcg_at_k([3, 2, 1], 3) == pytest.approx(1.0)   # perfect ordering
    assert ndcg_at_k([0, 0, 0], 3) == 0.0                   # nothing relevant
    assert ndcg_at_k([2, 0, 0], 1) == pytest.approx(1.0)    # top-1 hit
    # single relevant item ranked last — strictly between 0 and 1
    assert 0.0 < ndcg_at_k([0, 2], 2) < 1.0


def test_precision_recall_mrr_exact():
    from ml.metrics import mrr_at_k, precision_at_k, recall_at_k

    labels = [2, 0, 2, 0]
    assert precision_at_k(labels, 2) == pytest.approx(0.5)
    assert recall_at_k(labels, 2) == pytest.approx(0.5)
    assert mrr_at_k(labels, 4) == pytest.approx(1.0)
    assert mrr_at_k([0, 0, 2], 3) == pytest.approx(1 / 3)
    assert mrr_at_k([0, 0, 0], 3) == 0.0
    assert precision_at_k([0, 0], 2) == 0.0
    # k beyond the list retrieves every relevant item; partial case: 1 of 3
    assert recall_at_k([1, 1, 1], 10) == 1.0
    assert recall_at_k([1, 0, 1, 0, 1], 2) == pytest.approx(1 / 3)


def test_brier_ece_f1_exact():
    from ml.metrics import (
        brier_score,
        expected_calibration_error,
        micro_f1,
        precision_recall_f1,
    )

    probs = [0.9, 0.8, 0.3, 0.4]
    outcomes = [1, 1, 0, 0]
    assert brier_score(probs, outcomes) == pytest.approx(0.075)
    # 10-bin ECE: bin9 |1-0.9|×1/4 + bin8 |1-0.8|×1/4 + bin3 0.3×1/4 +
    # bin4 0.4×1/4 = (0.1 + 0.2 + 0.3 + 0.4)/4 = 0.25
    assert expected_calibration_error(probs, outcomes) == pytest.approx(0.25)

    p, r, f1 = precision_recall_f1(
        [True, True, False, False], [True, True, True, False]
    )
    assert p == pytest.approx(2 / 3)
    assert r == pytest.approx(1.0)
    assert f1 == pytest.approx(0.8)

    assert micro_f1([({"a", "b", "c"}, {"a", "b"})]) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Golden data shape (plan §21 fixtures)
# ---------------------------------------------------------------------------

def test_golden_corpus_shape():
    postings = _jsonl(GOLDEN / "postings.jsonl")
    resumes = _jsonl(GOLDEN / "resumes.jsonl")
    transfers = _jsonl(GOLDEN / "transfer_pairs.jsonl")
    relevance = _jsonl(GOLDEN / "relevance_pairs.jsonl")

    assert len(postings) == 100
    assert len(resumes) == 50
    positives = [p for p in transfers if p["transferable"]]
    negatives = [p for p in transfers if not p["transferable"]]
    assert len(positives) >= 100, "plan §21: ≥100 transferable pairs"
    assert len(negatives) >= 100, "plan §21: ≥100 non-transferable pairs"
    assert len(relevance) >= 18, "plan §21: labeled semantic queries"
    # every resume has exact expected skills
    assert all(r["expected"]["skills"] for r in resumes)


# ---------------------------------------------------------------------------
# §21 gates
# ---------------------------------------------------------------------------

def test_extraction_f1_gate():
    from ml.evaluate import extraction_report

    report = extraction_report()
    assert "skipped" not in report, report
    assert report["f1"] >= 0.85, (
        f"skill extraction F1 {report['f1']} < 0.85 gate: {report}"
    )


def _report(name: str) -> dict:
    path = REPORTS / name
    assert path.exists(), f"{path.name} missing — run python ml/evaluate_all.py"
    return json.loads(path.read_text(encoding="utf-8"))


def test_skill_graph_heldout_gate():
    """M1: held-out O*NET occupation->technology links (20% hidden per
    occupation). The old golden transfer pairs are NOT used as a gate: they
    were the clusters the previous model was trained on (circular)."""
    m1 = _report("master_project_evaluation.json")["M1"]
    assert m1["status"] == "ok"
    gnn = json.loads((ARTIFACTS / "skillgraph_gnn_v7.json").read_text(encoding="utf-8"))
    evals = gnn["evaluation"]
    ours = evals["GraphSAGE (this model)"]["E1_link_prediction_test"]["auc"]
    popularity = evals["Popularity (most-used tools)"]["E1_link_prediction_test"]["auc"]
    assert ours >= 0.90, f"GraphSAGE held-out link AUC {ours} < 0.90"
    assert ours > popularity, f"GraphSAGE {ours} must beat the popularity baseline {popularity}"
    assert gnn["calibration"]["min_prob"] > 0


def test_semantic_recall_gate():
    from ml.evaluate import semantic_recall

    report = semantic_recall(20)
    if "skipped" in report:
        pytest.skip(report["skipped"])
    assert report["recall_at_20"] >= 0.75, (
        f"Recall@20 {report['recall_at_20']} < 0.75 gate: {report}"
    )


def test_ranker_beats_formula_gate():
    """M2: DCN-v2 on held-out CareerBuilder users (later weeks) must beat the
    hand-written weighted-sum formula on NDCG@10 and AUC."""
    report = json.loads((ARTIFACTS / "dcn_ranker_v2.metrics.json").read_text(encoding="utf-8"))
    assert report["reported_on"] == "test", "gate must use the held-out test split"
    dcn, formula = report["dcn_v2"], report["baseline_heuristic"]
    assert dcn["n_eval_queries"] >= 500
    assert dcn["ndcg@10"] > formula["ndcg@10"] + 0.05, (dcn, formula)
    assert dcn["auc_roc"] > formula["auc_roc"], (dcn, formula)


def test_calibration_ece_gate():
    """M3: calibrator fitted on the validation week, ECE measured on the
    held-out test week, must be well calibrated and no worse than raw."""
    m3 = _report("master_project_evaluation.json")["M3"]
    test = m3["held_out_test"]
    assert test["n"] >= 1000
    assert test["ece_after"] <= 0.05, f"held-out ECE {test['ece_after']} > 0.05"
    assert test["ece_after"] <= test["ece_before"]
    tiers = m3["match_tiers"]["test"]
    # tiers must be ordered by real apply rate: top > strong > possible
    assert tiers["top"]["apply_rate"] > tiers["strong"]["apply_rate"] > tiers["possible"]["apply_rate"]


def test_serving_artifacts_present_and_loadable():
    """The models the app serves load from committed artifacts."""
    import app.services.calibration as calibration
    import app.services.rerank as rerank_mod

    for pattern in ("dcn_ranker_v*.pt", "calibrator_v*.joblib", "skill2vec_v*.joblib"):
        assert sorted(ARTIFACTS.glob(pattern)), f"missing serving artifact {pattern}"

    rerank_mod.reset_cache()
    model, meta, version = rerank_mod._load_dcn()
    assert model is not None and version, "committed DCN ranker must load for serving"
    assert meta["feature_order"] == rerank_mod.FEATURE_ORDER

    calibration.load_calibrator.cache_clear()
    p_low, p_high = calibration.predict_positive(20.0, logit=-3.0), calibration.predict_positive(80.0, logit=2.0)
    assert p_low is not None and p_high is not None and 0.0 <= p_low <= p_high <= 1.0

    from app.core.config import get_settings

    settings = get_settings()
    assert settings.recommender_strategy in ("deterministic", "ltr_blend", "ltr_full")
    assert settings.readiness_calibrator_enabled is True
    assert settings.readiness_band_high > settings.readiness_band_medium


# ---------------------------------------------------------------------------
# Feature vector contract + honest-degradation guards
# ---------------------------------------------------------------------------

def test_feature_vector_deterministic_and_serving_sized():
    from app.models.enums import ExperienceLevel, ProficiencyLevel
    from app.services.rerank import FEATURE_ORDER, FeatureContext, build_features
    from app.services.scoring import ComponentInputs, JobSkillReq, score_pair

    inputs = ComponentInputs(
        seeker_skills={"python": (ProficiencyLevel.ADVANCED, 3.0)},
        job_skills=[
            JobSkillReq(name="python", is_required=True),
            JobSkillReq(name="kubernetes", is_required=True),
        ],
        semantic_cosine=0.77,
        years_of_experience=5.0,
        job_level=ExperienceLevel.MID,
    )
    result = score_pair(inputs)
    ctx = FeatureContext(job_age_days=3.5, applicant_count=2.0,
                         profile_completeness=80.0, historical_apply_rate=0.04)
    v1 = build_features(inputs, result, ctx)
    v2 = build_features(inputs, result, ctx)
    assert v1 == v2, "feature build must be deterministic"
    assert len(v1) == len(FEATURE_ORDER) == 31
    assert len(set(FEATURE_ORDER)) == len(FEATURE_ORDER), "feature names must be unique"
    assert all(isinstance(x, (int, float)) and math.isfinite(x) for x in v1)
    # required coverage: python held, kubernetes missing -> 0.5
    assert v1[FEATURE_ORDER.index("required_coverage_ratio")] == pytest.approx(0.5)
    assert v1[FEATURE_ORDER.index("raw_bi_encoder_cosine")] == pytest.approx(0.77)
    assert v1[FEATURE_ORDER.index("job_age_days")] == pytest.approx(3.5)


def test_dcn_feature_order_mismatch_degrades_honestly():
    """A ranker trained on a different feature order must not load (rows are
    built positionally) — serving then falls back instead of mis-scoring."""
    import torch

    import app.services.rerank as rerank_mod

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        torch.save(
            {"feature_order": ["wrong_feature"], "state_dict": {}, "version": "v99"},
            tmp_path / "dcn_ranker_v99.pt",
        )
        original = rerank_mod.ARTIFACT_DIR
        rerank_mod.ARTIFACT_DIR = tmp_path
        rerank_mod.reset_cache()
        try:
            assert rerank_mod._load_dcn() == (None, None, None)
        finally:
            rerank_mod.ARTIFACT_DIR = original
            rerank_mod.reset_cache()


def test_legacy_ranklet_is_not_served():
    """The old synthetic-label LightGBM ranklet (30 legacy features) is kept
    only for history: its manifest no longer matches serving, so it must be
    skipped rather than loaded with misaligned features."""
    import app.services.rerank as rerank_mod

    rerank_mod.reset_cache()
    booster, version, _order = rerank_mod._load_lgbm()
    assert booster is None and version is None
