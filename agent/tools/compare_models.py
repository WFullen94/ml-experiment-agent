from __future__ import annotations

from itertools import combinations

from scipy.stats import ttest_rel, wilcoxon


def compare_models(fold_scores: dict[str, list[float]], metric: str) -> dict:
    """
    Pairwise comparison of models using both Wilcoxon signed-rank and paired t-test.

    Wilcoxon is non-parametric and robust but cannot reach p<0.05 with n=5 folds
    (min achievable ≈ 0.0625). The paired t-test assumes normally distributed
    differences but has more power with few observations. Reporting both lets the
    user see where they agree and where one has more sensitivity than the other.

    Parameters
    ----------
    fold_scores : {method: [score_per_fold]}  — from cross_validate output
    metric      : label propagated to the result dict

    Returns
    -------
    {
        "metric": str,
        "comparisons": [
            {
                "method_a": str,
                "method_b": str,
                "mean_a": float,
                "mean_b": float,
                "delta": float,       # mean_b - mean_a  (positive = b wins)
                "wilcoxon_p": float,
                "ttest_p": float,
                "significant": bool,  # True if either test p < 0.05
                "tests_agree": bool,  # both tests reach same significance conclusion
                "winner": str | None, # None if not significant by either test
            }
        ],
        "overall_winner": str | None,
        "note": str,
    }
    """
    methods = list(fold_scores.keys())
    comparisons = []

    for a, b in combinations(methods, 2):
        scores_a = fold_scores[a]
        scores_b = fold_scores[b]
        mean_a = round(sum(scores_a) / len(scores_a), 4)
        mean_b = round(sum(scores_b) / len(scores_b), 4)
        delta = round(mean_b - mean_a, 4)

        w_p = _wilcoxon_p(scores_a, scores_b)
        t_p = _ttest_p(scores_a, scores_b)

        w_sig = w_p < 0.05
        t_sig = t_p < 0.05
        significant = w_sig or t_sig
        winner = (b if delta > 0 else a) if significant else None

        comparisons.append({
            "method_a":    a,
            "method_b":    b,
            "mean_a":      mean_a,
            "mean_b":      mean_b,
            "delta":       delta,
            "wilcoxon_p":  round(w_p, 4),
            "ttest_p":     round(t_p, 4),
            "significant": significant,
            "tests_agree": w_sig == t_sig,
            "winner":      winner,
        })

    n_folds = len(next(iter(fold_scores.values()))) if fold_scores else 0
    overall_winner = _tally_winner(comparisons, methods)

    return {
        "metric":         metric,
        "comparisons":    comparisons,
        "overall_winner": overall_winner,
        "note": (
            f"n={n_folds} folds: Wilcoxon min p ≈ 0.0625 (non-parametric, robust); "
            f"paired t-test has more power but assumes normal differences. "
            f"'significant' = True if either test reaches p<0.05."
        ),
    }


# --- helpers -----------------------------------------------------------------

def _wilcoxon_p(a: list[float], b: list[float]) -> float:
    diffs = [x - y for x, y in zip(a, b)]
    if all(d == 0 for d in diffs):
        return 1.0
    try:
        _, p = wilcoxon(a, b)
        return float(p)
    except ValueError:
        return 1.0


def _ttest_p(a: list[float], b: list[float]) -> float:
    diffs = [x - y for x, y in zip(a, b)]
    if all(d == 0 for d in diffs):
        return 1.0
    try:
        _, p = ttest_rel(a, b)
        return float(p)
    except Exception:
        return 1.0


def _tally_winner(comparisons: list[dict], methods: list[str]) -> str | None:
    if not comparisons:
        return methods[0] if len(methods) == 1 else None
    wins = {m: 0 for m in methods}
    for c in comparisons:
        if c["winner"]:
            wins[c["winner"]] += 1
    best = max(wins, key=lambda m: wins[m])
    return best if wins[best] > 0 else None
