from __future__ import annotations

from itertools import combinations

from scipy.stats import wilcoxon


def compare_models(fold_scores: dict[str, list[float]], metric: str) -> dict:
    """
    Pairwise Wilcoxon signed-rank test on matched CV fold scores.

    Parameters
    ----------
    fold_scores : {method: [score_per_fold]}  — from cross_validate output
    metric      : label for the result dict

    Returns
    -------
    {
        "metric": str,
        "comparisons": [
            {
                "method_a": str, "method_b": str,
                "mean_a": float, "mean_b": float,
                "delta": float,          # mean_b - mean_a (positive = b wins)
                "p_value": float,
                "significant": bool,     # p < 0.05
                "winner": str | None,    # None if not significant
            }
        ],
        "overall_winner": str | None,    # method with most significant wins
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

        p_value = _wilcoxon_p(scores_a, scores_b)
        significant = p_value < 0.05
        winner = (b if delta > 0 else a) if significant else None

        comparisons.append({
            "method_a":   a,
            "method_b":   b,
            "mean_a":     mean_a,
            "mean_b":     mean_b,
            "delta":      delta,
            "p_value":    round(p_value, 4),
            "significant": significant,
            "winner":     winner,
        })

    overall_winner = _tally_winner(comparisons, methods)
    n_folds = len(next(iter(fold_scores.values()))) if fold_scores else 0

    return {
        "metric":         metric,
        "comparisons":    comparisons,
        "overall_winner": overall_winner,
        # Wilcoxon with n=5 folds cannot reach p<0.05 (min achievable ≈ 0.0625).
        # Use overall_winner as the directional recommendation; treat p_value as
        # indicative only. More folds or a paired t-test would give more power.
        "note": f"limited statistical power: {n_folds} folds (Wilcoxon min p ≈ 0.0625)",
    }


# --- helpers -----------------------------------------------------------------

def _wilcoxon_p(a: list[float], b: list[float]) -> float:
    diffs = [x - y for x, y in zip(a, b)]
    if all(d == 0 for d in diffs):
        return 1.0   # identical scores — no difference detectable
    try:
        _, p = wilcoxon(a, b)
        return float(p)
    except ValueError:
        # wilcoxon raises if n < 1 or all differences are zero
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
