"""Evaluation of the ml-experiment-agent using the agent-eval harness.

This package drives the REAL agent — real methodology Gate, real sklearn
toolbox (real model training, cross-validation, and Wilcoxon/t-test model
comparison), and real scaffold generator. Only the LLM planner is replaced by a
deterministic ScriptedPlanner so the evaluation runs offline and reproducibly
(no ANTHROPIC_API_KEY required).

Everything under evaluation is real; only the plan *proposal* is scripted.
"""
