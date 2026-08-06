# fastapi-endpoints

A single-turn evaluation of a model's ability to write **correct FastAPI endpoint code** from a
natural-language spec. Built with the verifiers **v0** API (`load_environment` →
`vf.SingleTurnEnv`), the format the Environments Hub installs today.

This is a compact, self-contained **learning-rep** environment: it demonstrates the exact shape
a "Software Library Evals" bounty submission needs (dataset + a multi-function weighted rubric)
without external services.

## What it tests

Given a spec like *"a POST endpoint at `/users` that accepts a Pydantic body with `name` and
`age`"*, the model must return a FastAPI snippet. The reward is a weighted sum of **static
checks** (the code is parsed with `ast`, never executed — so grading is deterministic, fast, and
safe):

| check | weight | reward when |
|---|---:|---|
| `valid_python` | 0.2 | the snippet parses as Python |
| `constructs_app` | 0.2 | it instantiates `FastAPI()` |
| `correct_route` | 0.4 | it declares a route with the required method **and** exact path |
| `uses_pydantic_when_required` | 0.2 | it defines a `BaseModel` iff the task needs a request body |

The route check is the highest-weighted signal; the Pydantic check is two-sided (penalizes
adding a model when none is needed), so the reward is discriminative rather than a checklist a
model can pass by dumping boilerplate.

## Run it

```bash
uv pip install -e .
# vf-eval needs an OpenAI-compatible endpoint:
export OPENAI_API_KEY=sk-...            # and OPENAI_BASE_URL=... for a non-OpenAI provider
uv run vf-eval fastapi-endpoints -n 5 -r 3
```

Sample rollouts and the reward distribution are written to `outputs/`.

## Design notes / how to extend toward the real bounty

- **Grow the dataset.** 5 hand-authored tasks are enough to validate the harness; a bounty
  submission wants breadth (dependencies with `Depends`, response models, status codes, query
  params with validation, routers, async DB deps) and difficulty tiers.
- **Add execution-based rewards.** Static checks are a floor. The strongest version spins up the
  app with FastAPI's `TestClient` in a sandbox and asserts real request/response behavior — a
  `vf.ToolEnv`/multi-turn variant, or an in-runtime verifier script (see the v1 `gsm8k_v1`
  pattern that runs a `uv` verifier inside the rollout runtime).
- **Baseline before publishing.** Record a known model's score in this README so reviewers can
  see the reward is calibrated (not saturated at 1.0 or stuck at 0).

## Status

Reward logic validated locally against crafted good/bad completions:

```bash
python3 validate_fastapi_env.py
# -> good POST 1.00, wrong path 0.60, missing model 0.80, non-code 0.00
```

A full `vf-eval` run requires a model endpoint/API key (see "Run it" above).

---

*Built as a warm-up reference environment for the Prime Intellect Environments Hub
"Software Library Evals" program. Uses the verifiers v0 `load_environment` API.*
