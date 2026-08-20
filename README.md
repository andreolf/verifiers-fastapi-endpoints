# fastapi-endpoints

An evaluation of a model's ability to write **correct FastAPI code** from a natural-language spec
(or to repair a broken snippet). Built with the verifiers **v0** API (`load_environment` →
`vf.SingleTurnEnv`), the format the Environments Hub installs today.

Scoring combines a cheap **static floor** with an **execution-based** signal: each generated app
is started under FastAPI's `TestClient` in an isolated subprocess and hit with real requests, so
the reward reflects actual behavior — not surface pattern-matching.

## What it tests

16 tasks across three difficulty tiers, covering: basic routes, integer path params, Pydantic
request bodies, optional query params, explicit status codes (`201`/`204`), `APIRouter` with a
prefix, dependency injection (`Depends`), `response_model`, header params (`Header`), query
constraints (`Query(le=...)`), and two "fix the broken snippet" repair tasks.

Crucially it also exercises **error paths**, which are where models most often go wrong: a `404`
via `HTTPException`, automatic `422` on a malformed Pydantic body, `422` on a missing required
header, and `422` on an out-of-range query parameter. Each of these is asserted by a real
request, not by static inspection.

## Reward

| check | weight | reward when |
|---|---:|---|
| `valid_python` | 0.1 | the snippet parses as Python |
| `constructs_app` | 0.1 | it instantiates `FastAPI()` |
| `correct_method` | 0.1 | it declares a route with the required HTTP method |
| `endpoint_behaves` | **0.7** | fraction of behavior checks passing — the app is run with `TestClient` and each request's status code + JSON body is asserted |

Execution dominates by design: the static checks are a partial-credit floor, and behavior is the
real signal. A model can't score well by emitting plausible boilerplate — the endpoint has to
actually respond correctly.

## Validation (no model/API key needed)

The rubric is validated against crafted good/bad solutions, including the real execution path:

```bash
python -m venv .venv && . .venv/bin/activate
pip install fastapi httpx
python validate_fastapi_env.py
```

Expected: every correct solution (health, POST+Pydantic, optional query param, status 201,
router, `Depends`, `response_model`, repair) scores **1.00**; a syntactically-valid app that
returns the wrong body scores **0.30** (static floor only); non-code scores **0.00**.

## Run a real eval

```bash
uv pip install -e .
export OPENAI_API_KEY=sk-...            # and OPENAI_BASE_URL=... for a non-OpenAI provider
uv run vf-eval fastapi-endpoints -n 11 -r 3
```

Sample rollouts and the reward distribution are written to `outputs/`.

## Roadmap toward a Hub-quality FastAPI eval

- **More breadth/difficulty:** nested/enum bodies, header & cookie params, form data, background
  tasks, error paths (422/404), and multi-endpoint apps with shared state.
- **Harden the sandbox:** the behavior reward currently runs the model's code in a subprocess
  with a timeout — fine locally, but on the Hub/training this belongs in verifiers' sandboxed
  runtime; migrate to that for untrusted execution.
- **Calibrated baseline:** commit `outputs/` from a real `vf-eval` run and record a frontier
  model's score here so reviewers can see the reward is neither saturated nor floored.

## Safety note

The `endpoint_behaves` reward executes model-generated code. It runs in a subprocess with a
timeout as a local approximation; do not run untrusted completions outside a proper sandbox.

---

*Built as a reference environment for the Prime Intellect Environments Hub "Software Library
Evals" program. Uses the verifiers v0 `load_environment` API.*
