"""fastapi_endpoints: an eval of a model's ability to write correct FastAPI code.

Each task gives a natural-language spec (or a broken snippet to repair) for a FastAPI app. The
model returns a Python snippet, scored by a rubric that combines:

- a cheap **static floor** (the code parses, constructs `FastAPI()`, and declares a route with the
  required HTTP method), and
- an **execution-based** signal (the app is started with FastAPI's `TestClient` in an isolated
  subprocess and real requests are asserted against expected status codes and JSON bodies).

Execution is weighted 0.7 — behavior is the real signal, and it can't be gamed by emitting
boilerplate. This is the verifiers **v0** API (`load_environment` -> `vf.SingleTurnEnv`).

Local note: the behavior reward runs the model's code in a subprocess with a timeout. On the
Hub / in training this runs inside verifiers' sandboxed runtime; the subprocess is the local
approximation. Do not run untrusted completions outside a sandbox.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import verifiers as vf
from datasets import Dataset

SYSTEM_PROMPT = (
    "You are an expert Python developer. Write a single, self-contained FastAPI code snippet "
    "that satisfies the request. Assume `fastapi` is installed. Return only a Python code block; "
    "do not explain."
)

# Each task: a natural-language (or repair) prompt plus machine-checkable requirements.
# `method` drives the static route-method check; `exec` drives the behavior check (a list of
# requests with expected status and an optional JSON subset the response must match).
TASKS: list[dict] = [
    {
        "question": "Create a FastAPI app with a GET endpoint at '/health' that returns the JSON "
        "object {'status': 'ok'}.",
        "info": {"tier": 1, "method": "get",
                 "exec": [{"method": "get", "path": "/health", "status": 200, "json_subset": {"status": "ok"}}]},
    },
    {
        "question": "Create a FastAPI app with a GET endpoint at '/items/{item_id}' that takes an "
        "integer path parameter item_id and returns {'item_id': item_id}.",
        "info": {"tier": 1, "method": "get",
                 "exec": [{"method": "get", "path": "/items/42", "status": 200, "json_subset": {"item_id": 42}}]},
    },
    {
        "question": "Create a FastAPI app with a POST endpoint at '/users' that accepts a JSON body "
        "with fields 'name' (str) and 'age' (int) via a Pydantic model, and returns the created user as JSON.",
        "info": {"tier": 2, "method": "post",
                 "exec": [{"method": "post", "path": "/users", "json": {"name": "Ada", "age": 36},
                           "status": 200, "json_subset": {"name": "Ada", "age": 36}}]},
    },
    {
        "question": "Create a FastAPI app with a GET endpoint at '/search' that has an optional query "
        "parameter 'q' (string, default None) and returns {'q': q}.",
        "info": {"tier": 2, "method": "get",
                 "exec": [{"method": "get", "path": "/search?q=cat", "status": 200, "json_subset": {"q": "cat"}},
                          {"method": "get", "path": "/search", "status": 200, "json_subset": {"q": None}}]},
    },
    {
        "question": "Create a FastAPI app with a POST endpoint at '/items' that accepts a Pydantic body "
        "with 'name' (str), creates the item, and returns HTTP status 201 with body {'name': name}.",
        "info": {"tier": 2, "method": "post",
                 "exec": [{"method": "post", "path": "/items", "json": {"name": "pen"},
                           "status": 201, "json_subset": {"name": "pen"}}]},
    },
    {
        "question": "Create a FastAPI app with a DELETE endpoint at '/items/{item_id}' that deletes an "
        "item by its integer id and returns {'deleted': item_id}.",
        "info": {"tier": 1, "method": "delete",
                 "exec": [{"method": "delete", "path": "/items/5", "status": 200, "json_subset": {"deleted": 5}}]},
    },
    {
        "question": "Create a FastAPI app that includes an APIRouter mounted with prefix '/api'. The "
        "router has a GET '/ping' endpoint returning {'ping': 'pong'}. Mount the router on the app.",
        "info": {"tier": 3, "method": "get",
                 "exec": [{"method": "get", "path": "/api/ping", "status": 200, "json_subset": {"ping": "pong"}}]},
    },
    {
        "question": "Create a FastAPI app with a GET endpoint at '/whoami' that resolves the current user "
        "through a dependency using Depends (the dependency returns the fixed username 'guest'), and "
        "returns {'user': username}.",
        "info": {"tier": 3, "method": "get",
                 "exec": [{"method": "get", "path": "/whoami", "status": 200, "json_subset": {"user": "guest"}}]},
    },
    {
        "question": "Create a FastAPI app that defines a Pydantic model User with fields id (int) and "
        "name (str). Add a GET '/users/{user_id}' endpoint declared with response_model=User that "
        "returns a user with the given id and name 'Ada'.",
        "info": {"tier": 3, "method": "get",
                 "exec": [{"method": "get", "path": "/users/7", "status": 200, "json_subset": {"id": 7, "name": "Ada"}}]},
    },
    {
        "question": "The following FastAPI code has a bug: it returns the wrong shape. Fix it so GET "
        "'/ping' returns the JSON object {'msg': 'pong'}.\n\n```python\nfrom fastapi import FastAPI\n"
        "app = FastAPI()\n\n@app.get('/ping')\ndef ping():\n    return 'pong'\n```",
        "info": {"tier": 2, "method": "get",
                 "exec": [{"method": "get", "path": "/ping", "status": 200, "json_subset": {"msg": "pong"}}]},
    },
    {
        "question": "This FastAPI code has a bug: the path parameter is treated as a string, so it "
        "concatenates instead of doubling. Fix it so GET '/double/{item_id}' parses an integer and "
        "returns {'doubled': item_id * 2}.\n\n```python\nfrom fastapi import FastAPI\napp = FastAPI()\n\n"
        "@app.get('/double/{item_id}')\ndef double(item_id):\n    return {'doubled': item_id * 2}\n```",
        "info": {"tier": 2, "method": "get",
                 "exec": [{"method": "get", "path": "/double/5", "status": 200, "json_subset": {"doubled": 10}}]},
    },
]

# Runs inside an isolated subprocess: load the model's app, exercise it with TestClient.
_HARNESS = r'''
import json, sys, importlib.util
from pathlib import Path


def load_app(path):
    spec = importlib.util.spec_from_file_location("submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from fastapi import FastAPI
    app = getattr(mod, "app", None)
    if isinstance(app, FastAPI):
        return app
    for value in vars(mod).values():
        if isinstance(value, FastAPI):
            return value
    return None


def main():
    submission, spec = sys.argv[1], sys.argv[2]
    checks = json.loads(Path(spec).read_text())
    out = {"passed": 0, "total": len(checks)}
    try:
        from fastapi.testclient import TestClient
        app = load_app(submission)
        if app is None:
            print(json.dumps({**out, "error": "no FastAPI app found"})); return
        client = TestClient(app)
    except Exception as exc:  # report, don't crash the scorer
        print(json.dumps({**out, "error": f"load: {exc!r}"})); return

    passed = 0
    for check in checks:
        try:
            resp = client.request(check["method"].upper(), check["path"], json=check.get("json"))
            ok = resp.status_code == check["status"]
            if ok and "json_subset" in check:
                body = resp.json()
                ok = isinstance(body, dict) and all(body.get(k) == v for k, v in check["json_subset"].items())
            passed += int(ok)
        except Exception:  # a failed request is just a miss
            pass
    print(json.dumps({"passed": passed, "total": len(checks)}))


main()
'''


def _info(info) -> dict:
    """Reward fns receive `info` as a dict (verifiers parses JSON-string info); tolerate both."""
    return json.loads(info) if isinstance(info, str) else info


def _extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


def _iter_route_methods(tree: ast.AST):
    verbs = {"get", "post", "put", "delete", "patch", "options", "head"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr in verbs:
                    yield dec.func.attr


def _run_behavior_checks(code: str, checks: list[dict], timeout: int = 25) -> dict:
    with tempfile.TemporaryDirectory() as d:
        dp = Path(d)
        (dp / "submission.py").write_text(code)
        (dp / "checks.json").write_text(json.dumps(checks))
        (dp / "harness.py").write_text(_HARNESS)
        try:
            proc = subprocess.run(
                [sys.executable, str(dp / "harness.py"), str(dp / "submission.py"), str(dp / "checks.json")],
                capture_output=True, text=True, timeout=timeout, cwd=str(dp),
            )
        except subprocess.TimeoutExpired:
            return {"passed": 0, "total": len(checks)}
        lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
        try:
            return json.loads(lines[-1]) if lines else {"passed": 0, "total": len(checks)}
        except json.JSONDecodeError:
            return {"passed": 0, "total": len(checks)}


def load_environment(**kwargs) -> vf.Environment:
    dataset = Dataset.from_list(
        [{"question": t["question"], "info": json.dumps(t["info"])} for t in TASKS]
    )

    async def valid_python(completion, **_) -> float:
        try:
            ast.parse(_extract_code(completion[-1]["content"]))
            return 1.0
        except SyntaxError:
            return 0.0

    async def constructs_app(completion, **_) -> float:
        try:
            tree = ast.parse(_extract_code(completion[-1]["content"]))
        except SyntaxError:
            return 0.0
        return 1.0 if any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "FastAPI"
            for n in ast.walk(tree)
        ) else 0.0

    async def correct_method(completion, info, **_) -> float:
        try:
            tree = ast.parse(_extract_code(completion[-1]["content"]))
        except SyntaxError:
            return 0.0
        return 1.0 if _info(info)["method"] in set(_iter_route_methods(tree)) else 0.0

    async def endpoint_behaves(completion, info, **_) -> float:
        checks = _info(info).get("exec") or []
        if not checks:
            return 0.0
        code = _extract_code(completion[-1]["content"])
        result = await asyncio.to_thread(_run_behavior_checks, code, checks)
        total = result.get("total") or len(checks)
        return result.get("passed", 0) / total if total else 0.0

    rubric = vf.Rubric(
        funcs=[valid_python, constructs_app, correct_method, endpoint_behaves],
        weights=[0.1, 0.1, 0.1, 0.7],
    )
    return vf.SingleTurnEnv(dataset=dataset, system_prompt=SYSTEM_PROMPT, rubric=rubric)
