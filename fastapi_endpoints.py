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
    {
        "question": "Create a FastAPI app with a GET endpoint at '/items/{item_id}' (integer). If "
        "item_id == 1, return {'item_id': 1, 'name': 'widget'}; otherwise raise an HTTPException with "
        "status code 404.",
        "info": {"tier": 3, "method": "get",
                 "exec": [{"method": "get", "path": "/items/1", "status": 200,
                           "json_subset": {"item_id": 1, "name": "widget"}},
                          {"method": "get", "path": "/items/2", "status": 404}]},
    },
    {
        "question": "Create a FastAPI app with a POST endpoint at '/signup' that accepts a Pydantic "
        "body with 'email' (str) and 'age' (int) and returns it. Rely on FastAPI's automatic request "
        "validation for malformed bodies.",
        "info": {"tier": 3, "method": "post",
                 "exec": [{"method": "post", "path": "/signup", "json": {"email": "a@b.com", "age": 30},
                           "status": 200, "json_subset": {"email": "a@b.com", "age": 30}},
                          {"method": "post", "path": "/signup", "json": {"email": "a@b.com", "age": "not-an-int"},
                           "status": 422}]},
    },
    {
        "question": "Create a FastAPI app with a GET endpoint at '/whoami' that reads a required request "
        "header 'X-User' (use fastapi.Header) and returns {'user': <that header value>}.",
        "info": {"tier": 3, "method": "get",
                 "exec": [{"method": "get", "path": "/whoami", "headers": {"X-User": "alice"},
                           "status": 200, "json_subset": {"user": "alice"}},
                          {"method": "get", "path": "/whoami", "status": 422}]},
    },
    {
        "question": "Create a FastAPI app with a GET endpoint at '/search' that has a query parameter "
        "'limit' (int) with default 10, constrained to be at most 100 using Query(le=100). Return "
        "{'limit': limit}.",
        "info": {"tier": 3, "method": "get",
                 "exec": [{"method": "get", "path": "/search?limit=50", "status": 200, "json_subset": {"limit": 50}},
                          {"method": "get", "path": "/search", "status": 200, "json_subset": {"limit": 10}},
                          {"method": "get", "path": "/search?limit=200", "status": 422}]},
    },
    {
        "question": "Create a FastAPI app with a DELETE endpoint at '/items/{item_id}' (integer) that "
        "returns HTTP status 204 with no response body.",
        "info": {"tier": 2, "method": "delete",
                 "exec": [{"method": "delete", "path": "/items/3", "status": 204}]},
    },
    {
        "question": "Create a FastAPI app with a POST endpoint at '/orders' that accepts a Pydantic body "
        "with a nested 'customer' object ({'name': str, 'email': str}) and a 'total' (float), using "
        "nested Pydantic models, and returns the order.",
        "info": {"tier": 3, "method": "post",
                 "exec": [{"method": "post", "path": "/orders",
                           "json": {"customer": {"name": "Ada", "email": "a@b.com"}, "total": 9.5},
                           "status": 200,
                           "json_subset": {"customer": {"name": "Ada", "email": "a@b.com"}, "total": 9.5}},
                          {"method": "post", "path": "/orders",
                           "json": {"customer": {"name": "Ada"}, "total": 9.5}, "status": 422}]},
    },
    {
        "question": "Create a FastAPI app with a GET endpoint at '/color/{name}' where 'name' is a string "
        "Enum with members red, green, blue. Return {'color': name}. Rely on FastAPI validation for "
        "invalid values.",
        "info": {"tier": 3, "method": "get",
                 "exec": [{"method": "get", "path": "/color/red", "status": 200, "json_subset": {"color": "red"}},
                          {"method": "get", "path": "/color/purple", "status": 422}]},
    },
    {
        "question": "Create a FastAPI app with an in-memory store. POST '/notes' accepts a Pydantic body "
        "{'text': str}, assigns an incrementing integer id starting at 1, stores the note, and returns "
        "{'id': id, 'text': text}. GET '/notes/{note_id}' returns the stored note, or raises 404 if it "
        "does not exist.",
        "info": {"tier": 3, "method": "post",
                 "exec": [{"method": "post", "path": "/notes", "json": {"text": "hi"},
                           "status": 200, "json_subset": {"id": 1, "text": "hi"}},
                          {"method": "get", "path": "/notes/1", "status": 200, "json_subset": {"id": 1, "text": "hi"}},
                          {"method": "get", "path": "/notes/999", "status": 404}]},
    },
    {
        "question": "Create a FastAPI app defining a Pydantic model UserOut with only 'username' (str). "
        "Add a GET '/me' endpoint declared with response_model=UserOut that returns a dict "
        "{'username': 'ada', 'password': 'secret'}, so the password is filtered out of the response.",
        "info": {"tier": 3, "method": "get",
                 "exec": [{"method": "get", "path": "/me", "status": 200,
                           "json_subset": {"username": "ada"}, "json_absent": ["password"]}]},
    },
    {
        "question": "Create a FastAPI app with a PUT endpoint at '/items/{item_id}' (integer) that accepts "
        "a Pydantic body with 'name' (str) and 'price' (float) and returns {'id': item_id, 'name': name, "
        "'price': price}.",
        "info": {"tier": 2, "method": "put",
                 "exec": [{"method": "put", "path": "/items/7", "json": {"name": "pen", "price": 1.5},
                           "status": 200, "json_subset": {"id": 7, "name": "pen", "price": 1.5}}]},
    },
    {
        "question": "Create a FastAPI app with a GET '/items' endpoint that serves from a fixed in-memory "
        "list of five items [{'id': 1}, {'id': 2}, {'id': 3}, {'id': 4}, {'id': 5}], using query params "
        "'limit' (int, default 10) and 'offset' (int, default 0), returning items[offset:offset+limit] "
        "as a JSON list.",
        "info": {"tier": 3, "method": "get",
                 "exec": [{"method": "get", "path": "/items?limit=2&offset=1", "status": 200,
                           "json_equals": [{"id": 2}, {"id": 3}]},
                          {"method": "get", "path": "/items", "status": 200,
                           "json_equals": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}]}]},
    },
    {
        "question": "Create a FastAPI app with a POST '/upload' endpoint that accepts an uploaded file "
        "(UploadFile) in a form field named 'file', reads its contents, and returns "
        "{'filename': <name>, 'size': <number of bytes>}.",
        "info": {"tier": 3, "method": "post",
                 "exec": [{"method": "post", "path": "/upload", "files": {"file": ["note.txt", "hello"]},
                           "status": 200, "json_subset": {"filename": "note.txt", "size": 5}}]},
    },
    {
        "question": "Create a FastAPI app with a GET '/secure' endpoint that uses a dependency reading the "
        "'x-api-key' header (via Header) and raises HTTPException 401 unless it equals 'letmein'. On "
        "success return {'ok': True}.",
        "info": {"tier": 3, "method": "get",
                 "exec": [{"method": "get", "path": "/secure", "headers": {"x-api-key": "letmein"},
                           "status": 200, "json_subset": {"ok": True}},
                          {"method": "get", "path": "/secure", "headers": {"x-api-key": "nope"},
                           "status": 401}]},
    },
    {
        "question": "Create a FastAPI app with a POST '/log' endpoint that accepts a Pydantic body "
        "{'msg': str}, uses BackgroundTasks to append msg to an in-memory list, and returns HTTP 202 with "
        "{'queued': True}. Add a GET '/logs' endpoint returning {'logs': [...]} with the accumulated "
        "messages.",
        "info": {"tier": 3, "method": "post",
                 "exec": [{"method": "post", "path": "/log", "json": {"msg": "hi"},
                           "status": 202, "json_subset": {"queued": True}},
                          {"method": "get", "path": "/logs", "status": 200, "json_subset": {"logs": ["hi"]}}]},
    },
    {
        "question": "Create a FastAPI app with a PATCH '/items/{item_id}' (integer) that accepts a Pydantic "
        "body where 'name' (str) and 'price' (float) are both optional (default None), and returns "
        "{'id': item_id, 'updated': <dict of only the fields the client actually sent>} using exclude_unset.",
        "info": {"tier": 3, "method": "patch",
                 "exec": [{"method": "patch", "path": "/items/3", "json": {"name": "pen"},
                           "status": 200, "json_subset": {"id": 3, "updated": {"name": "pen"}}}]},
    },
    {
        "question": "Create a FastAPI app with a WebSocket endpoint at '/ws' that accepts the connection, "
        "receives a text message, and sends back 'echo: ' followed by that message.",
        "info": {"tier": 3, "method": "websocket",
                 "exec": [{"ws": {"path": "/ws", "send": "hi", "expect": "echo: hi"}}]},
    },
    {
        "question": "Create a FastAPI app with a GET '/dashboard' endpoint that reads a cookie named "
        "'session' (via Cookie) and returns {'user': 'ada'} if it equals 'valid', otherwise raises "
        "HTTPException 401.",
        "info": {"tier": 3, "method": "get",
                 "exec": [{"method": "get", "path": "/dashboard", "headers": {"Cookie": "session=valid"},
                           "status": 200, "json_subset": {"user": "ada"}},
                          {"method": "get", "path": "/dashboard", "headers": {"Cookie": "session=bad"},
                           "status": 401}]},
    },
    {
        "question": "Create a FastAPI app that defines a custom exception class TeapotError and registers "
        "an exception handler (via @app.exception_handler) returning a JSONResponse with status 418 and "
        "body {'error': 'teapot'}. Add a GET '/brew' endpoint that raises TeapotError.",
        "info": {"tier": 3, "method": "get",
                 "exec": [{"method": "get", "path": "/brew", "status": 418, "json_subset": {"error": "teapot"}}]},
    },
    {
        "question": "Create a FastAPI app with a parent APIRouter mounted at prefix '/api' that includes a "
        "child APIRouter mounted at prefix '/v1'. The child has a GET '/status' endpoint returning "
        "{'status': 'up'}, so the final path is '/api/v1/status'.",
        "info": {"tier": 3, "method": "get",
                 "exec": [{"method": "get", "path": "/api/v1/status", "status": 200,
                           "json_subset": {"status": "up"}}]},
    },
    {
        "question": "Create a FastAPI app with a GET '/old' endpoint that returns a RedirectResponse to "
        "'/new' with status code 307.",
        "info": {"tier": 2, "method": "get",
                 "exec": [{"method": "get", "path": "/old", "no_redirect": True, "status": 307,
                           "resp_headers": {"location": "/new"}}]},
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
            if "ws" in check:
                # websocket sub-check: connect, send text, expect an echoed reply
                spec = check["ws"]
                with client.websocket_connect(spec["path"]) as ws:
                    ws.send_text(spec["send"])
                    got = ws.receive_text()
                passed += int(got == spec["expect"])
                continue
            files = None
            if "files" in check:
                # each entry is [filename, text-content]; sent as multipart/form-data
                files = {k: (v[0], v[1].encode() if isinstance(v[1], str) else v[1]) for k, v in check["files"].items()}
            resp = client.request(
                check["method"].upper(), check["path"],
                json=check.get("json"), headers=check.get("headers"),
                data=check.get("data"), files=files,
                follow_redirects=not check.get("no_redirect", False),
            )
            ok = resp.status_code == check["status"]
            if ok and "resp_headers" in check:
                ok = all(resp.headers.get(k.lower()) == v for k, v in check["resp_headers"].items())
            if ok and "json_equals" in check:
                ok = resp.json() == check["json_equals"]
            if ok and ("json_subset" in check or "json_absent" in check):
                body = resp.json()
                if isinstance(body, dict):
                    if "json_subset" in check:
                        ok = ok and all(body.get(k) == v for k, v in check["json_subset"].items())
                    if "json_absent" in check:
                        ok = ok and all(k not in body for k in check["json_absent"])
                else:
                    ok = False
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
    verbs = {"get", "post", "put", "delete", "patch", "options", "head", "websocket"}
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
