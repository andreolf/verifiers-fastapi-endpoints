"""fastapi_endpoints: a single-turn eval of a model's ability to write correct FastAPI code.

Each task gives a natural-language spec for one FastAPI endpoint. The model returns a Python
snippet; the rubric parses it statically (no code execution, no network) and scores it against
the spec via several weighted checks: valid Python, app construction, the right route
method+path, path/query params, and Pydantic request models when required.

This is a v0-API environment (`load_environment` -> `vf.SingleTurnEnv`), the format the
Environments Hub installs today. See README.md for design notes and how to run it.
"""

from __future__ import annotations

import ast
import re

import verifiers as vf
from datasets import Dataset

SYSTEM_PROMPT = (
    "You are an expert Python developer. Write a single, self-contained FastAPI code snippet "
    "that satisfies the request. Return only a Python code block; do not explain."
)

# Each row: a spec plus the machine-checkable requirements used by the rubric.
# `info` is a dict; the environment parses JSON-string info automatically, but a dict is fine too.
TASKS: list[dict] = [
    {
        "question": "Create a FastAPI app with a GET endpoint at '/health' that returns {'status': 'ok'}.",
        "info": {"method": "get", "path": "/health", "needs_model": False, "path_params": []},
    },
    {
        "question": "Create a FastAPI app with a GET endpoint at '/items/{item_id}' that takes an "
        "integer path parameter item_id and returns it.",
        "info": {"method": "get", "path": "/items/{item_id}", "needs_model": False, "path_params": ["item_id"]},
    },
    {
        "question": "Create a FastAPI app with a POST endpoint at '/users' that accepts a JSON body "
        "with fields 'name' (str) and 'age' (int) using a Pydantic model, and returns the created user.",
        "info": {"method": "post", "path": "/users", "needs_model": True, "path_params": []},
    },
    {
        "question": "Create a FastAPI app with a DELETE endpoint at '/items/{item_id}' that deletes an "
        "item by its integer id and returns a confirmation message.",
        "info": {"method": "delete", "path": "/items/{item_id}", "needs_model": False, "path_params": ["item_id"]},
    },
    {
        "question": "Create a FastAPI app with a PUT endpoint at '/items/{item_id}' that accepts a "
        "Pydantic body with 'name' (str) and 'price' (float) and updates the item.",
        "info": {"method": "put", "path": "/items/{item_id}", "needs_model": True, "path_params": ["item_id"]},
    },
]


def _extract_code(text: str) -> str:
    """Pull the first fenced code block, or fall back to the whole reply."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def _iter_route_decorators(tree: ast.AST):
    """Yield (method, path) for every FastAPI-style route decorator in the module."""
    methods = {"get", "post", "put", "delete", "patch", "options", "head"}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            # matches @app.get("/x") / @router.post("/x") etc.
            if (
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Attribute)
                and dec.func.attr in methods
                and dec.args
                and isinstance(dec.args[0], ast.Constant)
                and isinstance(dec.args[0].value, str)
            ):
                yield dec.func.attr, dec.args[0].value


def load_environment(**kwargs) -> vf.Environment:
    dataset = Dataset.from_list(TASKS)

    async def valid_python(completion, **_) -> float:
        """0.2 — the snippet parses as Python at all (gate for the rest)."""
        try:
            ast.parse(_extract_code(completion[-1]["content"]))
            return 1.0
        except SyntaxError:
            return 0.0

    async def constructs_app(completion, **_) -> float:
        """0.2 — instantiates FastAPI()."""
        code = _extract_code(completion[-1]["content"])
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0.0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "FastAPI":
                return 1.0
        return 0.0

    async def correct_route(completion, info, **_) -> float:
        """0.4 — declares a route with the required HTTP method and exact path."""
        code = _extract_code(completion[-1]["content"])
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0.0
        want = (info["method"], info["path"])
        return 1.0 if want in set(_iter_route_decorators(tree)) else 0.0

    async def uses_pydantic_when_required(completion, info, **_) -> float:
        """0.2 — defines a Pydantic BaseModel iff the task requires a request body model."""
        code = _extract_code(completion[-1]["content"])
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0.0
        defines_model = any(
            isinstance(node, ast.ClassDef)
            and any(isinstance(b, ast.Name) and b.id == "BaseModel" for b in node.bases)
            for node in ast.walk(tree)
        )
        return 1.0 if defines_model == bool(info["needs_model"]) else 0.0

    rubric = vf.Rubric(
        funcs=[valid_python, constructs_app, correct_route, uses_pydantic_when_required],
        weights=[0.2, 0.2, 0.4, 0.2],
    )

    return vf.SingleTurnEnv(dataset=dataset, system_prompt=SYSTEM_PROMPT, rubric=rubric)
