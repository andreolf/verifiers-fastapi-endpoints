"""Validate the real fastapi_endpoints.load_environment() and its reward logic without
installing the full verifiers framework.

We inject minimal stubs for `verifiers` and `datasets` so the actual environment module imports
and runs, then score crafted good/bad completions and assert the rubric behaves.
"""

import asyncio
import sys
import types
from pathlib import Path

# --- stub `verifiers` ---
vf = types.ModuleType("verifiers")


class Environment:  # return-type placeholder
    pass


class Rubric:
    def __init__(self, funcs, weights=None):
        self.funcs = funcs
        self.weights = weights or [1.0] * len(funcs)


class SingleTurnEnv(Environment):
    def __init__(self, dataset=None, system_prompt=None, rubric=None):
        self.dataset = dataset
        self.system_prompt = system_prompt
        self.rubric = rubric


vf.Environment = Environment
vf.Rubric = Rubric
vf.SingleTurnEnv = SingleTurnEnv
sys.modules["verifiers"] = vf

# --- stub `datasets` ---
ds = types.ModuleType("datasets")


class Dataset(list):
    @classmethod
    def from_list(cls, rows):
        return cls(rows)


ds.Dataset = Dataset
sys.modules["datasets"] = ds

# --- import the real environment module (flat repo: module sits beside this script) ---
sys.path.insert(0, str(Path(__file__).parent))
import fastapi_endpoints as env_mod  # noqa: E402

env = env_mod.load_environment()
assert isinstance(env, SingleTurnEnv), "load_environment must return a SingleTurnEnv"
assert len(env.dataset) == 5, f"expected 5 tasks, got {len(env.dataset)}"
assert env.rubric is not None and len(env.rubric.funcs) == 4
assert env.rubric.weights == [0.2, 0.2, 0.4, 0.2]
print(f"OK: env loaded — {len(env.dataset)} tasks, {len(env.rubric.funcs)} reward fns, "
      f"weights sum={sum(env.rubric.weights)}")


def completion(text):
    return [{"role": "assistant", "content": text}]


async def score(comp, info):
    total = 0.0
    parts = {}
    for fn, w in zip(env.rubric.funcs, env.rubric.weights):
        # reward fns accept (completion, info=..., **_); pass both, extras ignored
        try:
            r = await fn(completion=comp, info=info)
        except TypeError:
            r = await fn(completion=comp)
        parts[fn.__name__] = r
        total += w * r
    return total, parts


# Task 3: POST /users with a Pydantic body -> needs_model True
post_info = {"method": "post", "path": "/users", "needs_model": True, "path_params": []}

GOOD_POST = completion(
    """```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    name: str
    age: int

@app.post("/users")
def create_user(user: User):
    return user
```"""
)

WRONG_PATH = completion(
    """```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    name: str
    age: int

@app.post("/user")   # wrong path
def create_user(user: User):
    return user
```"""
)

NO_MODEL = completion(
    """```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/users")
def create_user(name: str, age: int):   # no Pydantic model
    return {"name": name, "age": age}
```"""
)

GARBAGE = completion("here is some prose, not code at all")


async def main():
    t_good, p_good = await score(GOOD_POST, post_info)
    t_path, p_path = await score(WRONG_PATH, post_info)
    t_nomodel, p_nomodel = await score(NO_MODEL, post_info)
    t_garbage, p_garbage = await score(GARBAGE, post_info)

    print(f"\ngood POST      total={t_good:.2f}  {p_good}")
    print(f"wrong path     total={t_path:.2f}  {p_path}")
    print(f"missing model  total={t_nomodel:.2f}  {p_nomodel}")
    print(f"garbage        total={t_garbage:.2f}  {p_garbage}")

    # assertions: reward must discriminate (approx to avoid float-repr noise)
    def approx(a, b):
        return abs(a - b) < 1e-9

    assert approx(t_good, 1.0), f"perfect answer should score 1.0, got {t_good}"
    assert approx(t_path, 0.6), f"wrong path should lose the 0.4 route weight -> 0.6, got {t_path}"
    assert approx(t_nomodel, 0.8), f"missing required model should lose 0.2 -> 0.8, got {t_nomodel}"
    assert approx(t_garbage, 0.0), f"non-code should score 0.0, got {t_garbage}"
    print("\nALL ASSERTIONS PASSED — reward is discriminative and correctly weighted.")


asyncio.run(main())
