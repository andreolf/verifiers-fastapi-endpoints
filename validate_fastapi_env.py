"""Validate fastapi_endpoints.load_environment() and its rubric without the full verifiers
framework. We stub `verifiers`/`datasets`, then score crafted good/bad completions — including
the real execution-based reward, which runs each app under FastAPI's TestClient in a subprocess.

Run with a Python that has `fastapi` + `httpx` installed (the behavior reward shells out to
`sys.executable`):  python validate_fastapi_env.py
"""

import asyncio
import sys
import types
from pathlib import Path

# --- stub `verifiers` ---
vf = types.ModuleType("verifiers")


class Environment:
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
assert isinstance(env, SingleTurnEnv)
assert len(env.dataset) == len(env_mod.TASKS) >= 36
assert env.rubric.weights == [0.1, 0.1, 0.1, 0.7]
print(f"OK: env loaded — {len(env.dataset)} tasks, weights {env.rubric.weights}")


def comp(code_block):
    return [{"role": "assistant", "content": f"```python\n{code_block}\n```"}]


async def score(completion, info):
    total, parts = 0.0, {}
    for fn, w in zip(env.rubric.funcs, env.rubric.weights):
        try:
            r = await fn(completion=completion, info=info)
        except TypeError:
            r = await fn(completion=completion)
        parts[fn.__name__] = round(r, 3)
        total += w * r
    return round(total, 3), parts


def info_for(i):
    return env_mod.TASKS[i]["info"]


# --- crafted correct solutions, one per representative task type ---
GOOD = {
    0: "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/health')\ndef health():\n    return {'status': 'ok'}",
    2: "from fastapi import FastAPI\nfrom pydantic import BaseModel\napp = FastAPI()\nclass User(BaseModel):\n    name: str\n    age: int\n@app.post('/users')\ndef create(u: User):\n    return u",
    3: "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/search')\ndef search(q: str | None = None):\n    return {'q': q}",
    4: "from fastapi import FastAPI\nfrom pydantic import BaseModel\napp = FastAPI()\nclass Item(BaseModel):\n    name: str\n@app.post('/items', status_code=201)\ndef create(i: Item):\n    return {'name': i.name}",
    6: "from fastapi import FastAPI, APIRouter\napp = FastAPI()\nrouter = APIRouter(prefix='/api')\n@router.get('/ping')\ndef ping():\n    return {'ping': 'pong'}\napp.include_router(router)",
    7: "from fastapi import FastAPI, Depends\napp = FastAPI()\ndef current_user():\n    return 'guest'\n@app.get('/whoami')\ndef whoami(user: str = Depends(current_user)):\n    return {'user': user}",
    8: "from fastapi import FastAPI\nfrom pydantic import BaseModel\napp = FastAPI()\nclass User(BaseModel):\n    id: int\n    name: str\n@app.get('/users/{user_id}', response_model=User)\ndef get_user(user_id: int):\n    return {'id': user_id, 'name': 'Ada'}",
    10: "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/double/{item_id}')\ndef double(item_id: int):\n    return {'doubled': item_id * 2}",
    11: "from fastapi import FastAPI, HTTPException\napp = FastAPI()\n@app.get('/items/{item_id}')\ndef get_item(item_id: int):\n    if item_id == 1:\n        return {'item_id': 1, 'name': 'widget'}\n    raise HTTPException(status_code=404)",
    12: "from fastapi import FastAPI\nfrom pydantic import BaseModel\napp = FastAPI()\nclass Signup(BaseModel):\n    email: str\n    age: int\n@app.post('/signup')\ndef signup(s: Signup):\n    return s",
    13: "from fastapi import FastAPI, Header\napp = FastAPI()\n@app.get('/whoami')\ndef whoami(x_user: str = Header(...)):\n    return {'user': x_user}",
    14: "from fastapi import FastAPI, Query\napp = FastAPI()\n@app.get('/search')\ndef search(limit: int = Query(10, le=100)):\n    return {'limit': limit}",
    15: "from fastapi import FastAPI, Response\napp = FastAPI()\n@app.delete('/items/{item_id}', status_code=204)\ndef delete_item(item_id: int):\n    return Response(status_code=204)",
    16: "from fastapi import FastAPI\nfrom pydantic import BaseModel\napp = FastAPI()\nclass Customer(BaseModel):\n    name: str\n    email: str\nclass Order(BaseModel):\n    customer: Customer\n    total: float\n@app.post('/orders')\ndef create_order(o: Order):\n    return o",
    17: "from enum import Enum\nfrom fastapi import FastAPI\nclass Color(str, Enum):\n    red = 'red'\n    green = 'green'\n    blue = 'blue'\napp = FastAPI()\n@app.get('/color/{name}')\ndef get_color(name: Color):\n    return {'color': name}",
    18: "from fastapi import FastAPI, HTTPException\nfrom pydantic import BaseModel\napp = FastAPI()\nclass Note(BaseModel):\n    text: str\n_notes = {}\n_seq = {'n': 0}\n@app.post('/notes')\ndef create_note(n: Note):\n    _seq['n'] += 1\n    nid = _seq['n']\n    _notes[nid] = {'id': nid, 'text': n.text}\n    return _notes[nid]\n@app.get('/notes/{note_id}')\ndef get_note(note_id: int):\n    if note_id not in _notes:\n        raise HTTPException(status_code=404)\n    return _notes[note_id]",
    19: "from fastapi import FastAPI\nfrom pydantic import BaseModel\napp = FastAPI()\nclass UserOut(BaseModel):\n    username: str\n@app.get('/me', response_model=UserOut)\ndef me():\n    return {'username': 'ada', 'password': 'secret'}",
    20: "from fastapi import FastAPI\nfrom pydantic import BaseModel\napp = FastAPI()\nclass ItemUpdate(BaseModel):\n    name: str\n    price: float\n@app.put('/items/{item_id}')\ndef update_item(item_id: int, body: ItemUpdate):\n    return {'id': item_id, 'name': body.name, 'price': body.price}",
    21: "from fastapi import FastAPI\napp = FastAPI()\n_items = [{'id': 1}, {'id': 2}, {'id': 3}, {'id': 4}, {'id': 5}]\n@app.get('/items')\ndef list_items(limit: int = 10, offset: int = 0):\n    return _items[offset:offset + limit]",
    22: "from fastapi import FastAPI, UploadFile\napp = FastAPI()\n@app.post('/upload')\nasync def upload(file: UploadFile):\n    contents = await file.read()\n    return {'filename': file.filename, 'size': len(contents)}",
    23: "from fastapi import FastAPI, Depends, Header, HTTPException\napp = FastAPI()\ndef require_key(x_api_key: str = Header(...)):\n    if x_api_key != 'letmein':\n        raise HTTPException(status_code=401)\n@app.get('/secure')\ndef secure(_=Depends(require_key)):\n    return {'ok': True}",
    24: "from fastapi import FastAPI, BackgroundTasks\nfrom pydantic import BaseModel\napp = FastAPI()\nclass Msg(BaseModel):\n    msg: str\n_logs = []\ndef _append(m):\n    _logs.append(m)\n@app.post('/log', status_code=202)\ndef log(m: Msg, bg: BackgroundTasks):\n    bg.add_task(_append, m.msg)\n    return {'queued': True}\n@app.get('/logs')\ndef logs():\n    return {'logs': _logs}",
    25: "from fastapi import FastAPI\nfrom pydantic import BaseModel\napp = FastAPI()\nclass ItemPatch(BaseModel):\n    name: str | None = None\n    price: float | None = None\n@app.patch('/items/{item_id}')\ndef patch_item(item_id: int, body: ItemPatch):\n    return {'id': item_id, 'updated': body.model_dump(exclude_unset=True)}",
    26: "from fastapi import FastAPI, WebSocket\napp = FastAPI()\n@app.websocket('/ws')\nasync def ws(websocket: WebSocket):\n    await websocket.accept()\n    msg = await websocket.receive_text()\n    await websocket.send_text('echo: ' + msg)",
    27: "from fastapi import FastAPI, Cookie, HTTPException\napp = FastAPI()\n@app.get('/dashboard')\ndef dashboard(session: str | None = Cookie(default=None)):\n    if session != 'valid':\n        raise HTTPException(status_code=401)\n    return {'user': 'ada'}",
    28: "from fastapi import FastAPI, Request\nfrom fastapi.responses import JSONResponse\napp = FastAPI()\nclass TeapotError(Exception):\n    pass\n@app.exception_handler(TeapotError)\nasync def teapot_handler(request: Request, exc: TeapotError):\n    return JSONResponse(status_code=418, content={'error': 'teapot'})\n@app.get('/brew')\ndef brew():\n    raise TeapotError()",
    29: "from fastapi import FastAPI, APIRouter\napp = FastAPI()\nchild = APIRouter(prefix='/v1')\n@child.get('/status')\ndef status():\n    return {'status': 'up'}\nparent = APIRouter(prefix='/api')\nparent.include_router(child)\napp.include_router(parent)",
    30: "from fastapi import FastAPI\nfrom fastapi.responses import RedirectResponse\napp = FastAPI()\n@app.get('/old')\ndef old():\n    return RedirectResponse('/new', status_code=307)",
    31: "from fastapi import FastAPI\nfrom fastapi.responses import StreamingResponse\napp = FastAPI()\ndef gen():\n    yield 'a'\n    yield 'b'\n    yield 'c'\n@app.get('/stream')\ndef stream():\n    return StreamingResponse(gen())",
    32: "from typing import Annotated\nfrom fastapi import FastAPI, Depends\napp = FastAPI()\ndef current_user():\n    return 'guest'\n@app.get('/whoami2')\ndef whoami(user: Annotated[str, Depends(current_user)]):\n    return {'user': user}",
    33: "from fastapi import FastAPI\napp = FastAPI()\nsubapp = FastAPI()\n@subapp.get('/ping')\ndef sub_ping():\n    return {'sub': 'pong'}\napp.mount('/sub', subapp)",
    34: "from fastapi import FastAPI, Request\napp = FastAPI()\n@app.middleware('http')\nasync def add_header(request: Request, call_next):\n    response = await call_next(request)\n    response.headers['X-Custom'] = 'yes'\n    return response\n@app.get('/ping')\ndef ping():\n    return {'ping': 'pong'}",
    35: "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/widgets')\ndef widgets():\n    return []",
}

# static passes (valid + FastAPI() + get route) but behavior wrong -> should be 0.3
WRONG_BEHAVIOR = "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/health')\ndef health():\n    return {'status': 'DOWN'}"
GARBAGE = "here is some prose, not code at all"


async def main():
    failures = []

    for i, code in GOOD.items():
        total, parts = await score(comp(code), info_for(i))
        tag = env_mod.TASKS[i]["question"][:46].replace("\n", " ")
        print(f"task {i:2} GOOD   total={total:.2f}  {parts}  | {tag}")
        if total != 1.0:
            failures.append(f"task {i} good expected 1.0 got {total}")

    t_wrong, p_wrong = await score(comp(WRONG_BEHAVIOR), info_for(0))
    print(f"\nwrong-behavior total={t_wrong:.2f}  {p_wrong}  (static floor only)")
    if t_wrong != 0.3:
        failures.append(f"wrong-behavior expected 0.3 got {t_wrong}")

    t_g, p_g = await score([{"role": "assistant", "content": GARBAGE}], info_for(0))
    print(f"garbage        total={t_g:.2f}  {p_g}")
    if t_g != 0.0:
        failures.append(f"garbage expected 0.0 got {t_g}")

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("ALL ASSERTIONS PASSED — static floor + execution-based rewards behave correctly.")


asyncio.run(main())
