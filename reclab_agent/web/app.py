"""Local-only research workbench backed by the durable RunStore.

The default application stores requests but does not invent a configured live
agent or worker. The explicit demo endpoint runs a trusted synthetic fixture.
Install the ``service`` extra before importing this optional module.
"""
from __future__ import annotations
import asyncio
import inspect
import json
from pathlib import Path
from typing import Any, Callable
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from reclab_agent.domain.contracts import TaskSpec
from reclab_agent.memory.storage import RunStore


def _run_demo(state_root: Path) -> dict[str, Any]:
    # Keep demo startup optional and make the HTTP contract independently testable.
    from reclab_agent.examples.demo import run_demo
    return run_demo(state_root)


def create_app(root: Path, run_task: Callable | None = None) -> FastAPI:
    """Create a workbench for a repository root; bind the server to localhost.

    A supplied task runner receives ``(TaskSpec, run_id)`` and is responsible
    for writing lifecycle events, state, and its report to the same RunStore.
    Both synchronous and asynchronous runners are supported.
    """
    app = FastAPI(title="RecLab Agent Workbench", version="0.2.0")
    app.state.repository_root = Path(root)
    app.state.state_root = Path(root) / ".reclab"
    app.state.store = RunStore(app.state.state_root / "state")
    app.state.task_runner = run_task
    package_root = Path(__file__).parent
    templates = Jinja2Templates(directory=str(package_root / "templates"))
    app.mount("/static", StaticFiles(directory=str(package_root / "static")), name="static")

    def get_run(run_id: str) -> dict[str, Any]:
        try:
            return app.state.store.get(run_id)
        except (KeyError, ValueError) as error:
            raise HTTPException(status_code=404, detail="运行记录不存在") from error

    async def execute_task(task: TaskSpec, run_id: str) -> None:
        try:
            if inspect.iscoroutinefunction(run_task):
                await run_task(task, run_id)
            else:
                result = await run_in_threadpool(run_task, task, run_id)
                if inspect.isawaitable(result):
                    await result
        except Exception:
            # Persist a useful public failure without leaking credentials from
            # provider exceptions. The injected runner may record its own detail.
            current = app.state.store.get(run_id)
            if current["status"] not in RunStore.TERMINAL:
                state = {**current["state"], "error": "任务执行失败，请检查本机运行日志"}
                app.state.store.event(run_id, "runner_failed", {"reason": "task_runner_exception"})
                app.state.store.update(run_id, "failed", state)

    @app.get("/", include_in_schema=False)
    def workbench(request: Request):
        return templates.TemplateResponse(request=request, name="index.html", context={})

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "execution_mode": "external_runner" if run_task is not None else "offline",
            "demo": {"kind": "trusted_synthetic_popularity_fixture", "uses_llm": False},
            "live": {
                "runner_configured": run_task is not None,
                "model_backend_configured": None if run_task is not None else False,
                "worker_configured": None if run_task is not None else False,
            },
            "message": (
                "已接入外部任务执行器；其模型与 Worker 由调用方配置。"
                if run_task is not None else
                "当前为离线工作台。尚未配置模型与 Worker，提交任务会等待配置。"
            ),
        }

    @app.get("/api/runs")
    def list_runs(limit: int = Query(default=50, ge=1, le=200)):
        return {"runs": app.state.store.list(limit)}

    @app.get("/api/runs/{run_id}")
    def read_run(run_id: str):
        return get_run(run_id)

    @app.get("/api/runs/{run_id}/report")
    def read_report(run_id: str):
        run = get_run(run_id)
        return JSONResponse(status_code=200 if run["report"] is not None else 202, content={
            "run_id": run_id, "status": run["status"], "report": run["report"],
        })

    @app.get("/api/runs/{run_id}/events")
    async def events(request: Request, run_id: str, after: int = Query(default=0, ge=0)):
        get_run(run_id)
        last_id = request.headers.get("last-event-id", "")
        if last_id:
            try:
                after = max(after, int(last_id))
            except ValueError as error:
                raise HTTPException(status_code=400, detail="Last-Event-ID 必须是事件序号") from error

        async def stream():
            cursor = after
            idle_polls = 0
            while not await request.is_disconnected():
                batch = await run_in_threadpool(app.state.store.events, run_id, cursor)
                for event in batch:
                    if await request.is_disconnected():
                        return
                    cursor = event["seq"]
                    payload = json.dumps(event, ensure_ascii=False, allow_nan=False)
                    yield f"id: {cursor}\nevent: trace\ndata: {payload}\n\n"
                current = await run_in_threadpool(app.state.store.get, run_id)
                if current["status"] in RunStore.TERMINAL:
                    payload = json.dumps({"run_id": run_id, "status": current["status"], "last_seq": cursor})
                    yield f"event: end\ndata: {payload}\n\n"
                    return
                idle_polls = 0 if batch else idle_polls + 1
                if idle_polls >= 60:
                    yield ": heartbeat\n\n"
                    idle_polls = 0
                await asyncio.sleep(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
        })

    @app.post("/api/tasks", status_code=202)
    async def create_task(request: Request, background_tasks: BackgroundTasks):
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("任务必须是 JSON 对象")
            task = TaskSpec.from_dict(payload)
        except (ValueError, TypeError, OverflowError) as error:
            raise HTTPException(status_code=400, detail=f"无效任务：{error}") from error
        run_id = app.state.store.create(task)
        if run_task is None:
            state = {
                "configuration_required": ["model_backend", "worker"],
                "message": "任务已保存；配置模型与隔离 Worker 后才能执行真实实验。",
            }
            app.state.store.update(run_id, "waiting_input", state)
            app.state.store.event(run_id, "configuration_required", state)
        else:
            background_tasks.add_task(execute_task, task, run_id)
        return app.state.store.get(run_id)

    @app.post("/api/demo", status_code=201)
    def demo():
        try:
            return _run_demo(app.state.state_root)
        except ImportError as error:
            raise HTTPException(status_code=503, detail="合成演示模块尚不可用") from error

    return app
