"use strict";
const $ = (id) => document.getElementById(id),
  pretty = (x) => JSON.stringify(x, null, 2);
const state = {
  id: null,
  stream: null,
  seen: new Set(),
  events: [],
  requestKey: null,
};
const labels = {
  running: "运行中",
  completed: "完成",
  failed: "失败",
  partial: "部分完成",
  incomplete: "证据不足",
  waiting_input: "等待配置",
  queued: "等待执行",
  cancelled: "已取消",
  budget_exceeded: "预算耗尽",
};
async function api(path, options = {}) {
  const r = await fetch(path, options),
    v = await r.json();
  if (!r.ok) throw new Error(v.detail || "请求失败");
  return v;
}
function error(e) {
  $("notice").textContent = e.message || String(e);
}
function card(name, value) {
  const el = document.createElement("div");
  el.className = "card";
  const small = document.createElement("small"),
    strong = document.createElement("strong");
  small.textContent = name;
  strong.textContent = value;
  el.append(small, strong);
  return el;
}
async function list() {
  const data = await api("/api/runs");
  $("runs").replaceChildren();
  for (const run of data.runs) {
    const b = document.createElement("button");
    b.className = "run-item" + (run.run_id === state.id ? " selected" : "");
    b.textContent = run.task.question || run.task.task_id;
    const sub = document.createElement("small");
    sub.textContent = (labels[run.status] || run.status) + " · " + run.run_id;
    b.append(sub);
    b.onclick = () => select(run.run_id).catch(error);
    $("runs").append(b);
  }
  if (!data.runs.length) $("runs").textContent = "还没有保存的任务";
}
async function render() {
  if (!state.id) return;
  const run = await api("/api/runs/" + encodeURIComponent(state.id)),
    r = run.report || {},
    usage = r.budget_usage || run.state.budget_usage || {};
  $("title").textContent = run.task.question || run.task.task_id;
  $("notice").textContent =
    (labels[run.status] || run.status) + (r.error ? " · " + r.error : "");
  $("summary").replaceChildren(
    card("状态", labels[run.status] || run.status),
    card("实验次数", usage.experiment_attempts ?? "—"),
    card("模型请求", usage.model_requests ?? "—"),
  );
  $("state").textContent = pretty(run.state);
  $("diff").textContent = pretty({
    hypotheses: r.hypotheses || [],
    changes: r.changes || r.code_changes || [],
    editable_files: r.context?.editable_files || [],
    snapshots: (r.experiments || []).map((x) => ({
      id: x.record?.experiment_id,
      source_hash: x.record?.source_hash,
    })),
  });
  const view = run.report
    ? (await api("/api/runs/" + encodeURIComponent(state.id) + "/report")).view
    : null;
  $("report-body").textContent = view
    ? view.summary + "\n\n" + pretty(view)
    : "报告尚未生成";
  if (view?.entries?.length)
    $("diff").textContent = pretty({
      hypotheses: r.hypotheses || [],
      changes: view.entries.flatMap((x) => x.changes || []),
    });
  $("experiment-rows").replaceChildren();
  for (const x of r.experiments || []) {
    const row = document.createElement("tr");
    for (const text of [
      x.record?.candidate_id || x.candidate_id,
      labels[x.record?.status] || x.record?.status,
      pretty(x.metrics || {}),
    ]) {
      const cell = document.createElement("td");
      cell.textContent = text || "—";
      row.append(cell);
    }
    $("experiment-rows").append(row);
  }
  $("cancel").disabled =
    run.origin === "study_session" ||
    !["queued", "waiting_input"].includes(run.status);
}
async function select(id) {
  state.stream?.close();
  state.id = id;
  state.events = [];
  state.seen.clear();
  $("trace-body").textContent = "暂无事件";
  await render();
  await list();
  const stream = new EventSource(
    "/api/runs/" + encodeURIComponent(id) + "/events",
  );
  state.stream = stream;
  stream.addEventListener("trace", (e) => {
    const x = JSON.parse(e.data);
    if (state.seen.has(x.seq)) return;
    state.seen.add(x.seq);
    state.events.push(x);
    $("trace-body").textContent = state.events.map(pretty).join("\n");
    render().catch(error);
  });
  stream.addEventListener("end", () => {
    stream.close();
    render().catch(error);
    list().catch(error);
  });
}
$("refresh").onclick = () => Promise.all([list(), render()]).catch(error);
$("demo").onclick = async () => {
  try {
    $("demo").disabled = true;
    const run = await api("/api/demo", { method: "POST" });
    await select(run.run_id);
  } catch (e) {
    error(e);
  } finally {
    $("demo").disabled = false;
  }
};
$("cancel").onclick = async () => {
  try {
    await api("/api/runs/" + encodeURIComponent(state.id) + "/cancel", {
      method: "POST",
    });
    await select(state.id);
  } catch (e) {
    error(e);
  }
};
$("task-form").onsubmit = async (e) => {
  e.preventDefault();
  state.requestKey ??= crypto.randomUUID();
  const data = Object.fromEntries(new FormData(e.target));
  try {
    const run = await api("/api/tasks", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": state.requestKey,
      },
      body: JSON.stringify(data),
    });
    state.requestKey = null;
    await select(run.run_id);
  } catch (e) {
    error(e);
  }
};
document.querySelectorAll("[data-panel]").forEach(
  (b) =>
    (b.onclick = () => {
      document
        .querySelectorAll(".panel")
        .forEach((p) => (p.hidden = p.id !== b.dataset.panel));
      document
        .querySelectorAll("[data-panel]")
        .forEach((x) => x.classList.toggle("active", x === b));
    }),
);
api("/health")
  .then((h) => {
    $("connection").textContent =
      h.execution_mode === "offline" ? "本地只读与演示" : "已连接任务执行器";
  })
  .catch(error);
list().catch(error);
setInterval(() => {
  if (state.id) render().catch(error);
}, 3000);
