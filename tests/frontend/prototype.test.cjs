"use strict";
// Dependency-free DOM contract tests, not a browser or visual acceptance test.
const assert = require("node:assert/strict");
const {readFileSync} = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {test} = require("node:test");
const root = path.join(__dirname,"../../reclab_agent/web/static");
const html = readFileSync(path.join(root,"prototype.html"),"utf8");
const script = readFileSync(path.join(root,"workbench-prototype.js"),"utf8");

class Element {
  constructor(tag) { this.tagName = tag.toUpperCase(); this.children=[]; this.attributes={}; this.dataset={}; this.value=""; this.hidden=false; this.style={}; this.open=false; this.listeners={}; this._text=""; }
  set textContent(value) { this._text=String(value);this.children=[]; }
  get textContent() { return this._text+this.children.map(c=>c.textContent).join(""); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children=[...nodes];this._text=""; }
  setAttribute(name,value) { this.attributes[name]=String(value);if(name==="id")this.id=value;if(name.startsWith("data-"))this.dataset[name.slice(5)]=value; }
  addEventListener(name,fn) { this.listeners[name]=fn; }
  focus() { this.focused=true; }
  showModal() { this.open=true; }
  close() { this.open=false; }
  get classList() { return {add:()=>{},remove:()=>{},toggle:()=>true}; }
}
function boot() {
  const nodes=[], ids=new Map();
  for(const match of html.matchAll(/<([a-z][\w-]*)([^>]*)>/gi)) {
    const node=new Element(match[1]);
    for(const attr of match[2].matchAll(/([\w-]+)="([^"]*)"/g))node.setAttribute(attr[1],attr[2]);
    if(node.attributes.id)ids.set(node.attributes.id,node);
    nodes.push(node);
  }
  ["status-filter","event-role","event-type"].forEach(id=>ids.get(id).value="all");
  const document={getElementById:id=>ids.get(id),createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag),
    createTextNode:text=>{const n=new Element("text");n.textContent=text;return n;},
    querySelectorAll:selector=>selector==='[role="tabpanel"]'?nodes.filter(n=>n.attributes.role==="tabpanel"):
      nodes.filter(n=>n.attributes[selector.slice(1,-1)]!==undefined),addEventListener:()=>{},activeElement:new Element("body")};
  const context=vm.createContext({document,console,setTimeout:()=>0,clearTimeout:()=>{},navigator:{clipboard:{writeText:async()=>{}}},FormData:class{constructor(form){return Object.entries(form.data);}}});
  vm.runInContext(script,context);
  return {ids,run:code=>vm.runInContext(code,context)};
}
function descendants(node) { return [node,...node.children.flatMap(descendants)]; }

test("prototype assets are local and make no service calls",()=>{
  assert.doesNotMatch(html,/(?:src|href)="https?:/);
  assert.doesNotMatch(script,/\bfetch\s*\(|new\s+EventSource\s*\(|\b(?:localStorage|sessionStorage)\b/);
  assert.match(html,/示例数据/);
});
test("initial view displays only two qualifying milestones, not the highest raw score",()=>{
  const {ids,run}=boot();
  assert.equal(ids.get("milestone-count").textContent,"2 次突破");
  assert.match(ids.get("metric-cards").textContent,/0\.08530/);
  assert.match(ids.get("milestones").textContent,/调整学习率/);
  assert.doesNotMatch(ids.get("milestones").textContent,/增加注意力头数/);
  assert.match(ids.get("status-track").textContent,/未评分 \/ 执行失败/);
  assert.equal(run("state.selected.entries.length"),7);
});
test("all scored comparable candidates appear, and failed experiment has no zero-valued point",()=>{
  const {ids}=boot(), chart=descendants(ids.get("metric-chart"));
  const points=chart.filter(n=>n.attributes.role==="button");
  assert.equal(points.length,6);
  assert(points.some(n=>n.attributes["aria-label"].includes("0.08710")));
  assert(!points.some(n=>n.attributes["aria-label"].includes("扩大隐藏维度")));
  assert(!points.some(n=>n.attributes["aria-label"].includes("0.00000")));
});
test("chart points and milestones open evidence details",()=>{
  const {ids}=boot();
  const point=descendants(ids.get("metric-chart")).find(n=>n.attributes.role==="button");
  point.listeners.click();
  assert(ids.get("experiment-dialog").open);
  assert.match(ids.get("detail-body").textContent,/不代表真实指标/);
  ids.get("experiment-dialog").close();
  ids.get("milestones").children[1].onclick();
  assert.equal(ids.get("detail-title").textContent,"校正序列截断");
});
test("switching to decline, failure and insufficient evidence clears milestones",()=>{
  const {ids,run}=boot();
  for(const index of [1,2,3]) {
    run(`state.selected=fixtures[${index}];render()`);
    assert.equal(ids.get("milestone-count").textContent,"0 次突破");
    assert.match(ids.get("milestones").textContent,/尚无合格候选超过基线/);
    assert.match(ids.get("metric-cards").textContent,/0\.08045/);
  }
  assert.match(ids.get("audit-status").textContent,/证据不足/);
});
test("search, empty results and filters work",()=>{
  const {ids}=boot();ids.get("search").value="GRURec";ids.get("search").oninput();
  assert.equal(ids.get("run-list").children.length,1);
  ids.get("search").value="no-such-task";ids.get("search").oninput();
  assert.match(ids.get("run-list").textContent,/没有匹配/);
  ids.get("search").value="";ids.get("status-filter").value="failed";ids.get("status-filter").onchange();
  assert.equal(ids.get("run-list").children.length,1);
});
test("tabs support keyboard navigation and preserve selected panel",()=>{
  const {ids,run}=boot();
  ids.get("tab-overview").onkeydown({key:"ArrowRight",preventDefault(){}});
  assert.equal(run("state.tab"),"changes");
  assert.equal(ids.get("tab-changes").attributes["aria-selected"],"true");
  assert(ids.get("panel-overview").hidden);
  assert(!ids.get("panel-changes").hidden);
});
test("event filters isolate errors and audit role",()=>{
  const {ids}=boot();ids.get("event-type").value="error";ids.get("event-type").onchange();
  assert.equal(ids.get("event-list").children.length,1);
  assert.match(ids.get("event-list").textContent,/smoke_test/);
  ids.get("event-type").value="all";ids.get("event-role").value="auditor";ids.get("event-role").onchange();
  assert.equal(ids.get("event-list").children.length,1);
  assert.match(ids.get("event-list").textContent,/submit_audit/);
});
test("new in-memory task clears stale chart and evidence; cancellation is local",()=>{
  const {ids}=boot();const form=ids.get("task-form");form.data={task_id:"fixture-new",question:"<img src=x onerror=alert(1)>",dataset_id:"beauty",baseline_id:"baseline"};form.reset=()=>{};
  form.onsubmit({preventDefault(){},target:form});
  assert.equal(ids.get("run-title").textContent,form.data.question);
  assert.match(ids.get("metric-chart").textContent,/暂无已核验/);
  assert.equal(ids.get("status-track").children.length,0);
  assert.match(ids.get("hypothesis-content").textContent,/尚无研究假设/);
  assert(!ids.get("cancel-task").hidden);
  ids.get("cancel-task").onclick();
  assert.match(ids.get("run-status").textContent,/已取消/);
  assert(ids.get("cancel-task").hidden);
  assert.doesNotMatch(script,/\.innerHTML\s*=|insertAdjacentHTML|document\.write/);
});
test("zero baseline and dense milestone labels render without non-finite SVG values",()=>{
  const {ids,run}=boot();
  assert.equal(run("relative(null)"),"无法计算");
  run(`renderChart([{id:'b',name:'基线',value:0,best:0,verified:true,comparable:true,eligible:true},...Array.from({length:12},(_,i)=>({id:'m'+i,name:'改进'+i,value:(i+1)*.001,best:(i+1)*.001,verified:true,comparable:true,eligible:true,milestone:true,relative:null,decision:'刷新最佳'}))])`);
  const nodes=descendants(ids.get("metric-chart"));
  const labels=nodes.filter(n=>n.attributes.class==="point-label");
  assert.equal(labels.length,12);
  assert.equal(new Set(labels.map(n=>n.attributes.y)).size,12);
  assert(nodes.every(n=>Object.values(n.attributes).every(v=>!String(v).match(/NaN|Infinity/))));
});
