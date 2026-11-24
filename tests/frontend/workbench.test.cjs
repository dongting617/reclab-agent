"use strict";
const assert=require("node:assert/strict"),{readFileSync}=require("node:fs"),path=require("node:path"),vm=require("node:vm"),{test}=require("node:test");
const web=path.join(__dirname,"../../reclab_agent/web");
const html=readFileSync(path.join(web,"templates/index.html"),"utf8");
const app=readFileSync(path.join(web,"static/app.js"),"utf8");
const chart=readFileSync(path.join(web,"static/workbench-charts.js"),"utf8");
class Element {
  constructor(tag){this.tagName=tag.toUpperCase();this.children=[];this.attributes={};this.dataset={};this.value="";this.hidden=false;this.style={};this.open=false;this.listeners={};this._text="";this.scrollTop=0;this.scrollHeight=500;this.clientHeight=100;}
  set textContent(v){this._text=String(v);this.children=[];} get textContent(){return this._text+this.children.map(c=>c.textContent).join("");}
  append(...nodes){this.children.push(...nodes);}replaceChildren(...nodes){this.children=[...nodes];this._text="";}
  setAttribute(key,value){this.attributes[key]=String(value);if(key==="id")this.id=value;if(key.startsWith("data-"))this.dataset[key.slice(5)]=value;}
  addEventListener(key,fn){this.listeners[key]=fn;}focus(){this.focused=true;}showModal(){this.open=true;}close(){this.open=false;}
  get classList(){return{add(){},remove(){},toggle(){return true;}};}
}
const flush=()=>new Promise(resolve=>setImmediate(resolve));
function fixture(id="a"){
  const baseline={id:"基线",experiment_id:"baseline",record:{status:"completed",experiment_id:"baseline",score:{source:"score.json"}},metrics:{"ndcg@20":.08,"recall@20":.15},verified:true,changes:[],warnings:[]};
  const entry={id:"candidate",experiment_id:"candidate",record:{status:"completed",experiment_id:"candidate",score:{source:"score.json"}},metrics:{"ndcg@20":.10,"recall@20":.16},verified:true,comparable:true,comparison_eligible:true,milestone:true,relative:{"ndcg@20":25},deltas:{"ndcg@20":.02},changes:[{path:"src/model.py",diff:"- WIDTH=64\n+ WIDTH=128"}],warnings:[],decision:"刷新最佳"};
  const report={stages:[{stage:"analysis"}],budget_usage:{model_requests:4},experiments:[]};
  const run={run_id:id,status:"completed",origin:"study_session",task:{question:"Research "+id},state:{},report,updated:1};
  const view={primary:"ndcg@20",metrics:["ndcg@20","recall@20"],baseline,entries:[entry],best:entry,improved:true,summary:"Measured improvement "+id,progress:{groups:[{key:"single",name:"ReAct",baseline:{id:"基线",experiment_id:"baseline",value:.08},points:[{id:"candidate",experiment_id:"candidate",order:1,value:.1,best:.1,milestone:true,qualified:true,relative:25,delta:.02,previous_delta:.02,decision:"刷新最佳"}]}]}};
  return{run,view};
}
const response=value=>({ok:true,status:200,json:async()=>value});
function boot(customFetch){
  const nodes=[],ids=new Map();
  for(const match of html.matchAll(/<([a-z][\w-]*)([^>]*)>/gi)){const node=new Element(match[1]);for(const attr of match[2].matchAll(/([\w-]+)="([^"]*)"/g))node.setAttribute(attr[1],attr[2]);node.hidden=/\bhidden(?:\s|$)/.test(match[2]);if(node.id)ids.set(node.id,node);nodes.push(node);}
  ["status-filter","event-role","event-type"].forEach(id=>ids.get(id).value="all");
  const document={getElementById:id=>{assert(ids.has(id),"Unknown DOM ID "+id);return ids.get(id);},createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag),createTextNode:text=>{const n=new Element("text");n.textContent=text;return n;},querySelectorAll:selector=>selector==='[role="tabpanel"]'?nodes.filter(n=>n.attributes.role==="tabpanel"):nodes.filter(n=>n.attributes[selector.slice(1,-1)]!==undefined),addEventListener(){},activeElement:new Element("body")};
  const requests=[],streams=[],timers=new Map();let nextTimer=0,keys=0;
  class Stream{constructor(url){this.url=url;this.listeners={};streams.push(this);}addEventListener(name,fn){this.listeners[name]=fn;}close(){this.closed=true;}emit(name,data){this.listeners[name]?.({data:JSON.stringify(data)});}}
  const fetch=async(url,options={})=>{requests.push({url,options});if(url==="/health")return response({execution_mode:"offline"});if(url==="/api/runs")return response({runs:[fixture().run]});if(customFetch)return customFetch(url,options);const id=url.split("/")[3],f=fixture(id);return response(url.endsWith("/report")?{report:f.run.report,view:f.view}:f.run);};
  const context=vm.createContext({document,console,fetch,AbortController,EventSource:Stream,crypto:{randomUUID:()=>"key-"+(++keys)},navigator:{clipboard:{writeText:async()=>{}}},setTimeout:fn=>{timers.set(++nextTimer,fn);return nextTimer;},clearTimeout:id=>timers.delete(id),setInterval:()=>0,FormData:class{constructor(form){return Object.entries(form.data);}}});
  vm.runInContext(chart,context);vm.runInContext(app,context);
  return{ids,requests,streams,timers,run:code=>vm.runInContext(code,context)};
}
function descendants(node){return[node,...node.children.flatMap(descendants)];}

test("production template loads live controller, not fixtures; default startup is read-only",async()=>{
  const b=boot();await flush();assert(!html.includes("workbench-prototype.js"));assert(!html.includes("示例数据"));
  assert(b.requests.every(r=>!r.options.method||r.options.method==="GET"));assert.match(b.ids.get("run-list").textContent,/Research a/);
  assert.doesNotMatch(app,/\.innerHTML\s*=|insertAdjacentHTML|document\.write/);
});
test("live report drives chart, details, diff and report summary consistently",async()=>{
  const b=boot();await flush();await b.run('selectRun("a")');
  assert.match(b.ids.get("metric-cards").textContent,/25\.00%/);assert.equal(b.ids.get("milestone-count").textContent,"1 次突破");
  const point=descendants(b.ids.get("metric-chart")).filter(n=>n.attributes.role==="button")[1];point.listeners.click();
  assert(b.ids.get("experiment-dialog").open);assert.match(b.ids.get("detail-body").textContent,/WIDTH=128/);
  assert.match(b.ids.get("report-content").textContent,/Measured improvement a/);
  assert.match(b.ids.get("audit-status").textContent,/未记录/);
  const stages=b.ids.get("stages").children;assert.match(stages[0].textContent,/未记录/);assert.match(stages[3].textContent,/已有阶段记录/);
});
test("A→B→A race ignores stale A/B responses even when abort is ignored",async()=>{
  const pending=[];const b=boot(url=>new Promise(resolve=>pending.push({url,resolve})));await flush();
  const oldA=b.run('selectRun("a")'),oldB=b.run('selectRun("b")'),newA=b.run('selectRun("a")');
  assert.equal(pending.length,6);
  const finish=(start,label)=>{for(const req of pending.slice(start,start+2)){const f=fixture(label);req.resolve(response(req.url.endsWith("/report")?{report:f.run.report,view:f.view}:f.run));}};
  finish(4,"a");await newA;finish(2,"b");await oldB;finish(0,"stale-a");await oldA;
  assert.equal(b.ids.get("run-title").textContent,"Research a");assert.equal(b.streams.length,1);assert.match(b.streams[0].url,/\/a\/events$/);
});
test("SSE deduplicates, ignores old session callbacks, and preserves scroll and expanded events",async()=>{
  const b=boot();await flush();await b.run('selectRun("a")');const old=b.streams[0];
  old.emit("trace",{seq:1,event:"tool_call",payload:{event:"tool_call",tool:"read_source"}});old.emit("trace",{seq:1,event:"tool_call"});
  assert.equal(b.run("state.events.length"),1);const list=b.ids.get("event-list");list.scrollTop=10;list.children[0].open=true;
  old.emit("trace",{seq:2,event:"model_turn",payload:{event:"model_turn",phase:"audit_judge"}});
  assert.equal(list.scrollTop,10);assert(list.children[0].open);
  b.ids.get("event-role").value="auditor";b.ids.get("event-role").onchange();assert.equal(list.children.length,1);assert.match(list.textContent,/审计判断/);
  await b.run('selectRun("b")');old.emit("trace",{seq:3,event:"stale"});assert.equal(b.run("state.events.length"),0);assert(old.closed);
});
test("same payload retries reuse idempotency key; changed payload gets a new key",async()=>{
  const b=boot(async()=>({ok:false,status:503,json:async()=>({detail:"controlled outage"})}));await flush();
  const form=b.ids.get("task-form");form.data={task_id:"task",question:"question",dataset_id:"data",baseline_id:"base"};form.reset=()=>{};
  await form.onsubmit({preventDefault(){},target:form});await form.onsubmit({preventDefault(){},target:form});
  form.data.question="changed";await form.onsubmit({preventDefault(){},target:form});const posts=b.requests.filter(r=>r.url==="/api/tasks");
  assert.equal(posts.length,3);assert.equal(posts[0].options.headers["Idempotency-Key"],posts[1].options.headers["Idempotency-Key"]);assert.notEqual(posts[1].options.headers["Idempotency-Key"],posts[2].options.headers["Idempotency-Key"]);
  assert.match(b.ids.get("task-error").textContent,/controlled outage/);
});
test("refresh failure retains prior content and overlapping refreshes coalesce",async()=>{
  let failing=false;const b=boot(async url=>{if(failing)throw new Error("offline");const f=fixture("a");return response(url.endsWith("/report")?{report:f.run.report,view:f.view}:f.run);});await flush();await b.run('selectRun("a")');
  const before=b.ids.get("metric-cards").textContent;failing=true;const count=b.requests.length;
  await Promise.all([b.run("refreshRun()"),b.run("refreshRun()")]);
  assert.equal(b.requests.length-count,2);assert.equal(b.ids.get("metric-cards").textContent,before);assert.match(b.ids.get("notice").textContent,/保留上次成功内容/);
});
test("logs limit defaults to 200 and can load earlier events",async()=>{
  const b=boot();await flush();await b.run('selectRun("a");state.events=Array.from({length:250},(_,i)=>({seq:i,event:"trace"}));drawEvents()');
  const list=b.ids.get("event-list");assert.equal(list.children.length,201);assert.match(list.children[0].textContent,/加载更早/);list.children[0].onclick();assert.equal(list.children.length,250);
});
