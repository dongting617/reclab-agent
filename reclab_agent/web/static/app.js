"use strict";
const $ = id => document.getElementById(id);
const labels = {running:"运行中",completed:"已结束",failed:"执行失败",partial:"部分完成",incomplete:"证据不足",waiting_input:"等待配置",queued:"等待执行",cancelled:"已取消",timed_out:"超时"};
const state = {id:null,runs:[],run:null,view:null,tab:"overview",group:null,generation:0,controller:null,stream:null,
  events:[],seen:new Set(),eventLimit:200,requestKey:null,requestPayload:null,refreshing:null,refreshQueued:false,eventTimer:null};
function el(tag, className="", text) {
  const node=document.createElement(tag); if(className)node.className=className; if(text!==undefined)node.textContent=text; return node;
}
function badge(text,tone="neutral") {
  const node=el("span","tag "+tone);node.append(el("i","tag-dot"),document.createTextNode(text));return node;
}
const number=v=>typeof v==="number"&&Number.isFinite(v);
const format=v=>number(v)?String(Number(v.toPrecision(6))):"—";
const signed=v=>number(v)?(v>0?"+":"")+v.toPrecision(4):"—";
const relative=v=>number(v)?(v>0?"+":"")+v.toFixed(2)+"%":"无法计算";
const tone=status=>["failed","timed_out","rejected"].includes(status)?"negative":["incomplete","waiting_input","partial","insufficient_evidence"].includes(status)?"warning":status==="completed"||status==="approved"?"positive":"neutral";
let toastTimer;
function toast(message) {$("toast").textContent=message;$("toast").hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>{$("toast").hidden=true;},4500);}
async function api(path,options={}) {
  const response=await fetch(path,options),value=await response.json();
  if(!response.ok)throw new Error(typeof value.detail==="string"?value.detail:"请求失败（"+response.status+"）");return value;
}
function reportError(error) {
  if(error.name==="AbortError")return;$("notice").hidden=false;$("notice").textContent="请求失败，保留上次成功内容。"+error.message;
}
function showTab(name,focus=false) {
  state.tab=name;
  document.querySelectorAll("[data-tab]").forEach(button=>{const selected=button.dataset.tab===name;button.setAttribute("aria-selected",String(selected));button.tabIndex=selected?0:-1;if(selected&&focus)button.focus();});
  document.querySelectorAll('[role="tabpanel"]').forEach(panel=>{panel.hidden=panel.id!=="panel-"+name;});
}
function rawDetails(title,value) {const node=el("details","raw-details");node.append(el("summary","",title),el("pre","",JSON.stringify(value,null,2)));return node;}
function pathRow(path) {
  const node=el("div","evidence-path"),button=el("button","text-action","复制路径");
  button.onclick=async()=>{try{await navigator.clipboard.writeText(path);toast("已复制证据路径。");}catch{toast("无法访问剪贴板，请选中路径手动复制。");}};
  node.append(el("span","",path),button);return node;
}
function diffBlock(change) {
  const node=el("details","diff-section");node.open=true;node.append(el("summary","",change.path||"未记录文件"));
  const pre=el("pre","code-diff");String(change.diff||"未记录实际差异").split("\n").forEach(line=>pre.append(el("span",line.startsWith("+")&&!line.startsWith("+++")?"diff-add":line.startsWith("-")&&!line.startsWith("---")?"diff-remove":"diff-line",line||" ")));
  node.append(pre,el("p","fine-print",(change.evidence||"冻结源码差异")+(change.truncated?" · 内容已截断":"")));return node;
}
function currentGroup(){return state.view?.progress?.groups?.find(group=>group.key===state.group)||state.view?.progress?.groups?.[0];}
function allEntries(){return state.view?[state.view.baseline,...(state.view.entries||[])].filter(Boolean):[];}
function showDetail(id) {
  const entry=allEntries().find(item=>item.experiment_id===id);if(!entry)return;
  $("detail-title").textContent=entry.display_id||entry.id;const body=$("detail-body");body.replaceChildren();const grid=el("div","detail-grid");
  [["实验 ID",entry.experiment_id],[state.view.primary,format(entry.metrics?.[state.view.primary])],["评分核验",entry.verified?"已核验":"未核验"],["接受条件",entry.decision||"基线"]].forEach(([name,value])=>{const cell=el("div");cell.append(el("small","",name),el("strong","",value));grid.append(cell);});body.append(grid);
  [entry.failure,...(entry.warnings||[])].filter(Boolean).forEach(text=>body.append(el("p","callout",text)));
  body.append(el("h3","","实际代码差异"));if(entry.changes?.length)entry.changes.forEach(change=>body.append(diffBlock(change)));else body.append(el("p","content-prose","没有可用的冻结源码差异证据。"));
  if(entry.record?.score?.source)body.append(pathRow(entry.record.score.source));
  body.append(rawDetails("独立评分来源与身份",{metrics:entry.metrics,verified:entry.verified,score:entry.record?.score,dataset_id:entry.record?.dataset_id,seed:entry.record?.seed,protocol_hash:entry.record?.protocol_hash,source_hash:entry.record?.source_hash,evaluation_hash:entry.evaluation_hash}),rawDetails("过程诊断",entry.process_metrics||{message:"未记录过程诊断"}),rawDetails("独立审计",entry.audit||{message:"未记录审计"}));
  $("experiment-dialog").showModal();
}
function updated(timestamp,full=true) {if(!number(timestamp))return "未记录时间";const date=new Date(timestamp*1000);return full?date.toLocaleString("zh-CN",{hour12:false}):date.toLocaleTimeString("zh-CN",{hour:"2-digit",minute:"2-digit",hour12:false});}
function drawList() {
  const query=$("search").value.toLowerCase().trim(),filter=$("status-filter").value;$("run-list").replaceChildren();
  const rows=state.runs.filter(run=>(filter==="all"||filter===run.status)&&[run.task.question,run.task.task_id,run.task.dataset_id,run.run_id].join(" ").toLowerCase().includes(query));
  rows.forEach(run=>{
    const button=el("button","run-item"+(run.run_id===state.id?" selected":""));button.setAttribute("aria-pressed",String(run.run_id===state.id));
    const demo=run.report?.kind==="evidence_reference"||run.state?.kind==="evidence_reference";
    button.append(el("span","run-title",run.task.question||run.task.task_id),el("span","run-description",demo?"本地证据演示 · 无模型调用":run.task.dataset_id||run.report?.context?.dataset_id||"研究会话"));
    const bottom=el("span","run-bottom");bottom.append(badge(labels[run.status]||run.status,tone(run.status)),el("span","",updated(run.updated,false)));button.append(bottom);button.onclick=()=>selectRun(run.run_id).catch(reportError);$("run-list").append(button);
  });
  if(!rows.length)$("run-list").append(el("p","empty-state",state.runs.length?"没有匹配的研究会话":"尚无研究会话。可登记任务或运行本地证据演示。"));
}
let listing=null;
async function listRuns(){
  if(listing)return listing;
  listing=(async()=>{const data=await api("/api/runs");state.runs=data.runs;drawList();})();
  try{return await listing;}finally{listing=null;}
}
function drawStages() {
  const records=state.run.state?.stages||state.run.report?.stages||[],observed=new Set(records.map(item=>item.stage)),latest=records.at(-1)?.stage;$("stages").replaceChildren();
  [["background","背景调研"],["hypothesis","研究假设"],["experiment","实验执行"],["analysis","结果分析"]].forEach(([key,name],index)=>{
    const seen=observed.has(key),failed=seen&&key===latest&&state.run.status==="failed",item=el("div","stage"+(failed?" failed":seen?"":" pending"));item.append(el("span","stage-icon",failed?"!":seen?"✓":String(index+1)));
    const label=el("div","stage-label",name);label.append(el("small","",failed?"此阶段中断":seen?"已有阶段记录":"未记录"));item.append(label);$("stages").append(item);
  });
}
function metricCard(name,value,note,positive=false,suffix="") {
  const card=el("div","metric-card");card.append(el("div","metric-name",name));const strong=el("div","metric-value",value);if(suffix)strong.append(el("small","",suffix));card.append(strong,el("div","metric-note"+(positive?" positive-text":""),note));return card;
}
function drawProgress() {
  const view=state.view,group=currentGroup();$("chart-metric").textContent=view?.primary||"未配置主指标";$("group-picker").replaceChildren();
  (view?.progress?.groups||[]).forEach(item=>{const option=el("option","",item.name);option.value=item.key;$("group-picker").append(option);});$("group-picker").value=group?.key||"";$("group-picker-label").hidden=(view?.progress?.groups?.length||0)<2;
  RecLabCharts.render($("metric-chart"),group,view?.primary||"主指标",showDetail);$("status-track").replaceChildren(el("span","","实验记录"));
  (group?.points||[]).forEach(point=>{const entry=allEntries().find(item=>item.experiment_id===point.experiment_id),failed=entry?.record?.status==="failed";const button=el("button","status-pill "+(failed?"negative":!point.qualified?"warning":""),`${String(point.order).padStart(2,"0")} · ${failed?"未评分 / 执行失败":(point.value==null?"未绘图 / ":"")+point.decision}`);button.onclick=()=>showDetail(point.experiment_id);$("status-track").append(button);});
  const milestones=(group?.points||[]).filter(point=>point.milestone);$("milestone-count").textContent=milestones.length+" 次突破";$("milestones").replaceChildren();
  milestones.forEach(point=>{const button=el("button","milestone-item"),top=el("div","milestone-top"),entry=allEntries().find(item=>item.experiment_id===point.experiment_id);top.append(el("span","","实验 "+point.order),el("strong","",relative(point.relative)));
    button.append(top,el("h4","",point.id),el("p","",`${format(point.value)} · 较上次最佳 ${signed(point.previous_delta)}`),el("p","",`较基线 ${signed(point.delta)} · ${(entry?.changes||[]).map(change=>change.path).join("、")||"未记录改动摘要"}`));button.onclick=()=>showDetail(point.experiment_id);$("milestones").append(button);});
  if(!milestones.length)$("milestones").append(el("p","milestone-empty","尚无合格候选超过基线。未核验、不可比较和无评分记录不会形成提升标记。"));
}
function drawExperiments() {
  const view=state.view;$("experiment-rows").replaceChildren();$("experiment-head").replaceChildren();["候选 / 改动",...(view?.metrics||[]),"相对基线","评分 / 资格","结论"].forEach(name=>$("experiment-head").append(el("th","",name)));
  allEntries().forEach(entry=>{const base=entry===view.baseline;if(!entry.record||!Object.keys(entry.record).length)return;
    const row=el("tr",base?"baseline-row":entry.milestone?"row-milestone":""),name=el("td"),button=el("button","table-button",entry.display_id||entry.id);button.onclick=()=>showDetail(entry.experiment_id);name.append(button,el("small","",entry.record.status||"未记录"));row.append(name);
    (view.metrics||[]).forEach(metric=>row.append(el("td","",format(entry.metrics?.[metric]))));row.append(el("td",entry.deltas?.[view.primary]>0?"positive-text":entry.deltas?.[view.primary]<0?"negative-text":"",base?"基准":entry.metrics?relative(entry.relative?.[view.primary]):"—"));
    const verification=el("td");verification.append(badge(entry.metrics?entry.verified?"已核验":"未核验":"未评分",entry.verified?"positive":"neutral"),el("small","",base?"固定比较基准":entry.comparable?"比较条件一致":"不可比较或信息不足"));const result=el("td");result.append(badge(base?"基线":entry.milestone?"刷新最佳":entry.decision,base?"neutral":entry.milestone?"positive":entry.comparison_eligible?"neutral":"warning"));row.append(verification,result);$("experiment-rows").append(row);
  });$("experiment-empty").hidden=allEntries().some(entry=>Object.keys(entry.record||{}).length);
}
function panel(title,subtitle) {const node=el("section","surface"),heading=el("div","surface-heading"),text=el("div");text.append(el("h2","",title));if(subtitle)text.append(el("p","",subtitle));heading.append(text);node.append(heading);return node;}
function drawChanges() {
  const target=$("hypothesis-content");target.replaceChildren();const hypotheses=state.run.report?.hypotheses||[];
  if(!hypotheses.length)target.append(el("div","surface empty-state","尚无登记的研究假设"));
  hypotheses.forEach(hypothesis=>{const node=panel(hypothesis.candidate_id||"研究假设","宿主登记的假设与诊断要求"),body=el("div","content-prose");Object.entries(hypothesis).filter(([key])=>key!=="candidate_id").forEach(([key,value])=>{const names={hypothesis:"假设",expected_observation:"预期观察",risks:"风险",process_metrics:"过程诊断",causal_chain:"预期因果链",evidence_gaps:"证据缺口"};body.append(el("h3","",names[key]||key),el("p","",typeof value==="string"?value:JSON.stringify(value,null,2)));});node.append(body);target.append(node);});
  let count=0;(state.view?.entries||[]).forEach(entry=>{if(!entry.changes?.length)return;const node=panel(entry.display_id||entry.id,"冻结源码快照或控制器记录的实际差异");entry.changes.forEach(change=>node.append(diffBlock(change)));const button=el("button","text-action","查看实验与证据 →");button.onclick=()=>showDetail(entry.experiment_id);node.append(button);target.append(node);count++;});
  if(!count)target.append(el("div","surface empty-state","没有可用的实际源码差异；不会用模型描述代替。"));
}
function drawAuditAndIssues() {
  const entries=state.view?.entries||[],audits=entries.filter(entry=>entry.audit),verdicts=audits.map(entry=>entry.audit.audit?.verdict);
  const approved=entry=>entry.audit.status==="completed"&&entry.audit.audit_approved===true&&entry.audit.audit?.verdict==="approved";
  const auditText=!audits.length?"未记录":verdicts.includes("rejected")?"存在拒绝":verdicts.includes("insufficient_evidence")?"证据不足":audits.every(approved)?"已记录通过":"待核验";
  $("audit-status").replaceChildren(badge(auditText,!audits.length?"neutral":audits.every(approved)?"positive":"warning"));$("audit-content").replaceChildren();
  if(!audits.length)$("audit-content").append(el("p","content-prose","该会话没有可用的独立审计记录，不推断为通过。"));
  audits.forEach(entry=>{const row=el("div","audit-row"),body=el("div");body.append(el("strong","",entry.id),el("small","",entry.audit.audit?.summary||entry.audit.status||"审计状态未记录"));const button=el("button","text-action","查看证据与审计 →");button.onclick=()=>showDetail(entry.experiment_id);body.append(button);row.append(el("span","audit-symbol",approved(entry)?"✓":"!"),body);$("audit-content").append(row);});
  const issues=[...new Set([state.run.state?.message,state.run.state?.error,state.view?.error,state.run.state?.requires_reconciliation||state.run.report?.requires_reconciliation?"远端作业停止尚未确认，请先核对执行状态。":null,...entries.flatMap(entry=>[entry.failure,...(entry.warnings||[]),!entry.comparison_eligible&&entry.metrics?entry.id+"："+entry.decision:null])].filter(Boolean))];
  $("issue-count").textContent=issues.length+" 项";$("issues").replaceChildren();issues.forEach(text=>{const row=el("div","issue-item");row.append(el("span","issue-marker"),el("span","",text));$("issues").append(row);});if(!issues.length)$("issues").append(el("p","content-prose","暂无已记录的异常。单种子验证集结果仅适用于当前实验范围。"));
}
function drawReport() {
  const target=$("report-content");target.replaceChildren();if(!state.view){target.append(el("div","surface empty-state","报告尚未生成"));return;}
  const node=panel("研究结论","结论、图表与实验对比采用同一套证据资格规则。"),body=el("div","content-prose");body.append(el("p","",state.view.summary));(state.view.limitations||[]).forEach(text=>body.append(el("p","",text)));(state.view.audit_notes||[]).forEach(text=>body.append(el("p","",text)));node.append(body,rawDetails("查看报告视图 JSON",state.view));target.append(node);
  if(state.view.reference_results){const checks=panel("本地证据参考检查","这不是端到端研究或模型提升评测。");state.view.reference_results.forEach(row=>{const line=el("div","audit-row");line.append(badge(row.passed?"通过":"失败",row.passed?"positive":"negative"),el("span","",row.id));checks.append(line);});target.append(checks);}
}
function eventInfo(event){const payload=event.payload||event;return{payload,type:payload.event||event.event||"trace",role:payload.role||payload.phase||(payload.tool?"host":"unrecorded"),tool:payload.tool||""};}
function drawEvents() {
  const target=$("event-list"),scrollTop=target.scrollTop,atBottom=target.scrollHeight-target.scrollTop-target.clientHeight<50,role=$("event-role").value,type=$("event-type").value,query=$("event-search").value.toLowerCase();
  const opened=new Set([...target.children].filter(node=>node.open).map(node=>node.dataset.seq));
  const events=state.events.filter(event=>{const info=eventInfo(event),error=info.type.includes("error")||info.payload.result?.ok===false;const roleMatches=role==="all"||(role==="auditor"?String(info.role).startsWith("audit"):role==="researcher"?info.role==="researcher"||info.role==="research":info.role===role);return roleMatches&&(type==="all"||(type==="error"?error:type===info.type))&&JSON.stringify(event).toLowerCase().includes(query);});
  target.replaceChildren();if(events.length>state.eventLimit){const more=el("button","button secondary","加载更早事件");more.onclick=()=>{state.eventLimit+=200;drawEvents();};target.append(more);}
  events.slice(-state.eventLimit).forEach(event=>{const info=eventInfo(event),details=el("details","event-item"),summary=el("summary"),roles={researcher:"主研究者",research:"主研究者",auditor:"独立审计",audit_inspect:"审计检查",audit_judge:"审计判断",host:"宿主工具",unrecorded:"未记录角色"};details.dataset.seq=String(event.seq);details.open=opened.has(String(event.seq));summary.append(el("span","event-time","#"+event.seq),badge(roles[info.role]||info.role),el("span","event-title",[info.type,info.tool,info.payload.error_type].filter(Boolean).join(" · ")));details.append(summary,el("pre","",JSON.stringify(event,null,2)));target.append(details);});
  if(!events.length)target.append(el("p","empty-state",state.events.length?"没有匹配的事件":"暂无持久化事件"));target.scrollTop=atBottom?target.scrollHeight:scrollTop;
}
function draw() {
  const run=state.run,view=state.view,primary=view?.primary||"主指标";$("breadcrumb-current").textContent=run.run_id;$("run-title").textContent=run.task.question||run.task.task_id;$("run-status").replaceChildren(badge(labels[run.status]||run.status,tone(run.status)));
  $("run-meta").textContent=[run.run_id,run.task.dataset_id||run.report?.context?.dataset_id,"更新 "+updated(run.updated)].filter(Boolean).join("　·　");$("cancel-task").hidden=run.origin==="study_session"||!["queued","waiting_input"].includes(run.status);$("experiment-count").textContent=view?.entries?.length||0;
  const demo=run.report?.kind==="evidence_reference";$("result-status").replaceChildren(badge(demo?"证据演示":view?.improved?"存在合格提升":"尚无合格提升",!demo&&view?.improved?"positive":"neutral"));
  $("conclusion").className="conclusion"+(run.status==="failed"?" error-box":run.status==="incomplete"||run.status==="waiting_input"?" warning-box":"");const text=el("div");text.append(el("h2","",view?.summary||run.state?.message||"尚无可用报告"),el("p","",demo?"本地参考工具调用，不使用模型或 GPU，不代表推荐收益。":"以独立评分、比较条件与过程证据判断改进；运行结束不等于质量提升。"));$("conclusion").replaceChildren(el("span","conclusion-icon",run.status==="failed"?"!":"•"),text);
  const usage=run.report?.budget_usage||run.state?.budget_usage||{},best=view?.best,verifiedBest=best?.verified?best:null;
  $("metric-cards").replaceChildren(metricCard("基线 "+primary,format(view?.baseline?.metrics?.[primary]),view?.baseline?.verified?"独立评分已核验":"评分未核验或未记录"),metricCard("最佳合格 "+primary,format(verifiedBest?.metrics?.[primary]),view?.improved?relative(best.relative?.[primary])+" 相对基线":"尚无合格候选超过基线",!!view?.improved),metricCard("实验记录",String(allEntries().filter(entry=>entry.record?.experiment_id).length),"包含基线、失败与未评分记录",false,"次"),metricCard("模型请求",String(usage.model_requests??run.report?.model_calls??usage.model_calls??"—"),"未报告费用不视为零费用"));
  drawStages();drawProgress();drawExperiments();drawAuditAndIssues();drawChanges();drawReport();drawList();showTab(state.tab);
}
async function refreshRun() {
  if(!state.id)return;if(state.refreshing){state.refreshQueued=true;return state.refreshing.catch(()=>{});}
  const id=state.id,generation=state.generation,controller=new AbortController();state.controller=controller;
  const work=(async()=>{const base="/api/runs/"+encodeURIComponent(id);const [run,report]=await Promise.all([api(base,{signal:controller.signal}),api(base+"/report",{signal:controller.signal})]);
    if(id!==state.id||generation!==state.generation)return;
    if(JSON.stringify(run.report)!==JSON.stringify(report.report)){scheduleRefresh();return;}
    state.run=run;state.view=report.view;$("notice").hidden=true;$("workspace-content").hidden=false;$("welcome").hidden=true;draw();})();state.refreshing=work;
  try{await work;}catch(error){if(id===state.id&&generation===state.generation)reportError(error);}finally{if(state.refreshing===work){state.refreshing=null;if(state.refreshQueued){state.refreshQueued=false;scheduleRefresh();}}}
}
function scheduleRefresh(){if(state.eventTimer)return;state.eventTimer=setTimeout(()=>{state.eventTimer=null;refreshRun();},250);}
async function selectRun(id) {
  state.stream?.close();state.controller?.abort();state.generation++;state.refreshing=null;state.refreshQueued=false;clearTimeout(state.eventTimer);state.eventTimer=null;
  const generation=state.generation;
  state.id=id;state.run=null;state.view=null;state.group=null;state.events=[];state.seen.clear();state.eventLimit=200;$("workspace-content").hidden=true;$("welcome").hidden=false;$("welcome").textContent="正在加载研究会话…";$("notice").hidden=true;drawEvents();drawList();$("sidebar").classList.remove("open");$("sidebar-toggle").setAttribute("aria-expanded","false");if($("experiment-dialog").open)$("experiment-dialog").close();
  await refreshRun();if(state.id!==id||generation!==state.generation)return;const stream=new EventSource("/api/runs/"+encodeURIComponent(id)+"/events");state.stream=stream;
  stream.addEventListener("trace",event=>{if(generation!==state.generation)return;try{const value=JSON.parse(event.data);if(!Number.isInteger(value.seq)||state.seen.has(value.seq))return;state.seen.add(value.seq);state.events.push(value);state.events.sort((a,b)=>a.seq-b.seq);drawEvents();scheduleRefresh();}catch{$("notice").hidden=false;$("notice").textContent="事件格式异常，已保留之前的记录。";}});
  stream.addEventListener("error",()=>{if(generation===state.generation)$("event-connection").textContent="事件流断开，正在重连；保留已有记录";});stream.addEventListener("open",()=>{if(generation===state.generation)$("event-connection").textContent="事件流已连接，按序号去重与续读";});
  stream.addEventListener("end",()=>{stream.close();if(generation!==state.generation)return;$("event-connection").textContent="本轮事件已结束";scheduleRefresh();listRuns().catch(reportError);});
}
document.querySelectorAll("[data-tab]").forEach(button=>{button.onclick=()=>showTab(button.dataset.tab);button.onkeydown=event=>{const buttons=[...document.querySelectorAll("[data-tab]")],i=buttons.indexOf(button);const next=event.key==="ArrowRight"?(i+1)%buttons.length:event.key==="ArrowLeft"?(i-1+buttons.length)%buttons.length:event.key==="Home"?0:event.key==="End"?buttons.length-1:null;if(next!==null){event.preventDefault();showTab(buttons[next].dataset.tab,true);}};});
document.querySelectorAll("[data-close]").forEach(button=>{button.onclick=()=>$(button.dataset.close).close();});
$("search").oninput=drawList;$("status-filter").onchange=drawList;$("new-task").onclick=()=>$("task-dialog").showModal();$("group-picker").onchange=()=>{state.group=$("group-picker").value;drawProgress();};$("sidebar-toggle").onclick=()=>$("sidebar-toggle").setAttribute("aria-expanded",String($("sidebar").classList.toggle("open")));
$("refresh-button").onclick=()=>Promise.all([listRuns(),refreshRun()]).catch(reportError);
$("cancel-task").onclick=async()=>{const id=state.id,generation=state.generation;$("cancel-task").disabled=true;try{await api("/api/runs/"+encodeURIComponent(id)+"/cancel",{method:"POST"});if(generation===state.generation)await selectRun(id);await listRuns();}catch(error){if(generation===state.generation)reportError(error);}finally{$("cancel-task").disabled=false;}};
$("demo-button").onclick=async()=>{$("demo-button").disabled=true;try{const run=await api("/api/demo",{method:"POST"});await listRuns();await selectRun(run.run_id);}catch(error){reportError(error);}finally{$("demo-button").disabled=false;}};
$("task-form").onsubmit=async event=>{
  event.preventDefault();const form=event.target,payload=Object.fromEntries(new FormData(form)),encoded=JSON.stringify(payload);if(state.requestPayload!==encoded){state.requestPayload=encoded;state.requestKey=crypto.randomUUID();}const button=$("task-submit");button.disabled=true;
  try{const run=await api("/api/tasks",{method:"POST",headers:{"Content-Type":"application/json","Idempotency-Key":state.requestKey},body:encoded});state.requestKey=null;state.requestPayload=null;$("task-dialog").close();form.reset();$("search").value="";$("status-filter").value="all";await listRuns();await selectRun(run.run_id);toast("任务已保存；实际执行取决于已配置的执行器。");}catch(error){$("task-error").textContent=error.message;}finally{button.disabled=false;}
};
$("task-form").addEventListener("input",()=>{$("task-error").textContent="";});$("event-role").onchange=drawEvents;$("event-type").onchange=drawEvents;$("event-search").oninput=drawEvents;
document.addEventListener("keydown",event=>{if(event.key==="/"&&!$("task-dialog").open&&!$("experiment-dialog").open&&!["INPUT","TEXTAREA","SELECT"].includes(document.activeElement.tagName)){event.preventDefault();$("sidebar").classList.add("open");$("sidebar-toggle").setAttribute("aria-expanded","true");$("search").focus();}});
api("/health").then(health=>{$("connection-label").textContent=health.execution_mode==="offline"?"本地工作台":"已连接执行器";$("execution-label").textContent=health.execution_mode==="offline"?"执行器未配置":"外部执行器";}).catch(reportError);
listRuns().catch(reportError);setInterval(()=>{if(state.id)refreshRun();listRuns().catch(reportError);},3000);
