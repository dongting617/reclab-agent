"use strict";
/* Design fixture only. No fetch, EventSource, storage, model or execution calls. */
const $ = (id) => document.getElementById(id);
function el(tag, className = "", text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function badge(text, tone = "neutral") {
  const node = el("span", "tag " + tone);
  node.append(el("i", "tag-dot"), document.createTextNode(text));
  return node;
}
const format = (value) => typeof value === "number" ? value.toFixed(5) : "—";
const signed = (value, digits = 5) => value == null ? "—" : (value > 0 ? "+" : "") + value.toFixed(digits);
const relative = (value) => value == null ? "无法计算" : signed(value, 2) + "%";
const clone = (value) => JSON.parse(JSON.stringify(value));

// Eligibility, deltas and milestones represent an already normalized report view.
// The production adapter must obtain these from the host, never infer acceptance here.
const fixtureEntries = [
  {id:"baseline",name:"原始基线",kind:"baseline",value:.08045,recall:.15543,delta:0,relative:0,best:.08045,eligible:true,verified:true,comparable:true,decision:"基线",tone:"neutral",note:"固定配置 · seed 2026",status:"completed"},
  {id:"candidate-01",name:"调整 dropout",value:.07742,recall:.14777,delta:-.00303,relative:-3.7663,best:.08045,eligible:false,verified:true,comparable:true,decision:"护栏未达标",tone:"negative",note:"dropout：0.40 → 0.15",status:"completed",diff:["- dropout = 0.40","+ dropout = 0.15"]},
  {id:"candidate-02",name:"调整学习率",value:.08210,recall:.15620,delta:.00165,relative:2.05096,best:.08210,eligible:true,verified:true,comparable:true,milestone:true,previousDelta:.00165,decision:"刷新最佳",tone:"positive",note:"学习率：1e-3 → 5e-4",status:"completed",diff:["- learning_rate = 1e-3","+ learning_rate = 5e-4"]},
  {id:"candidate-03",name:"扩大隐藏维度",value:null,recall:null,delta:null,relative:null,best:.08210,eligible:false,verified:false,comparable:false,decision:"执行失败",tone:"negative",note:"Smoke Test 未通过 · 张量维度不匹配",status:"failed",diff:["- hidden_size = 64","+ hidden_size = 128"]},
  {id:"candidate-04",name:"校正序列截断",value:.08530,recall:.16180,delta:.00485,relative:6.02859,best:.08530,eligible:true,verified:true,comparable:true,milestone:true,previousDelta:.00320,decision:"刷新最佳",tone:"positive",note:"按时间保留最近的有效交互",status:"completed",diff:["- sequence = history[:max_length]","+ sequence = history[-max_length:]"]},
  {id:"candidate-05",name:"调整权重衰减",value:.08410,recall:.15900,delta:.00365,relative:4.5370,best:.08530,eligible:true,verified:true,comparable:true,decision:"未超过最佳",tone:"neutral",note:"优于基线，低于当前最佳",status:"completed",diff:["- weight_decay = 0","+ weight_decay = 1e-5"]},
  {id:"candidate-06",name:"增加注意力头数",value:.08710,recall:.16210,delta:.00665,relative:8.2660,best:.08530,eligible:false,verified:true,comparable:true,decision:"证据不足",tone:"warning",note:"必需的过程诊断未完成",status:"completed",diff:["- attention_heads = 2","+ attention_heads = 4"]}
];
const fixtures = [
  {id:"design-gains",title:"SASRec · Beauty 推荐效果优化",short:"SASRec · Beauty 效果优化",subtitle:"Beauty / SASRec",status:"completed",statusText:"已结束",tone:"positive",updated:"14:32",runId:"study-sasrec-beauty-001",summary:"找到 2 次合格改进，当前最佳来自「校正序列截断」",explanation:"NDCG@20 相对基线提升 6.03%，Recall@20 未下降。更高的原始分数因缺少必需诊断，未计入最佳结果。",entries:fixtureEntries,audit:"approved",requests:24,attempts:7,stages:4},
  {id:"design-decline",title:"SASRec · Dropout 参数验证",short:"SASRec · Dropout 参数验证",subtitle:"Beauty / SASRec",status:"completed",statusText:"已结束",tone:"neutral",updated:"13:48",runId:"study-sasrec-dropout-002",summary:"候选未改善，保留原始基线",explanation:"NDCG@20 与 Recall@20 均低于基线。保留负结果，用于后续假设与参数选择。",entries:fixtureEntries.slice(0,2),audit:"approved",requests:8,attempts:2,stages:4},
  {id:"design-failure",title:"GRURec · 模型结构迁移",short:"GRURec · 模型结构迁移",subtitle:"Beauty / GRURec",status:"failed",statusText:"执行失败",tone:"negative",updated:"12:16",runId:"study-grurec-migration-003",summary:"Smoke Test 未通过，没有可用的候选评分",explanation:"候选出现张量维度不匹配，未进入训练与独立评分。已有基线可继续使用，需先修复结构接口。",entries:[clone(fixtureEntries[0]),clone(fixtureEntries[3])],audit:"none",requests:6,attempts:2,stages:2},
  {id:"design-evidence",title:"SASRec · 注意力结构研究",short:"SASRec · 注意力结构研究",subtitle:"Beauty / SASRec",status:"incomplete",statusText:"证据不足",tone:"warning",updated:"11:05",runId:"study-attention-004",summary:"指标提高，但目前不足以接受候选",explanation:"必需过程诊断未完成，独立审计返回证据不足。该候选不进入最佳曲线。",entries:[clone(fixtureEntries[0]),{...clone(fixtureEntries[6]),best:.08045}],audit:"insufficient",requests:12,attempts:2,stages:4}
];
const state = {selected:fixtures[0],tab:"overview",events:[],eventLimit:200};
const statusNames = {completed:"已结束",failed:"执行失败",incomplete:"证据不足",waiting_input:"等待配置"};
let toastTimer;
function toast(message) {
  $("toast").textContent = message; $("toast").hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { $("toast").hidden = true; }, 4000);
}
function showTab(name, focus = false) {
  state.tab = name;
  document.querySelectorAll("[data-tab]").forEach((button) => {
    const selected = button.dataset.tab === name;
    button.setAttribute("aria-selected", String(selected)); button.tabIndex = selected ? 0 : -1;
    if (selected && focus) button.focus();
  });
  document.querySelectorAll('[role="tabpanel"]').forEach((panel) => { panel.hidden = panel.id !== "panel-" + name; });
}
function renderList() {
  const query = $("search").value.trim().toLowerCase(), filter = $("status-filter").value;
  const matches = fixtures.filter((run) => (filter === "all" || filter === run.status) &&
    [run.title,run.subtitle,run.runId].join(" ").toLowerCase().includes(query));
  $("run-list").replaceChildren();
  matches.forEach((run) => {
    const button = el("button", "run-item" + (state.selected.id === run.id ? " selected" : ""));
    button.type = "button"; button.setAttribute("aria-pressed", String(state.selected.id === run.id));
    button.append(el("span", "run-title",run.short),el("span", "run-description",run.subtitle));
    const bottom = el("span", "run-bottom"); bottom.append(badge(run.statusText,run.tone),el("span","",run.updated)); button.append(bottom);
    button.onclick = () => { state.selected = run; state.eventLimit = 200; render(); $("sidebar").classList.remove("open"); $("sidebar-toggle").setAttribute("aria-expanded","false"); };
    $("run-list").append(button);
  });
  if (!matches.length) $("run-list").append(el("p","empty-state","没有匹配的研究会话"));
}
function metricCard(name, value, note, positive = false, suffix = "") {
  const node = el("div","metric-card"); node.append(el("div","metric-name",name));
  const number = el("div","metric-value",value); if(suffix) number.append(el("small","",suffix));
  node.append(number,el("div","metric-note" + (positive ? " positive-text" : ""),note)); return node;
}
function renderStages(run) {
  $("stages").replaceChildren();
  ["背景调研","研究假设","实验执行","结果分析"].forEach((name,index) => {
    const failed = run.status === "failed" && index === run.stages;
    const done = index < run.stages;
    const stage = el("div","stage" + (failed ? " failed" : done ? "" : " pending"));
    stage.append(el("span","stage-icon",failed ? "!" : done ? "✓" : String(index + 1)));
    const label = el("div","stage-label",name); label.append(el("small","",failed ? "执行中断" : done ? "已有记录" : "未记录")); stage.append(label); $("stages").append(stage);
  });
}
function svgNode(tag, attrs = {}, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg",tag);
  Object.entries(attrs).forEach(([key,value]) => node.setAttribute(key,String(value)));
  if(text !== undefined) node.textContent = text; return node;
}
function renderChart(entries) {
  const container = $("metric-chart"); container.replaceChildren();
  $("status-track").replaceChildren();
  const baseline = entries[0], points = entries.filter(e => e.value != null && e.verified && e.comparable);
  const width = Math.max(620,entries.length * 96), left = 59, right = 29;
  const marked = entries.filter(e => e.milestone), labelRows = marked.length;
  // Each milestone gets its own band: no overlap even with dense experiments.
  const top = Math.max(65, 18 + labelRows * 35), plotHeight = 190, bottom = top + plotHeight, height = bottom + 41;
  const values = points.map(e => e.value);
  if(!values.length) { container.append(el("div","empty-state","暂无已核验、可比较的验证集评分")); return; }
  const min = Math.min(...values), max = Math.max(...values), pad = Math.max((max-min)*.28,Math.abs(max)*.025,.0001);
  const low = min-pad, high = max+pad;
  const x = (index) => left + index * (width-left-right) / Math.max(entries.length-1,1);
  const y = (value) => top + (high-value) / (high-low) * plotHeight;
  const svg = svgNode("svg",{viewBox:`0 0 ${width} ${height}`,class:"metric-svg",role:"img","aria-labelledby":"chart-title chart-description"});
  svg.append(svgNode("title",{id:"chart-title"},"NDCG@20 核心目标提升图 · 示例数据"),svgNode("desc",{id:"chart-description"},"按实验顺序展示独立评分。虚线为基线，绿色阶梯为最佳合格结果。候选详情亦可通过下方列表及实验对比表访问。"));
  for(let i=0;i<5;i++) {
    const value = low + (high-low)*i/4, yy = y(value);
    svg.append(svgNode("line",{x1:left,y1:yy,x2:width-right,y2:yy,class:"grid"}),svgNode("text",{x:left-12,y:yy+3,"text-anchor":"end",class:"axis-label"},value.toFixed(4)));
  }
  if(baseline.value != null) svg.append(svgNode("line",{x1:left,y1:y(baseline.value),x2:width-right,y2:y(baseline.value),class:"baseline"}));
  let path = "";
  entries.forEach((entry,index) => {
    if(entry.best == null) return;
    path += path ? ` H ${x(index)} V ${y(entry.best)}` : `M ${x(index)} ${y(entry.best)}`;
  });
  svg.append(svgNode("path",{d:path,class:"best-path"}));
  let labelIndex = 0;
  entries.forEach((entry,index) => {
    const xx = x(index);
    svg.append(svgNode("text",{x:xx,y:bottom+26,"text-anchor":"middle",class:"axis-label"},index===0?"基线":String(index).padStart(2,"0")));
    if(!points.includes(entry)) return;
    const yy = y(entry.value), group = svgNode("g",{class:"point",tabindex:"0",role:"button","aria-label":`${entry.name}，NDCG@20 ${format(entry.value)}，${entry.decision}，查看详情`});
    group.append(svgNode("title",{},`${entry.name} · ${format(entry.value)} · ${entry.decision}`));
    if(entry.milestone) {
      const labelY = 18 + labelIndex++ * 35;
      const labelX = Math.min(width-175,Math.max(left,xx-20));
      group.append(svgNode("path",{d:`M ${labelX+12} ${labelY+19} L ${xx} ${yy-11}`,class:"leader"}));
      group.append(svgNode("text",{x:labelX,y:labelY,class:"point-label"},`${entry.name} ${relative(entry.relative)}`));
      group.append(svgNode("text",{x:labelX,y:labelY+13,class:"point-value"},`NDCG@20 ${format(entry.value)}`));
      group.append(svgNode("circle",{cx:xx,cy:yy,r:10,class:"point-halo"}));
    }
    group.append(svgNode("circle",{cx:xx,cy:yy,r:entry.milestone?5:4,fill:entry.eligible?"#388d64":"#fff",stroke:entry.eligible?"#388d64":entry.tone==="warning"?"#bda269":"#a5b1a7","stroke-width":2}));
    group.addEventListener("click",() => showDetail(entry));
    group.addEventListener("keydown",(event) => { if(event.key==="Enter"||event.key===" ") { event.preventDefault();showDetail(entry); }});
    svg.append(group);
  });
  container.append(svg);
  $("status-track").replaceChildren(el("span","","实验记录"));
  entries.slice(1).forEach((entry,index) => {
    const button = el("button","status-pill " + (!entry.eligible ? entry.tone : ""),`${String(index+1).padStart(2,"0")} · ${entry.value==null ? "未评分 / "+entry.decision : entry.decision}`);
    button.onclick = () => showDetail(entry); button.setAttribute("aria-label",`${entry.name}：${entry.decision}，查看详情`); $("status-track").append(button);
  });
  if(entries.length===1) $("status-track").append(el("span","","尚无候选实验"));
}
function renderMilestones(entries) {
  const milestones = entries.filter(e=>e.milestone); $("milestone-count").textContent = `${milestones.length} 次突破`;
  $("milestones").replaceChildren();
  milestones.forEach(entry => {
    const button = el("button","milestone-item");
    const top = el("div","milestone-top"); top.append(el("span","",entry.id),el("strong","",relative(entry.relative)));
    button.append(top,el("h4","",entry.name),el("p","",`${format(entry.value)} · 较上次最佳 ${signed(entry.previousDelta)}`),el("p","",`较基线 ${signed(entry.delta)} · ${entry.note}`));
    button.onclick = ()=>showDetail(entry); $("milestones").append(button);
  });
  if(!milestones.length) $("milestones").append(el("p","milestone-empty","尚无合格候选超过基线。负结果及证据缺口仍完整保留。"));
}
function renderAudit(run) {
  const approved = run.audit === "approved", missing = run.audit === "none";
  $("audit-status").replaceChildren(badge(missing?"未记录":approved?"示例 · 通过":"示例 · 证据不足",missing?"neutral":approved?"positive":"warning"));
  $("audit-content").replaceChildren();
  const row = el("div","audit-row"); row.append(el("span","audit-symbol",missing?"—":approved?"✓":"!"));
  const body = el("div"); body.append(el("strong","",missing?"当前会话没有独立审计记录":approved?"已核查比较范围、护栏与候选结论":"缺少必需诊断，暂不支持接受候选"),el("small","","此处为审计界面示例，新审计模式尚未完成真实验收。"));
  if(!missing) { const button=el("button","text-action","查看审计证据 →"); button.onclick=()=>showDetail(run.entries.find(e=>e.milestone)||run.entries.at(-1)); body.append(button); }
  row.append(body); $("audit-content").append(row);
}
function issuesFor(run) {
  if(run.status==="waiting_input") return [["任务等待配置","原型不连接模型与执行器。"]];
  const issues = [];
  run.entries.filter(e=>e.status==="failed").forEach(e=>issues.push([e.name+"未通过检查",e.note]));
  run.entries.filter(e=>e.decision==="证据不足").forEach(e=>issues.push([e.name+"需要补充证据",e.note]));
  issues.push(["单种子结果的适用范围","仅反映当前数据、配置与验证集，不代表普遍收益。"]); return issues;
}
function renderIssues(run) {
  const issues=issuesFor(run); $("issue-count").textContent = issues.length+" 项"; $("issues").replaceChildren();
  issues.forEach(([title,description])=>{const row=el("div","issue-item"),body=el("div");body.append(el("strong","",title),el("small","",description));row.append(el("span","issue-marker"),body);$("issues").append(row);});
}
function renderExperiments(entries) {
  $("experiment-rows").replaceChildren();
  entries.forEach(entry=>{
    const row=el("tr",entry.kind==="baseline"?"baseline-row":entry.milestone?"row-milestone":"");
    const name=el("td"),button=el("button","table-button",entry.name);button.onclick=()=>showDetail(entry);name.append(button,el("small","",entry.id));
    const gain=el("td",entry.delta>0?"positive-text":entry.delta<0?"negative-text":"",entry.value==null?"—":relative(entry.relative));
    const eligibility=el("td");eligibility.append(badge(entry.value==null?"未评分":entry.verified?"评分已核验":"未核验",entry.verified?"positive":"neutral"),el("small","",entry.value==null?entry.note:entry.eligible?"满足比较与接受条件":entry.note));
    const result=el("td");result.append(badge(entry.decision,entry.tone));
    row.append(name,el("td","",format(entry.value)),gain,el("td","",format(entry.recall)),eligibility,result);$("experiment-rows").append(row);
  });
}
function rawDetails(title,value) {
  const details=el("details","raw-details");details.append(el("summary","",title),el("pre","",JSON.stringify(value,null,2)));return details;
}
function diffBlock(lines) {
  const pre=el("pre","code-diff");lines.forEach(line=>pre.append(el("span",line.startsWith("+")?"diff-add":line.startsWith("-")?"diff-remove":"",line)));return pre;
}
function evidencePath(path) {
  const node=el("div","evidence-path"),copy=el("button","text-action","复制路径");
  copy.onclick=async()=>{try{await navigator.clipboard.writeText(path);toast("已复制示例路径；该路径不指向真实实验文件。");}catch{toast("无法使用剪贴板，请选中路径手动复制。");}};
  node.append(el("span","",path),copy);return node;
}
function showDetail(entry) {
  $("detail-title").textContent=entry.name;const body=$("detail-body");body.replaceChildren();
  body.append(el("p","callout","示例证据，仅用于界面确认，不代表真实指标、代码改动或审计结果。"));
  const grid=el("div","detail-grid");
  [["候选",entry.id],["验证集 NDCG@20",format(entry.value)],["相对基线",entry.value==null?"—":relative(entry.relative)],["当前结论",entry.decision]].forEach(([key,value])=>{const item=el("div");item.append(el("small","",key),el("strong","",value));grid.append(item);});body.append(grid);
  body.append(el("h3","","资格与证据"));
  const list=el("div","content-prose");list.append(el("p","",entry.note),el("p","",entry.value==null?"未产生评分。不将失败当作零分。":`评分：${entry.verified?"示例已核验":"尚未核验"} · 比较条件：${entry.comparable?"一致":"不可比较"} · 接受资格：${entry.eligible?"满足":"不满足"}`));body.append(list);
  if(entry.diff){body.append(el("h3","","src/model.py · 示例差异"),diffBlock(entry.diff));}else{body.append(el("p","content-prose","没有新增改动证据。"));}
  body.append(evidencePath(`examples/${state.selected.id}/${entry.id}/score.json`));
  body.append(rawDetails("展开示例评分与过程诊断",{kind:"design_fixture",metrics:{"ndcg@20":entry.value,"recall@20":entry.recall},process_requirement:entry.decision==="证据不足"?"missing":"example_only",audit:state.selected.audit,decision:entry.decision}));
  $("experiment-dialog").showModal();
}
function renderChanges(run) {
  const target=$("hypothesis-content");target.replaceChildren();
  if(!run.entries.length){target.append(el("div","surface empty-state","尚无研究假设与代码改动记录"));return;}
  const panel=el("section","surface"),heading=el("div","surface-heading"),title=el("div");title.append(el("h2","","研究假设"),el("p","","以下为设计示例，不是历史实验归因。"));heading.append(title,badge("示例计划"));panel.append(heading);
  const grid=el("div","hypothesis-grid");
  [["可检验假设","按时间保留最近有效交互，可能让编码器更好地表示用户近期偏好。"],["预期观察","同协议下 NDCG@20 提高，Recall@20 不低于基线；检查输入序列长度和有效掩码。"],["风险与混杂因素","不得改变验证集用户、目标物品或过滤规则；统一随机种子与运行配置。"],["必需诊断","保留预处理差异、序列切片检查和独立评分；缺少必需诊断则结论为证据不足。"]].forEach(([name,text])=>{const item=el("div","content-prose");item.append(el("h3","",name),el("p","",text));grid.append(item);});panel.append(grid);target.append(panel);
  run.entries.filter(e=>e.diff).forEach(entry=>{const card=el("section","surface"),head=el("div","surface-heading"),content=el("div");content.append(el("h2","",entry.name),el("p","",entry.id+" · src/model.py"));head.append(content,badge(entry.decision,entry.tone));card.append(head,diffBlock(entry.diff));const button=el("button","text-action","查看实验与评分证据 →");button.style.marginTop="14px";button.onclick=()=>showDetail(entry);card.append(button);target.append(card);});
}
function renderReport(run) {
  const target=$("report-content");target.replaceChildren();const panel=el("section","surface"),head=el("div","surface-heading"),title=el("div");title.append(el("h2","","研究结论"),el("p","","界面结论、图表和实验资格使用同一份示例视图。"));head.append(title,badge("示例报告","warning"));panel.append(head);
  const prose=el("div","content-prose");prose.append(el("h3","",run.summary),el("p","",run.explanation),el("p","","边界：单种子验证集结果，验证集参与候选选择；没有独立测试集收益或统计显著性的结论。"));panel.append(prose,rawDetails("查看原始示例 JSON",run));target.append(panel);
  const evidence=el("section","surface");evidence.append(el("h2","","评分与改动证据"));run.entries.forEach(entry=>{const p=el("p","content-prose");p.style.marginTop="14px";const button=el("button","text-action",entry.name+" · "+entry.decision+" →");button.onclick=()=>showDetail(entry);p.append(button);evidence.append(p);});target.append(evidence);
}
function eventsFor(run) {
  if(!run.entries.length)return [];
  const events=[{time:"14:00:02",role:"host",type:"tool_call",name:"resolve_baseline",summary:"读取基线与固定评分协议",result:{seed:2026,protocol:"design-only",baseline:"baseline"}}];
  run.entries.slice(1).forEach((entry,index)=>{
    const time=`14:${String(index*3+1).padStart(2,"0")}:12`;
    events.push({time,role:"researcher",type:"model_turn",name:"hypothesis",summary:entry.name+" · 提出候选改动",result:{candidate:entry.id,scope:"design fixture"}});
    events.push({time,role:"host",type:entry.status==="failed"?"error":"tool_call",name:entry.status==="failed"?"smoke_test":"score_experiment",summary:entry.note,result:{candidate:entry.id,value:entry.value,decision:entry.decision}});
  });
  if(run.audit!=="none")events.push({time:"14:28:41",role:"auditor",type:"model_turn",name:"submit_audit",summary:run.audit==="approved"?"核对证据，确认合格候选范围":"证据不足，不能接受候选",result:{verdict:run.audit,fixture:true}});
  return events;
}
function renderEvents() {
  const role=$("event-role").value,type=$("event-type").value,query=$("event-search").value.toLowerCase();
  const all=state.events.filter(event=>(role==="all"||event.role===role)&&(type==="all"||event.type===type)&&JSON.stringify(event).toLowerCase().includes(query));
  const target=$("event-list");target.replaceChildren();
  if(all.length>state.eventLimit){const more=el("button","button secondary","加载更早事件");more.onclick=()=>{state.eventLimit+=200;renderEvents();};target.append(more);}
  all.slice(-state.eventLimit).forEach(event=>{
    const details=el("details","event-item"),summary=el("summary");
    summary.append(el("span","event-time",event.time),badge({host:"宿主工具",researcher:"主研究者",auditor:"独立审计"}[event.role],event.type==="error"?"negative":"neutral"),el("span","event-title",event.name+" · "+event.summary));
    details.append(summary,el("pre","",JSON.stringify(event,null,2)));target.append(details);
  });
  if(!all.length)target.append(el("p","empty-state","没有匹配的事件"));
}
function render() {
  const run=state.selected,entries=run.entries,baseline=entries[0],best=entries.filter(e=>e.eligible&&e.verified&&e.comparable&&e.value!=null).reduce((acc,item)=>!acc||item.value>acc.value?item:acc,null);
  // Fixtures have explicit host-style acceptance flags. No real eligibility is evaluated here.
  $("breadcrumb-current").textContent=run.runId;$("run-title").textContent=run.title;
  $("run-status").replaceChildren(badge(run.statusText,run.tone));
  $("cancel-task").hidden=run.status!=="waiting_input";
  $("run-meta").textContent=`${run.runId}　·　${run.subtitle}　·　验证集 / seed 2026　·　示例时间 ${run.updated}`;
  $("experiment-count").textContent=entries.length;renderList();renderStages(run);
  $("conclusion").className="conclusion"+(run.tone==="warning"?" warning-box":run.tone==="negative"?" error-box":"");
  const content=el("div");content.append(el("h2","",run.summary),el("p","",run.explanation));$("conclusion").replaceChildren(el("span","conclusion-icon",run.tone==="negative"?"!":run.tone==="warning"?"!":"✓"),content);
  const change=best&&baseline ? best.value-baseline.value : null;
  $("metric-cards").replaceChildren(
    metricCard("基线 NDCG@20",format(baseline?.value),"固定比较基准 · 示例"),
    metricCard("最佳合格 NDCG@20",format(best?.value),change>0?`↑ ${relative(best.relative)} 相对基线`:"尚无合格候选超过基线",change>0),
    metricCard("实验记录",String(run.attempts),`${entries.filter(e=>e.status==="failed").length} 次失败 · 包含基线`,false,"次"),
    metricCard("模型请求",String(run.requests),"示例额度，非实际调用",false,"/ 32")
  );
  $("chart-metric").textContent="NDCG@20 ↑";renderChart(entries);renderMilestones(entries);renderAudit(run);renderIssues(run);renderExperiments(entries);renderChanges(run);renderReport(run);state.events=eventsFor(run);renderEvents();showTab(state.tab);
}
document.querySelectorAll("[data-tab]").forEach(button=>{
  button.onclick=()=>showTab(button.dataset.tab);
  button.onkeydown=(event)=>{const buttons=[...document.querySelectorAll("[data-tab]")],index=buttons.indexOf(button);let next;
    if(event.key==="ArrowRight")next=(index+1)%buttons.length;
    if(event.key==="ArrowLeft")next=(index-1+buttons.length)%buttons.length;
    if(event.key==="Home")next=0;if(event.key==="End")next=buttons.length-1;
    if(next!==undefined){event.preventDefault();showTab(buttons[next].dataset.tab,true);}
  };
});
document.querySelectorAll("[data-close]").forEach(button=>{button.onclick=()=>$(button.dataset.close).close();});
$("search").oninput=renderList;$("status-filter").onchange=renderList;
$("new-task").onclick=()=>$("task-dialog").showModal();
$("refresh-button").onclick=()=>{render();toast("示例视图已刷新；未请求真实会话。");};
$("cancel-task").onclick=()=>{if(state.selected.status!=="waiting_input")return;Object.assign(state.selected,{status:"cancelled",statusText:"已取消",tone:"neutral",summary:"示例任务已取消",explanation:"仅更新页面内的示例状态，未操作任何真实任务。"});render();toast("已取消页面内的等待任务。");};
$("sidebar-toggle").onclick=()=>{const opened=$("sidebar").classList.toggle("open");$("sidebar-toggle").setAttribute("aria-expanded",String(opened));};
$("demo-button").onclick=()=>{state.selected=fixtures[0];showTab("report");render();toast("当前展示静态证据示例，不会运行后台演示工具。");};
$("event-role").onchange=renderEvents;$("event-type").onchange=renderEvents;$("event-search").oninput=renderEvents;
$("task-form").onsubmit=(event)=>{
  event.preventDefault();const data=Object.fromEntries(new FormData(event.target));
  if(fixtures.some(run=>run.runId===data.task_id)){toast("此示例任务 ID 已存在，请更换。");return;}
  const created={id:"local-"+data.task_id,title:data.question,short:data.question,subtitle:data.dataset_id+" / 待配置",status:"waiting_input",statusText:"等待配置",tone:"warning",updated:"刚刚",runId:data.task_id,summary:"示例任务已登记，等待配置",explanation:"该任务仅存在于当前页面内，不会连接模型、启动训练或写入研究记录。",entries:[],audit:"none",requests:0,attempts:0,stages:0};
  fixtures.unshift(created);state.selected=created;$("status-filter").value="all";$("search").value="";$("task-dialog").close();showTab("overview");render();event.target.reset();toast("已登记示例任务；刷新页面后清除。");
};
document.addEventListener("keydown",event=>{if(event.key==="/"&&!$("task-dialog").open&&!$("experiment-dialog").open&&!["INPUT","TEXTAREA","SELECT"].includes(document.activeElement.tagName)){event.preventDefault();$("sidebar").classList.add("open");$("sidebar-toggle").setAttribute("aria-expanded","true");$("search").focus();}});
render();
