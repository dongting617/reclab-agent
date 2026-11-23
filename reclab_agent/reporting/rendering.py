"""Pure Markdown/HTML renderers over a verified report view."""
import re
from html import escape
from reclab_agent.reporting.model import _text, _fmt, _signed, _number


def _md(value):
    value = _text(value, 12_000).replace("\\", "\\\\").replace("|", "\\|").replace("\r", "").replace("\n", " ").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"([`*_\[\]()!#])", r"\\\1", value)


def _table(view):
    baseline = view["baseline"]
    rows = [[baseline["id"], *[_fmt((baseline["metrics"] or {}).get(name)) for name in view["metrics"]], "—", "基线"]]
    for group in view["groups"]:
        for entry in group["entries"]:
            change = entry["deltas"].get(view["primary"])
            relative = entry["relative"].get(view["primary"])
            delta = _signed(change) + (f" / {relative:+.2f}%" if relative is not None else "")
            rows.append([entry["display_id"],
                         *[_fmt((entry["metrics"] or {}).get(name)) for name in view["metrics"]],
                         delta, entry["decision"]])
    return ["候选", *view["metrics"], "主指标变化（绝对 / 相对）", "判定"], rows


def render_markdown(view):
    lines = [f"# {view['title']}", "", f"**状态：{_md(view['status'])}**", "", _md(view["summary"]), ""]
    if view["metadata"]:
        lines += ["## 实验范围", "", *[f"- **{key}**：{_md(value)}" for key, value in view["metadata"]], ""]
    headers, rows = _table(view)
    lines += ["## 实测结果", "", "| " + " | ".join(map(_md, headers)) + " |",
              "| " + " | ".join("---" for _ in headers) + " |",
              *["| " + " | ".join(map(_md, row)) + " |" for row in rows], "",
              "失败或缺失评分以 — 表示，不按零分参与比较。" + view["guard_note"], ""]
    if view["best"] is not None:
        best = view["best"]
        lines += ["## 最佳记录与产物", "", f"**{_md(best.get('display_id', best['id']))}**", ""]
        if best["links"]:
            lines += [" · ".join(f"[{_md(link['label'])}]({link['href']})" for link in best["links"]), ""]
        if not any(link["label"] == "模型权重" for link in best["links"]):
            lines += ["本报告未登记可打开的本地权重文件；其他模型产物请查看训练记录。", ""]
    lines += ["## 执行情况与资源", ""]
    lines += [f"实际训练尝试合计 {view['total_attempts']} 次（含基线）。", ""]
    for group in view["groups"]:
        resources = [f"训练尝试 {group['attempts']} 次", f"可用评分 {group['scored']} 个", f"无可用评分 {group['failed']} 个"]
        if _number(group["calls"]):
            resources.append(f"模型轮次 {group['calls']} 次")
        if _number(group["tokens"].get("total_tokens")):
            resources.append(f"记录 tokens {group['tokens']['total_tokens']:,}")
        if _number(group["server_seconds"]):
            resources.append(f"服务器训练/预测 {group['server_seconds']:.1f} 秒")
        lines.append(f"- **{_md(group['name'])}**：" + "；".join(resources) + "。")
        for skipped in group["skipped"]:
            lines.append(f"  - {_md(skipped['id'])}：{_md(skipped['reason'])}")
    if _number(view["wall_seconds"]):
        lines += ["", f"控制端墙钟耗时 {view['wall_seconds']:.1f} 秒，包含调用、传输和等待，不等同于 GPU 训练时间。"]
    lines += ["", "## 候选改动与失败记录", ""]
    if view.get("error"):
        lines += [f"流程错误：{_md(view['error'])}", ""]
    for entry in [view["baseline"], *view["entries"]]:
        if entry is view["baseline"] and not entry["failure"]:
            continue
        lines += [f"### {_md(entry.get('display_id', entry['id']))}", ""]
        if entry["failure"]:
            lines += [_md(entry["failure"]), ""]
        if entry["changes"]:
            for change in entry["changes"]:
                lines += [f"{_md(change['path'])}（{_md(change['evidence'])}）：", ""]
                # A long fence prevents source content from breaking the block.
                fence = "`" * max(3, max((len(run) for run in re.findall(r"`+", change["diff"])), default=0) + 1)
                lines += [fence + "diff", change["diff"], fence, ""]
                if change.get("truncated"):
                    lines += ["差异显示已截断，完整内容见源码快照。", ""]
        elif entry is not view["baseline"]:
            lines += ["未取得可核对的源码差异；不根据 Agent 的文字总结推断实际改动。", ""]
        if entry["links"]:
            lines += ["产物：" + " · ".join(f"[{_md(link['label'])}]({link['href']})" for link in entry["links"]), ""]
    if view["audit_notes"]:
        lines += ["## 补充审计与诊断", "", *["- " + _md(note) for note in view["audit_notes"]], ""]
    lines += ["## 结论边界与下一步", "", *["- " + _md(note) for note in view["limitations"]], ""]
    if not view["improved"]:
        lines += ["先检查失败日志或保护指标，再决定下一轮修改；不要把会话结束视为优化成功。", ""]
    else:
        lines += ["保留本轮验证集评估的最佳源码、模型产物及复现配置；基线与候选使用同一个固定随机种子，依据已记录的验证集指标给出结论。", ""]
    if view["json_href"]:
        lines += [f"[机器可读实验记录]({view['json_href']})", ""]
    return "\n".join(lines)


def render_html(view):
    """Self-contained offline HTML; no scripts, remote fonts or raw model HTML."""
    e = lambda value: escape(_text(value, 20_000), quote=True)
    headers, rows = _table(view)
    parts = ["<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">",
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             "<title>模型优化实验报告</title><style>",
             "body{margin:0;background:#f3f5f8;color:#17263c;font:16px/1.7 system-ui,'Microsoft YaHei',sans-serif}"
             "main{max-width:1080px;margin:32px auto;padding:0 22px 48px}header,section{background:white;border:1px solid #dce3ed;border-radius:14px;padding:24px;margin:18px 0}"
             "h1{font-size:30px;margin:8px 0 14px}h2{font-size:21px;margin:0 0 16px}h3{font-size:17px;margin:18px 0 8px}"
             ".badge{display:inline-block;background:#edf2fa;border-radius:20px;padding:3px 12px;color:#294b7e;font-size:14px}"
             ".summary{font-size:18px;line-height:1.8;border-left:4px solid #3a6ac0;padding-left:16px}"
             "dl{display:grid;grid-template-columns:130px 1fr;gap:7px 12px}dt{color:#5c6c82}dd{margin:0;overflow-wrap:anywhere}"
             ".scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums}"
             "th,td{text-align:left;padding:11px 12px;border-bottom:1px solid #e5eaf1}th{background:#f5f7fb;font-weight:600}"
             "tr:first-child td{color:#57687e}small,.muted{color:#5c6c82}details{border-top:1px solid #e5eaf1;padding:14px 0}summary{cursor:pointer;font-weight:600}"
             "pre{font:13px/1.6 ui-monospace,Consolas,monospace;background:#f4f6f9;padding:16px;overflow:auto;white-space:pre;max-height:480px}"
             "a{color:#245bad}li{margin:6px 0}.resource{padding:12px 0;border-bottom:1px solid #e5eaf1}"
             "@media(max-width:600px){main{padding:0 12px;margin-top:12px}header,section{padding:18px}h1{font-size:25px}dl{grid-template-columns:90px 1fr}}"
             "@media print{body{background:white}main{max-width:none;margin:0}header,section{break-inside:avoid;border-radius:0}pre{max-height:none;white-space:pre-wrap}details{display:block}}",
             "</style></head><body><main><header>", f'<span class="badge">{e(view["status"])}</span>',
             f'<h1>{e(view["title"])}</h1><p class="summary">{e(view["summary"])}</p>', "<dl>"]
    parts += [f"<dt>{e(key)}</dt><dd>{e(value)}</dd>" for key, value in view["metadata"]]
    parts += ["</dl></header><section><h2>实测结果</h2><div class=\"scroll\"><table><thead><tr>",
              *[f"<th>{e(header)}</th>" for header in headers], "</tr></thead><tbody>"]
    parts += ["<tr>" + "".join(f"<td>{e(cell)}</td>" for cell in row) + "</tr>" for row in rows]
    parts += ["</tbody></table></div><p class=\"muted\">失败或缺失评分以 — 表示，不按零分参与比较。" + e(view["guard_note"]) + "</p></section>"]
    if view["best"] is not None:
        best = view["best"]
        parts += ["<section><h2>最佳记录与产物</h2>", f'<p><strong>{e(best.get("display_id", best["id"]))}</strong></p>']
        if best["links"]:
            parts.append("<p>" + " · ".join(f'<a href="{e(link["href"])}">{e(link["label"])}</a>' for link in best["links"]) + "</p>")
        if not any(link["label"] == "模型权重" for link in best["links"]):
            parts.append('<p class="muted">本报告未登记可打开的本地权重文件；其他模型产物请查看训练记录。</p>')
        parts.append("</section>")
    parts += ["<section><h2>执行情况与资源</h2>", f'<p>实际训练尝试合计 {e(view["total_attempts"])} 次（含基线）。</p>']
    for group in view["groups"]:
        parts += [f'<div class="resource"><strong>{e(group["name"])}</strong> · 训练尝试 {e(group["attempts"])} 次 · 可用评分 {group["scored"]} 个 · 无可用评分 {group["failed"]} 个']
        details = []
        if _number(group["calls"]):
            details.append(f"模型轮次 {group['calls']} 次")
        if _number(group["tokens"].get("total_tokens")):
            details.append(f"记录 tokens {group['tokens']['total_tokens']:,}")
        if _number(group["server_seconds"]):
            details.append(f"服务器训练/预测 {group['server_seconds']:.1f} 秒")
        if details:
            parts.append('<div class="muted">' + e(" · ".join(details)) + "</div>")
        for skipped in group["skipped"]:
            parts.append(f'<p class="muted">{e(skipped["id"])}：{e(skipped["reason"])}</p>')
        parts.append("</div>")
    if _number(view["wall_seconds"]):
        parts.append(f'<p class="muted">控制端墙钟耗时 {view["wall_seconds"]:.1f} 秒，包含调用、传输和等待，不等同于 GPU 训练时间。</p>')
    parts += ["</section><section><h2>候选改动与失败记录</h2>"]
    if view.get("error"):
        parts.append(f'<p>流程错误：{e(view["error"])}</p>')
    for entry in [view["baseline"], *view["entries"]]:
        if entry is view["baseline"] and not entry["failure"]:
            continue
        parts += [f'<details><summary>{e(entry.get("display_id", entry["id"]))} · {e(entry.get("decision", "基线"))}</summary>']
        if entry["failure"]:
            parts.append(f'<p>{e(entry["failure"])}</p>')
        for warning in entry["warnings"]:
            parts.append(f'<p class="muted">{e(warning)}</p>')
        for change in entry["changes"]:
            parts += [f'<h3>{e(change["path"])}</h3><small>{e(change["evidence"])}</small>',
                      f'<pre><code>{e(change["diff"])}</code></pre>']
            if change.get("truncated"):
                parts.append('<p class="muted">差异显示已截断，完整内容见源码快照。</p>')
        if not entry["changes"] and entry is not view["baseline"]:
            parts.append('<p class="muted">未取得可核对的源码差异；不根据 Agent 的文字总结推断实际改动。</p>')
        if entry["links"]:
            parts.append("<p>产物：" + " · ".join(f'<a href="{e(link["href"])}">{e(link["label"])}</a>' for link in entry["links"]) + "</p>")
        parts.append("</details>")
    if view["audit_notes"]:
        parts += ["</section><section><h2>补充审计与诊断</h2><ul>",
                  *[f"<li>{e(note)}</li>" for note in view["audit_notes"]], "</ul>"]
    parts += ["</section><section><h2>结论边界与下一步</h2><ul>",
              *[f"<li>{e(note)}</li>" for note in view["limitations"]], "</ul>"]
    parts.append("<p>" + ("保留本轮验证集评估的最佳源码、模型产物及复现配置；基线与候选使用同一个固定随机种子，依据已记录的验证集指标给出结论。" if view["improved"] else
                          "先检查失败日志或保护指标，再决定下一轮修改；不要把会话结束视为优化成功。") + "</p>")
    if view["json_href"]:
        parts.append(f'<p><a href="{e(view["json_href"])}">机器可读实验记录</a></p>')
    parts.append("</section></main></body></html>\n")
    return "".join(parts)
