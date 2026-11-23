"""Public report facade; model validation and rendering live separately."""
from pathlib import Path
from reclab_agent.infrastructure.files import atomic_write, confined_path
from reclab_agent.reporting.model import build_report_view, _text
from reclab_agent.reporting.rendering import render_html, render_markdown


def write_user_report(output_directory, report, *, context=None):
    """Write human artifacts only; keep the caller's JSON and report unchanged."""
    output = Path(output_directory).absolute()
    output.mkdir(parents=True, exist_ok=True)
    view = build_report_view(report, output_directory=output, context=context)
    html_path, markdown_path = confined_path(output, "report.html"), confined_path(output, "report.md")
    atomic_write(html_path, render_html(view).encode("utf-8"))
    atomic_write(markdown_path, render_markdown(view).encode("utf-8"))
    return {"user_report_path": str(html_path), "user_report_markdown_path": str(markdown_path),
            "user_summary": view["summary"]}


def format_user_summary(report):
    summary = report.get("user_summary")
    if not summary:
        summary = build_report_view(report)["summary"]
    lines = ["实验结果：" + _text(summary, 4000)]
    for key, label in (("user_report_path", "可读报告"), ("user_report_markdown_path", "Markdown"),
                       ("report_path", "实验记录")):
        if report.get(key):
            lines.append(label + "：" + _text(report[key], 2000))
    if report.get("user_report_error"):
        lines.append("可读报告生成失败，已保留实验记录：" + _text(report["user_report_error"]))
    return "\n".join(lines)
