"use strict";
// Plot only host-qualified values; acceptance is never inferred in the browser.
const RecLabCharts = (() => {
  function svg(tag, attrs = {}, text) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function render(target, group, primary, onSelect) {
    target.replaceChildren();
    if (!group || !Number.isFinite(group.baseline.value)) {
      const empty = document.createElement("p"); empty.className = "empty-state";
      empty.textContent = "基线评分尚未核验，暂不能展示提升曲线。"; target.append(empty); return;
    }
    const points = [{...group.baseline, order: 0, best: group.baseline.value, qualified: true, baseline: true}, ...group.points];
    const scored = points.filter(p => Number.isFinite(p.value)), width = Math.max(620, points.length * 96);
    const top = Math.max(65, 18 + points.filter(p => p.milestone).length * 35), bottom = top + 190, height = bottom + 41;
    const values = scored.map(p => p.value), min = Math.min(...values), max = Math.max(...values);
    const pad = Math.max((max - min) * .28, Math.abs(max) * .025, .0001), low = min - pad, high = max + pad;
    const x = i => 60 + i * (width - 90) / Math.max(points.length - 1, 1), y = v => top + (high - v) / (high - low) * 190;
    const root = svg("svg", {viewBox: `0 0 ${width} ${height}`, class: "metric-svg", role: "group", "aria-label": `${primary} 核心目标提升图`});
    if (points.length > 8) root.style.width = width + "px";
    root.append(svg("title", {}, `${primary} · 验证集实际值`), svg("desc", {}, "虚线为基线，绿色阶梯为最佳合格结果；空心点未满足接受条件，失败不填零分。图点可用键盘打开详情。"));
    for (let i = 0; i < 5; i++) {
      const value = low + (high - low) * i / 4, yy = y(value);
      root.append(svg("line", {x1: 60, x2: width - 30, y1: yy, y2: yy, class: "grid"}), svg("text", {x: 48, y: yy + 3, "text-anchor": "end", class: "axis-label"}, value.toPrecision(4)));
    }
    root.append(svg("line", {x1: 60, x2: width - 30, y1: y(group.baseline.value), y2: y(group.baseline.value), class: "baseline"}));
    let path = "";
    points.forEach((p, i) => {if (Number.isFinite(p.best)) path += path ? ` H ${x(i)} V ${y(p.best)}` : `M ${x(i)} ${y(p.best)}`;});
    root.append(svg("path", {d: path, class: "best-path"}));
    let row = 0;
    points.forEach((p, i) => {
      const xx = x(i); root.append(svg("text", {x: xx, y: bottom + 26, "text-anchor": "middle", class: "axis-label"}, p.baseline ? "基线" : String(p.order).padStart(2, "0")));
      if (!Number.isFinite(p.value)) return;
      const yy = y(p.value), node = svg("g", {class: "point", tabindex: "0", role: "button", "pointer-events": "bounding-box", "aria-label": `${p.id}，${primary} ${p.value}，${p.decision || "基线"}，查看详情`});
      node.append(svg("title", {}, `${p.id} · ${p.value} · ${p.decision || "基线"}`));
      if (p.milestone) {
        const ly = 18 + row++ * 35, lx = Math.min(width - 230, Math.max(60, xx - 25));
        const gain = p.relative == null ? "相对提升无法计算" : `${p.relative > 0 ? "+" : ""}${p.relative.toFixed(2)}%`;
        const label = p.id.length > 20 ? p.id.slice(0, 18) + "…" : p.id;
        node.append(svg("path", {d: `M ${lx + 12} ${ly + 19} L ${xx} ${yy - 11}`, class: "leader"}), svg("text", {x: lx, y: ly, class: "point-label"}, `${label} ${gain}`), svg("text", {x: lx, y: ly + 13, class: "point-value"}, `${primary} ${p.value.toPrecision(6)}`), svg("circle", {cx: xx, cy: yy, r: 10, class: "point-halo"}));
      }
      node.append(svg("circle", {cx: xx, cy: yy, r: p.milestone ? 5 : 4, fill: p.qualified ? "#388d64" : "#fff", stroke: p.qualified ? "#388d64" : "#a99873", "stroke-width": 2}));
      node.addEventListener("click", () => onSelect(p.experiment_id));
      node.addEventListener("keydown", event => {if (event.key === "Enter" || event.key === " ") {event.preventDefault(); onSelect(p.experiment_id);}});
      root.append(node);
    });
    target.append(root);
  }
  return {render};
})();
