# 本地工作台截图式演示

2026-11-24，使用本地工作台实际展示已完成的 SASRec dropout 实验，没有启动新的训练或模型请求。

[演示视频](../workbench-demo-20261124.mp4)由六张实际界面截图按每张四秒合成，共 24 秒；这是截图式演示，不是连续录屏。原图没有修改，浏览器全屏隐藏了其他标签页。

| 画面 | 内容 |
|---|---|
| frame-01.jpg | 已完成会话、一次实验和 19 次模型请求 |
| frame-02.jpg | 假设与改动页 |
| frame-03.jpg | dropout 0.4 → 0.15 的假设、范围与过程约束 |
| frame-04.jpg | 基线和候选的独立验证集指标 |
| frame-05.jpg | 没有找到达标候选、保留基线的报告 |
| frame-06.jpg | 持久化调用轨迹 |

候选两项验证指标均下降。界面的“完成”表示会话结束，不能理解为模型提升。原始实验与恢复限制见 [GPU 验收报告](../gpu-acceptance-20261123.md)。

合成命令（Linux/WSL，需本地 FFmpeg）：

```bash
ffmpeg -framerate 1/4 -i reports/workbench-demo-20261124/frame-%02d.jpg \
  -c:v libx264 -r 24 -pix_fmt yuv420p -movflags +faststart \
  reports/workbench-demo-20261124.mp4
```
