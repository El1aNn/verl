# Overleaf Bundle 2026-04-06

这个目录是给 Overleaf 直接使用的双语报告工程。

## 主文件

- 默认主文件：`main.tex`（指向中文版）
- 中文版：`main_cn.tex`
- 英文版：`main_en.tex`

## 使用方式

1. 把整个目录压缩包上传到 Overleaf。
2. 默认直接编译 `main.tex` 即可生成中文版。
3. 如果要英文版，把主文件切换成 `main_en.tex`。
4. 如果 Overleaf 没有自动切到 `XeLaTeX`，请手动切换；工程内也附带了 `.latexmkrc` 来尽量强制使用 `xelatex`。
5. 按需修改 `metadata.tex` 里的题目、作者、单位、课程和日期。

## 目录说明

- `data/`：由本地 validation 结果和 SwanLab 导出曲线整理出的 CSV 数据。
- `tables/`：可直接 `\input{}` 到 LaTeX 的表格行。
- `figures/`：正文与附录会直接使用的 PNG 图片，已经包含更多 SwanLab 导出的静态曲线与对齐版 `n=1` 对比图。
- `scripts/build_report_assets.py`：重新生成 CSV 与表格的脚本。
- `shared-setup.tex`：中英文共用的包、绘图样式和 pgfplots 宏。

其中新增的 SwanLab 相关文件包括：

- `data/swanlab-training-curves.csv`
- `data/swanlab-validation-curves.csv`
- `data/swanlab-dynamic-summary.csv`
- `tables/swanlab-dynamics-rows.tex`

## 当前内容特点

- 中文版与英文版共用同一套数据和图表，但正文是分别写的，不是逐句硬翻。
- 当前正式稿已经采用新的对齐版 `n=1` 实验：`base_c_n1_val8_align_exp1_2gpu_20260406_195153`。
- 主结果图与大部分补充图已经固定为静态 PNG，Overleaf 编译更稳，也更适合快速增删图页。
- token 级补充图直接使用已有 PNG，方便插入与复用。
- SwanLab 导出的训练熵曲线、64-sample 聚合 token-summary 曲线、以及对齐版 `n=1` 与 `n=8` 的多指标对比图已经纳入正文与附录，用于补强“训练动态”和“token 级行为变化”的论证。

## 本地编译

- 本机已经完成 `XeLaTeX` 本地编译检查。
- 生成文件：
  - `main_cn.pdf`
  - `main_en.pdf`
- 当前页数：
  - 中文版 `27` 页
  - 英文版 `26` 页

## 已知事项

- reward shuffle 的数据完整性问题已经在正文中明确保留为 caveat。
- `n=1` 与 `n=8` 的比较现在已经统一到 `val_rollout_n=8` 的对齐口径，但 group-size 结论仍然只基于单次正式 run，而非多 seed 结果。
