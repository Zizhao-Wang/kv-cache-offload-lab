# 04 Result Management

结果目录约定：

- `results/summaries/`: 文本或 JSON 摘要
- `results/figures/`: 图像与绘图产物
- `results/tables/`: CSV、Markdown 表格

同时建议：

1. 原始日志存放在运行目录，不直接混入汇总目录。
2. 每次实验创建独立 run id。
3. 汇总脚本只消费稳定格式的日志和 manifest。
4. 关键对比结果优先沉淀为表格和简要结论。
