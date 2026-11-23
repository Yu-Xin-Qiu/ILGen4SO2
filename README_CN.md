# ILGen4SO2

面向选择性 SO2 捕集的离子液体条件生成、预测反馈与多目标筛选代码及数据。

本研究以熔点、黏度和毒性为条件构建 SELFIES–Transformer VAE，通过
R = AC × DE × S / SA_max 引导模型反馈，并结合全局和局部采样探索离子液体结构。
三个随机种子的最终采样共生成 30,000 次，得到 10,285 个去重的物化性质可行候选。
严格筛选保留 15 个候选，其中 11 个位于四目标 Pareto 前沿。

## 数据与代码

- 根目录 CSV：熔点、黏度、毒性及 SO2、CO2 溶解度实验数据库。
- `data/generated/`：逐轮反馈预测、最终采样和筛选结果。
- `data/processed/`：物化性质可行候选及按 R 排序的前四个 Pareto 候选。
- `data/figure_tables.json.gz`：论文图表的源数据。
- `data/quantum_chemistry.json`：优化结构及 Gaussian 计算输入。
- `work/text_generation_study/`：模型、训练、预测和反馈代码。
- `configs/current_protocol.json`：约束条件与采样参数。

## 使用方法

在仓库根目录运行：

```bash
python -m pip install -r requirements-analysis.txt
python scripts/reproduce_results.py
python scripts/plot_results.py
```

结果和 600 dpi 图片输出至 `reproduced/`。导出图表 CSV 和独立量化计算输入：

```bash
python scripts/export_source_data.py
```

## 模型来源

物化性质预测模型基于 [ILBERT](https://github.com/Yu-Xin-Qiu/ILBERT)，
原始[模型与数据](https://doi.org/10.5281/zenodo.14601047)已公开。
气体溶解度预测模型来自 [IL4GAS](https://github.com/Yu-Xin-Qiu/IL4GAS)。
本研究使用的模型见[模型说明](docs/MODELS.md)，指标与筛选方法见
[数据和方法](docs/METHODS_AND_DATA.md)。
