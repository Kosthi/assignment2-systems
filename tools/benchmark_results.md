# Transformer 模型性能基准测试结果

**测试时间**: 2026-01-31 00:23:15

## 测试配置

- **Batch Size**: 4
- **Context Length**: 128
- **Warmup Steps**: 5
- **Measured Steps**: 10
- **Dtype**: fp32
- **Mode**: fwd_bwd

## 测试结果

| Model   |   Total Mean (ms) |   Total Std (ms) |   Forward Mean (ms) |   Forward Std (ms) |   Backward Mean (ms) |   Backward Std (ms) |
|:--------|------------------:|-----------------:|--------------------:|-------------------:|---------------------:|--------------------:|
| small   |           463.607 |           27.921 |             211.791 |             36.843 |              251.816 |              14.318 |
