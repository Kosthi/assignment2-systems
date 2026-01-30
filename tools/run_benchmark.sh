#!/bin/bash
# Transformer 模型性能基准测试脚本
# 测试 5 种不同规模的模型并输出 Markdown 格式的结果表格

set -e

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 模型规模列表
MODELS=("small" "medium" "large" "xl" "2.7B")

MODELS=("small")

# 默认参数
BATCH_SIZE=${BATCH_SIZE:-4}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-128}
WARMUP_STEPS=${WARMUP_STEPS:-5}
STEPS=${STEPS:-10}
DTYPE=${DTYPE:-"fp32"}
MODE=${MODE:-"fwd_bwd"}
OUTPUT_FILE=${OUTPUT_FILE:-"benchmark_results.md"}

# 临时文件存储 JSON 结果
TEMP_JSON_FILE=$(mktemp)
trap "rm -f $TEMP_JSON_FILE" EXIT

echo "=== Transformer 模型性能基准测试 ==="
echo "Batch Size: $BATCH_SIZE"
echo "Context Length: $CONTEXT_LENGTH"
echo "Warmup Steps: $WARMUP_STEPS"
echo "Measured Steps: $STEPS"
echo "Dtype: $DTYPE"
echo "Mode: $MODE"
echo ""

# 收集所有模型的测试结果
echo "[" > "$TEMP_JSON_FILE"
first=true

for model in "${MODELS[@]}"; do
    echo "正在测试模型: $model ..."
    
    # 运行基准测试并获取 JSON 输出
    result=$(python benchmarking_script.py \
        --model "$model" \
        --batch_size "$BATCH_SIZE" \
        --context_length "$CONTEXT_LENGTH" \
        --warmup-steps "$WARMUP_STEPS" \
        --steps "$STEPS" \
        --dtype "$DTYPE" \
        --mode "$MODE" \
    )
    
    if [ "$first" = true ]; then
        first=false
    else
        echo "," >> "$TEMP_JSON_FILE"
    fi
    echo "$result" >> "$TEMP_JSON_FILE"
    
    echo "  完成: $model"
done

echo "]" >> "$TEMP_JSON_FILE"

echo ""
echo "所有测试完成，正在生成 Markdown 表格..."

# 使用 Python 生成 Markdown 表格
python3 << EOF
import json
import pandas as pd
from datetime import datetime

# 读取 JSON 结果
with open("$TEMP_JSON_FILE", "r") as f:
    results = json.load(f)

# 创建 DataFrame
data = []
for r in results:
    row = {
        "Model": r["model"],
        "Total Mean (ms)": f"{r['total_mean_ms']:.3f}",
        "Total Std (ms)": f"{r['total_std_ms']:.3f}",
        "Forward Mean (ms)": f"{r['fwd_mean_ms']:.3f}",
        "Forward Std (ms)": f"{r['fwd_std_ms']:.3f}",
    }
    if r.get("bwd_mean_ms") is not None:
        row["Backward Mean (ms)"] = f"{r['bwd_mean_ms']:.3f}"
        row["Backward Std (ms)"] = f"{r['bwd_std_ms']:.3f}"
    data.append(row)

df = pd.DataFrame(data)

# 生成 Markdown 文件
with open("$OUTPUT_FILE", "w") as f:
    f.write("# Transformer 模型性能基准测试结果\n\n")
    f.write(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("## 测试配置\n\n")
    f.write(f"- **Batch Size**: $BATCH_SIZE\n")
    f.write(f"- **Context Length**: $CONTEXT_LENGTH\n")
    f.write(f"- **Warmup Steps**: $WARMUP_STEPS\n")
    f.write(f"- **Measured Steps**: $STEPS\n")
    f.write(f"- **Dtype**: $DTYPE\n")
    f.write(f"- **Mode**: $MODE\n\n")
    f.write("## 测试结果\n\n")
    f.write(df.to_markdown(index=False))
    f.write("\n")

# 同时输出到控制台
print("\n" + "=" * 80)
print("测试结果 (Markdown 格式)")
print("=" * 80 + "\n")
print(df.to_markdown(index=False))
print("\n")
print(f"结果已保存到: $OUTPUT_FILE")
EOF

echo ""
echo "完成！"
