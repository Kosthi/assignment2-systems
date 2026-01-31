"""
Transformer 模型前向和反向传播性能基准测试脚本

该脚本用于测量 Transformer 模型在 GPU 上的前向传播和反向传播耗时。
主要功能：
1. 根据给定超参数初始化 Transformer 模型
2. 生成随机批次的测试数据
3. 执行预热步骤后，统计多次迭代的耗时
4. 支持仅测试前向传播或同时测试前向和反向传播

注意事项：
- 使用 timeit.default_timer() 进行高精度计时
- 每次迭代后调用 torch.cuda.synchronize() 确保 GPU 操作完成
  （因为 CUDA 调用是异步的，需要同步才能准确测量时间）
"""

from __future__ import annotations

import argparse  # 命令行参数解析模块
import json
import numpy as np
import statistics  # 统计计算模块（用于计算均值和标准差）
import timeit  # 高精度计时模块
from memory_estimation import ModelConfig, MODEL_CONFIGS
import torch  # PyTorch 深度学习框架
import torch.cuda.nvtx as nvtx


def _mean_std(xs: list[float]) -> tuple[float, float]:
    """计算列表的均值和标准差"""
    mean = np.mean(xs)
    std = np.std(xs, ddof=1)
    return mean, std


def _parse_dtype(dtype: str) -> torch.dtype:
    """
    将字符串类型的精度名称转换为 PyTorch 的 dtype 对象

    Args:
        dtype: 精度字符串，支持 "fp32", "fp16", "bf16"

    Returns:
        对应的 torch.dtype 对象

    Raises:
        ValueError: 不支持的精度类型
    """
    match dtype:
        case "fp32":
            return torch.float32  # 32位单精度浮点
        case "fp16":
            return torch.float16  # 16位半精度浮点
        case "bf16":
            return torch.bfloat16  # 16位 bfloat 格式（更大的动态范围）
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def _device_from_arg(device: str) -> torch.device:
    """
    根据参数确定运行设备

    Args:
        device: 设备字符串，"auto" 表示自动选择，或指定 "cuda"/"cpu"/"mps"

    Returns:
        torch.device 对象
    """
    if device == "auto":
        # 自动选择设备：优先 CUDA GPU，其次 Apple MPS，最后 CPU
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _synchronize_if_needed(device: torch.device) -> None:
    """
    如果是 CUDA 设备，执行同步操作

    重要说明：
    CUDA 调用是异步的 —— 当调用 CUDA 内核时，函数会立即返回控制权给 CPU，
    而不会等待 GPU 计算完成。因此需要调用 synchronize() 来等待所有 GPU
    内核执行完毕，才能准确测量 CUDA 内核的运行时间。

    Args:
        device: 当前使用的设备
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)  # 等待 GPU 上所有操作完成


def _format_seconds_as_ms(x_s: float) -> str:
    """
    将秒转换为毫秒格式的字符串

    Args:
        x_s: 以秒为单位的时间

    Returns:
        格式化的毫秒字符串，保留3位小数
    """
    return f"{x_s * 1e3:.3f} ms"


def _make_model(
    *,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
):
    """
    根据给定超参数创建 Transformer 语言模型

    Args:
        vocab_size: 词汇表大小
        context_length: 上下文长度（最大序列长度）
        d_model: 模型隐藏层维度
        num_layers: Transformer 层数
        num_heads: 注意力头数
        d_ff: 前馈网络中间层维度
        rope_theta: RoPE 位置编码的 theta 参数

    Returns:
        初始化的 BasicsTransformerLM 模型实例
    """
    # 延迟导入模型类，避免在不需要时加载整个模块
    from cs336_basics.model import BasicsTransformerLM

    return BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )


def init_model(model_config, vocab_size, context_length, mode, dtype, device):
    """模型初始化"""
    model = _make_model(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=model_config.d_model,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        d_ff=model_config.d_ff,
        rope_theta=model_config.rope_theta,
    ).to(device=device)  # 将模型移到目标设备

    # 如果使用混合精度训练，将模型转换为对应精度
    use_autocast = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    if use_autocast:
        model = model.to(dtype=dtype)

    # 设置模型为训练模式（fwd_bwd）或评估模式（fwd）
    model.train(mode == "fwd_bwd")

    return model


def benchmark(model, x, mode, warmup_steps, steps, device, dtype, use_autocast=False):
    """性能测量"""
    # 用于存储各阶段耗时的列表
    total_times_s: list[float] = []  # 总耗时（前向+反向）
    fwd_times_s: list[float] = []  # 前向传播耗时
    bwd_times_s: list[float] = []  # 反向传播耗时

    total_iters = warmup_steps + steps  # 总迭代次数 = 预热 + 正式测量
    timer = timeit.default_timer  # 使用系统最高精度时钟（比 time.time() 更精确）

    optimizer = torch.optim.AdamW(model.parameters())

    for i in range(total_iters):
        # 前向传播
        with nvtx.range("fwd"):
            t0 = timer()  # 记录前向开始时间

            # 使用自动混合精度进行前向传播
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
                logits = model(x)  # 前向传播，获取 logits
            # 如果需要反向传播，计算一个简单的损失（logits 均值）
            loss = logits.float().mean() if mode == "fwd_bwd" else None
            # 同步 GPU，确保前向传播完成
            _synchronize_if_needed(device)
            t1 = timer()  # 记录前向结束时间

        # 反向传播
        if mode == "fwd_bwd":
            # 梯度裁剪
            with nvtx.range("clip_grad_norm_"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                _synchronize_if_needed(device)
            # 优化器步进
            with nvtx.range("optimizer_step"):
                optimizer.step()
                _synchronize_if_needed(device)
            # 反向传播计算梯度
            with nvtx.range("bwd"):
                loss.backward()
                _synchronize_if_needed(device)
            # 清除梯度
            with nvtx.range("zero_grad"):
                model.zero_grad(set_to_none=True)
                _synchronize_if_needed(device)

        t2 = timer()  # 记录反向结束时间

        # 记录耗时（仅在预热后）
        if i >= warmup_steps:
            # 预热结束后才开始记录正式测量的耗时
            fwd_times_s.append(t1 - t0)  # 前向耗时
            if mode == "fwd_bwd":
                bwd_times_s.append(t2 - t1)  # 反向耗时
            total_times_s.append(t2 - t0)  # 总耗时

    return total_times_s, fwd_times_s, bwd_times_s


def main() -> None:
    """
    主函数：执行端到端的性能基准测试

    流程：
    1. 解析命令行参数
    2. 初始化模型和数据
    3. 执行预热步骤
    4. 执行正式测量步骤
    5. 输出统计结果
    """
    # 命令行参数解析
    parser = argparse.ArgumentParser()

    # 模型架构参数
    parser.add_argument("--model", type=str, default="small", choices=MODEL_CONFIGS.keys())  # 模型配置
    parser.add_argument("--vocab_size", type=int, default=10000)  # 词汇表大小
    parser.add_argument("--context_length", type=int, default=128)  # 上下文长度
    parser.add_argument("--rope_theta", type=float, default=10000.0)  # RoPE 位置编码的 theta 参数
    parser.add_argument("--batch_size", type=int, default=4)  # 批处理大小

    # 运行环境参数
    parser.add_argument("--device", type=str, default="auto")  # 运行设备
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",  # 数据精度
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument("--seed", type=int, default=42)  # 随机种子

    # 基准测试参数
    parser.add_argument("--warmup-steps", type=int, default=5)  # 预热步数（不计入计时）
    parser.add_argument("--steps", type=int, default=10)  # 正式测量步数
    parser.add_argument(
        "--mode",
        type=str,
        default="fwd_bwd",  # 测试模式
        choices=["fwd", "fwd_bwd"],
    )  # fwd: 仅前向, fwd_bwd: 前向+反向

    args = parser.parse_args()

    device = _device_from_arg(args.device)  # 确定运行设备
    dtype = _parse_dtype(args.dtype)  # 解析数据精度

    # 设置随机种子以确保可重复性
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)  # 为所有 GPU 设置种子

    # 初始化模型
    model_config = MODEL_CONFIGS[args.model]
    model = init_model(model_config, args.vocab_size, args.context_length, args.mode, dtype, device)

    # 生成随机 token ID 作为输入（范围 [0, vocab_size)）
    x = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),  # [batch_size, seq_len]
        dtype=torch.int64,
        device=device,
    )

    # 基准测试
    total_times_s, fwd_times_s, bwd_times_s = benchmark(
        model, x, args.mode, args.warmup_steps, args.steps, device, dtype
    )

    # 计算各阶段的均值和标准差
    total_mean_s, total_std_s = _mean_std(total_times_s)
    fwd_mean_s, fwd_std_s = _mean_std(fwd_times_s)
    bwd_mean_s, bwd_std_s = _mean_std(bwd_times_s) if bwd_times_s else (float("nan"), float("nan"))

    # JSON格式输出，便于脚本解析
    result = {
        "model": args.model,
        "device": str(device),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "mode": args.mode,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.steps,
        "total_mean_ms": total_mean_s * 1e3,
        "total_std_ms": total_std_s * 1e3,
        "fwd_mean_ms": fwd_mean_s * 1e3,
        "fwd_std_ms": fwd_std_s * 1e3,
        "bwd_mean_ms": bwd_mean_s * 1e3 if args.mode == "fwd_bwd" else None,
        "bwd_std_ms": bwd_std_s * 1e3 if args.mode == "fwd_bwd" else None,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
