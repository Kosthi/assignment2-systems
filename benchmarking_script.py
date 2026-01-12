from __future__ import annotations

import argparse
import statistics
import timeit

import torch


def _parse_dtype(dtype: str) -> torch.dtype:
    match dtype:
        case "fp32":
            return torch.float32
        case "fp16":
            return torch.float16
        case "bf16":
            return torch.bfloat16
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")


def _device_from_arg(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _format_seconds_as_ms(x_s: float) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)

    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=3072)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--mode", type=str, default="fwd_bwd", choices=["fwd", "fwd_bwd"])

    args = parser.parse_args()

    if args.seq_len > args.context_length:
        raise ValueError("--seq-len must be <= --context-length (RoPE cache length).")

    device = _device_from_arg(args.device)
    dtype = _parse_dtype(args.dtype)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = _make_model(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device=device)

    use_autocast = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    if use_autocast:
        model = model.to(dtype=dtype)

    model.train(args.mode == "fwd_bwd")

    x = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.seq_len),
        dtype=torch.int64,
        device=device,
    )

    total_times_s: list[float] = []
    fwd_times_s: list[float] = []
    bwd_times_s: list[float] = []
    total_iters = args.warmup_steps + args.steps
    timer = timeit.default_timer

    for i in range(total_iters):
        if args.mode == "fwd_bwd":
            model.zero_grad(set_to_none=True)

        _synchronize_if_needed(device)
        t0 = timer()

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
            logits = model(x)
            loss = logits.float().mean() if args.mode == "fwd_bwd" else None

        _synchronize_if_needed(device)
        t1 = timer()

        if args.mode == "fwd_bwd":
            assert loss is not None
            loss.backward()

        _synchronize_if_needed(device)
        t2 = timer()

        if i >= args.warmup_steps:
            fwd_times_s.append(t1 - t0)
            if args.mode == "fwd_bwd":
                bwd_times_s.append(t2 - t1)
            total_times_s.append(t2 - t0)

    def _mean_std(xs: list[float]) -> tuple[float, float]:
        mean = statistics.fmean(xs) if xs else float("nan")
        std = statistics.stdev(xs) if len(xs) > 1 else 0.0
        return mean, std

    total_mean_s, total_std_s = _mean_std(total_times_s)
    fwd_mean_s, fwd_std_s = _mean_std(fwd_times_s)
    bwd_mean_s, bwd_std_s = _mean_std(bwd_times_s) if bwd_times_s else (float("nan"), float("nan"))

    print(f"device={device}")
    print(f"dtype={args.dtype}")
    print(
        "model="
        + f"layers={args.num_layers} d_model={args.d_model} heads={args.num_heads} d_ff={args.d_ff} "
        + f"ctx={args.context_length} seq={args.seq_len} vocab={args.vocab_size}"
    )
    print(f"batch_size={args.batch_size}")
    print(f"mode={args.mode}")
    print(f"warmup_steps={args.warmup_steps} measured_steps={args.steps}")
    print(f"total_mean={_format_seconds_as_ms(total_mean_s)} total_std={_format_seconds_as_ms(total_std_s)}")
    print(f"fwd_mean={_format_seconds_as_ms(fwd_mean_s)} fwd_std={_format_seconds_as_ms(fwd_std_s)}")
    if args.mode == "fwd_bwd":
        print(f"bwd_mean={_format_seconds_as_ms(bwd_mean_s)} bwd_std={_format_seconds_as_ms(bwd_std_s)}")


if __name__ == "__main__":
    main()
