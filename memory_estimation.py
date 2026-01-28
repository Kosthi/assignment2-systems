#!/usr/bin/env python3
"""
GPU Memory Estimation Script for BasicsTransformerLM

This script estimates and measures the GPU memory required to run different
sizes of the Transformer language model for both inference and training.
"""

import gc
import sys
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

# Add the cs336-basics package to path
sys.path.insert(0, "/root/assignment2-systems/cs336-basics")

from cs336_basics.model import BasicsTransformerLM


@dataclass
class ModelConfig:
    """Configuration for a Transformer model."""

    name: str
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    vocab_size: int = 10000  # GPT-2 vocabulary size
    context_length: int = 2048
    rope_theta: float = 10000.0


# Model configurations from the user's table
MODEL_CONFIGS = [
    ModelConfig(name="small", d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    ModelConfig(name="medium", d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    ModelConfig(name="large", d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    ModelConfig(name="xl", d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    ModelConfig(name="2.7B", d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
]


def count_parameters(model: nn.Module) -> int:
    """Count the total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def get_parameter_breakdown(config: ModelConfig) -> dict:
    """Calculate the theoretical parameter count for each component."""
    d_model = config.d_model
    d_ff = config.d_ff
    num_layers = config.num_layers
    vocab_size = config.vocab_size

    # Per-layer parameters
    attention_params = 4 * d_model * d_model  # Q, K, V, O projections
    ffn_params = 3 * d_model * d_ff  # w1, w2, w3 in SwiGLU
    rmsnorm_params = 2 * d_model  # ln1, ln2
    layer_params = attention_params + ffn_params + rmsnorm_params

    # Total layer parameters
    total_layer_params = layer_params * num_layers

    # Embedding and output
    embedding_params = vocab_size * d_model
    lm_head_params = d_model * vocab_size  # Note: no bias in Linear
    final_ln_params = d_model

    return {
        "attention_per_layer": attention_params,
        "ffn_per_layer": ffn_params,
        "rmsnorm_per_layer": rmsnorm_params,
        "total_per_layer": layer_params,
        "all_layers": total_layer_params,
        "embedding": embedding_params,
        "lm_head": lm_head_params,
        "final_ln": final_ln_params,
        "total": total_layer_params + embedding_params + lm_head_params + final_ln_params,
    }


def bytes_to_gb(bytes_val: int) -> float:
    """Convert bytes to gigabytes."""
    return bytes_val / (1024**3)


def bytes_to_mb(bytes_val: int) -> float:
    """Convert bytes to megabytes."""
    return bytes_val / (1024**2)


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def estimate_memory_theoretical(config: ModelConfig, dtype: torch.dtype = torch.float32) -> dict:
    """
    Estimate memory requirements theoretically (without actually creating the model).

    Returns memory estimates in bytes.
    """
    bytes_per_param = 4 if dtype == torch.float32 else 2

    breakdown = get_parameter_breakdown(config)
    total_params = breakdown["total"]

    # Model weights memory
    model_memory = total_params * bytes_per_param

    # Gradient memory (same size as model)
    gradient_memory = model_memory

    # Optimizer state memory (Adam: 2x for momentum and variance, in FP32)
    optimizer_memory = total_params * 4 * 2  # Always FP32 for optimizer states

    # Activation memory estimation (rough, depends on batch size and seq length)
    # For a rough estimate: batch_size * seq_len * d_model * num_layers * factor
    # This is highly variable; we'll estimate for batch=1, seq=512
    batch_size = 4
    seq_len = 512

    # Activations per layer (rough estimate):
    # - Input to attention: batch * seq * d_model
    # - Q, K, V: 3 * batch * seq * d_model
    # - Attention scores: batch * num_heads * seq * seq
    # - Attention output: batch * seq * d_model
    # - FFN intermediate: batch * seq * d_ff * 2 (for SwiGLU gate)
    # - FFN output: batch * seq * d_model
    d_model = config.d_model
    d_ff = config.d_ff
    num_heads = config.num_heads

    activation_per_layer = (
        batch_size * seq_len * d_model * 4  # input, Q, K, V combined effect
        + batch_size * num_heads * seq_len * seq_len  # attention matrix
        + batch_size * seq_len * d_ff * 2  # FFN intermediate
    ) * bytes_per_param

    total_activation_memory = activation_per_layer * config.num_layers

    # KV Cache for inference (batch * layers * 2 * seq_len * d_model)
    kv_cache_memory = batch_size * config.num_layers * 2 * seq_len * d_model * bytes_per_param

    return {
        "params": total_params,
        "model_memory": model_memory,
        "gradient_memory": gradient_memory,
        "optimizer_memory": optimizer_memory,
        "activation_memory": total_activation_memory,
        "kv_cache_memory": kv_cache_memory,
        "inference_total": model_memory + kv_cache_memory + total_activation_memory // config.num_layers,
        "training_total": model_memory + gradient_memory + optimizer_memory + total_activation_memory,
    }


def measure_memory_actual(
    config: ModelConfig,
    dtype: torch.dtype = torch.float16,
    batch_size: int = 1,
    seq_len: int = 512,
    measure_training: bool = True,
) -> Optional[dict]:
    """
    Actually create the model and measure GPU memory usage.

    Returns None if CUDA is not available or if the model doesn't fit in memory.
    """
    if not torch.cuda.is_available():
        return None

    device = torch.device("cuda")
    clear_gpu_memory()

    results = {}

    try:
        # Record baseline memory
        baseline_memory = torch.cuda.memory_allocated()

        # Create model
        model = BasicsTransformerLM(
            vocab_size=config.vocab_size,
            context_length=seq_len,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            rope_theta=config.rope_theta,
        ).to(device=device, dtype=dtype)

        model_memory = torch.cuda.memory_allocated() - baseline_memory
        results["model_memory"] = model_memory
        results["params"] = count_parameters(model)

        # Measure inference memory
        model.eval()
        clear_gpu_memory()
        torch.cuda.reset_peak_memory_stats()

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

        with torch.no_grad():
            _ = model(input_ids)

        torch.cuda.synchronize()
        inference_peak = torch.cuda.max_memory_allocated()
        results["inference_peak"] = inference_peak
        results["inference_activation"] = inference_peak - model_memory

        if measure_training:
            # Measure training memory
            model.train()
            clear_gpu_memory()
            torch.cuda.reset_peak_memory_stats()

            # Create optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            optimizer_memory_after = torch.cuda.memory_allocated()

            # Forward pass
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
            logits = model(input_ids)

            # Compute loss
            targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
            loss = nn.functional.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.synchronize()
            training_peak = torch.cuda.max_memory_allocated()
            results["training_peak"] = training_peak

        # Cleanup
        del model
        if measure_training:
            del optimizer
        clear_gpu_memory()

        return results

    except torch.cuda.OutOfMemoryError as e:
        clear_gpu_memory()
        return {"error": f"OOM: {str(e)}"}
    except Exception as e:
        clear_gpu_memory()
        return {"error": str(e)}


def print_theoretical_estimates():
    """Print theoretical memory estimates for all model configurations."""
    print("=" * 100)
    print("THEORETICAL MEMORY ESTIMATION (Batch=4, Seq=512)")
    print("=" * 100)

    # FP32 estimates
    print("\n### FP32 (float32) ###\n")
    print(
        f"{'Model':<8} {'Params':>12} {'Model':>12} {'Gradients':>12} {'Optimizer':>12} {'Activations':>12} {'Inference':>12} {'Training':>12}"
    )
    print(f"{'':8} {'':>12} {'Memory':>12} {'Memory':>12} {'(Adam)':>12} {'Memory':>12} {'Total':>12} {'Total':>12}")
    print("-" * 100)

    for config in MODEL_CONFIGS:
        est = estimate_memory_theoretical(config, torch.float32)
        print(
            f"{config.name:<8} "
            f"{est['params'] / 1e6:>10.1f}M "
            f"{bytes_to_gb(est['model_memory']):>10.2f}GB "
            f"{bytes_to_gb(est['gradient_memory']):>10.2f}GB "
            f"{bytes_to_gb(est['optimizer_memory']):>10.2f}GB "
            f"{bytes_to_gb(est['activation_memory']):>10.2f}GB "
            f"{bytes_to_gb(est['inference_total']):>10.2f}GB "
            f"{bytes_to_gb(est['training_total']):>10.2f}GB"
        )

    # FP16/BF16 estimates
    print("\n### FP16/BF16 (float16/bfloat16) ###\n")
    print(
        f"{'Model':<8} {'Params':>12} {'Model':>12} {'Gradients':>12} {'Optimizer':>12} {'Activations':>12} {'Inference':>12} {'Training':>12}"
    )
    print(
        f"{'':8} {'':>12} {'Memory':>12} {'Memory':>12} {'(Adam FP32)':>12} {'Memory':>12} {'Total':>12} {'Total':>12}"
    )
    print("-" * 100)

    for config in MODEL_CONFIGS:
        est = estimate_memory_theoretical(config, torch.float16)
        print(
            f"{config.name:<8} "
            f"{est['params'] / 1e6:>10.1f}M "
            f"{bytes_to_gb(est['model_memory']):>10.2f}GB "
            f"{bytes_to_gb(est['gradient_memory']):>10.2f}GB "
            f"{bytes_to_gb(est['optimizer_memory']):>10.2f}GB "
            f"{bytes_to_gb(est['activation_memory']):>10.2f}GB "
            f"{bytes_to_gb(est['inference_total']):>10.2f}GB "
            f"{bytes_to_gb(est['training_total']):>10.2f}GB"
        )


def print_parameter_breakdown():
    """Print detailed parameter breakdown for each model."""
    print("\n" + "=" * 100)
    print("PARAMETER BREAKDOWN BY COMPONENT")
    print("=" * 100 + "\n")

    for config in MODEL_CONFIGS:
        breakdown = get_parameter_breakdown(config)
        print(
            f"### {config.name.upper()} (d_model={config.d_model}, d_ff={config.d_ff}, layers={config.num_layers}, heads={config.num_heads}) ###"
        )
        print(
            f"  Attention per layer:  {breakdown['attention_per_layer']:>15,} params ({bytes_to_mb(breakdown['attention_per_layer'] * 4):>8.2f} MB FP32)"
        )
        print(
            f"  FFN per layer:        {breakdown['ffn_per_layer']:>15,} params ({bytes_to_mb(breakdown['ffn_per_layer'] * 4):>8.2f} MB FP32)"
        )
        print(
            f"  RMSNorm per layer:    {breakdown['rmsnorm_per_layer']:>15,} params ({bytes_to_mb(breakdown['rmsnorm_per_layer'] * 4):>8.2f} MB FP32)"
        )
        print(
            f"  Total per layer:      {breakdown['total_per_layer']:>15,} params ({bytes_to_mb(breakdown['total_per_layer'] * 4):>8.2f} MB FP32)"
        )
        print(
            f"  All layers ({config.num_layers:>2}):      {breakdown['all_layers']:>15,} params ({bytes_to_gb(breakdown['all_layers'] * 4):>8.2f} GB FP32)"
        )
        print(
            f"  Token Embedding:      {breakdown['embedding']:>15,} params ({bytes_to_mb(breakdown['embedding'] * 4):>8.2f} MB FP32)"
        )
        print(
            f"  LM Head:              {breakdown['lm_head']:>15,} params ({bytes_to_mb(breakdown['lm_head'] * 4):>8.2f} MB FP32)"
        )
        print(
            f"  Final LayerNorm:      {breakdown['final_ln']:>15,} params ({bytes_to_mb(breakdown['final_ln'] * 4):>8.2f} MB FP32)"
        )
        print(f"  ----------------------------------------")
        print(
            f"  TOTAL:                {breakdown['total']:>15,} params ({bytes_to_gb(breakdown['total'] * 4):>8.2f} GB FP32)"
        )
        print()


def print_actual_measurements(batch_size: int = 1, seq_len: int = 512):
    """Measure and print actual GPU memory usage."""
    if not torch.cuda.is_available():
        print("\n[WARNING] CUDA is not available. Skipping actual measurements.\n")
        return

    print("\n" + "=" * 100)
    print(f"ACTUAL GPU MEMORY MEASUREMENTS (Batch={batch_size}, Seq={seq_len})")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {bytes_to_gb(torch.cuda.get_device_properties(0).total_memory):.2f} GB")
    print("=" * 100 + "\n")

    # Test with FP16
    print("### FP32 (float32) Measurements ###\n")
    print(f"{'Model':<8} {'Params':>12} {'Model':>12} {'Inference':>12} {'Inference':>12} {'Training':>12}")
    print(f"{'':8} {'':>12} {'Memory':>12} {'Activation':>12} {'Peak':>12} {'Peak':>12}")
    print("-" * 80)

    for config in MODEL_CONFIGS:
        results = measure_memory_actual(
            config,
            dtype=torch.float32,
            batch_size=batch_size,
            seq_len=seq_len,
            measure_training=True,
        )

        if results is None:
            print(f"{config.name:<8} {'CUDA not available':>60}")
        elif "error" in results:
            print(f"{config.name:<8} {results['error']:>60}")
        else:
            training_peak = results.get("training_peak", 0)
            training_str = f"{bytes_to_gb(training_peak):>10.2f}GB" if training_peak else "N/A"
            print(
                f"{config.name:<8} "
                f"{results['params'] / 1e6:>10.1f}M "
                f"{bytes_to_gb(results['model_memory']):>10.2f}GB "
                f"{bytes_to_gb(results['inference_activation']):>10.2f}GB "
                f"{bytes_to_gb(results['inference_peak']):>10.2f}GB "
                f"{training_str}"
            )


def print_gpu_recommendations():
    """Print GPU recommendations for each model size."""
    print("\n" + "=" * 100)
    print("GPU RECOMMENDATIONS")
    print("=" * 100 + "\n")

    recommendations = [
        ("small", "RTX 3060 12GB / RTX 4060 8GB", "RTX 3090 24GB / RTX 4090 24GB"),
        ("medium", "RTX 3090 24GB / RTX 4090 24GB", "A100 40GB / 2x RTX 4090"),
        ("large", "A100 40GB / 2x RTX 4090", "A100 80GB / 2x A100 40GB"),
        ("xl", "A100 80GB", "2x A100 80GB / 4x A100 40GB"),
        ("2.7B", "A100 80GB / H100 80GB", "4x A100 80GB / 2x H100 80GB"),
    ]

    print(f"{'Model':<8} {'Inference (FP16)':<35} {'Training (FP16 + AdamW)':<40}")
    print("-" * 85)
    for name, inference, training in recommendations:
        print(f"{name:<8} {inference:<35} {training:<40}")

    print("\n### Notes ###")
    print("- Inference estimates assume batch_size=1, seq_len=512")
    print("- Training estimates include model, gradients, optimizer states (Adam), and activations")
    print("- Actual memory may vary based on PyTorch version, CUDA version, and other factors")
    print("- Gradient checkpointing can reduce training memory by ~50-70% at the cost of ~30% slower training")
    print("- Mixed precision training (AMP) can further reduce memory while maintaining quality")


def main():
    """Main function to run all estimations."""
    print("\n" + "#" * 100)
    print("# GPU MEMORY ESTIMATION FOR BasicsTransformerLM")
    print("#" * 100)

    # Print parameter breakdown
    print_parameter_breakdown()

    # Print theoretical estimates
    print_theoretical_estimates()

    # Print actual measurements if GPU is available
    print_actual_measurements(batch_size=4, seq_len=512)

    # Print GPU recommendations
    print_gpu_recommendations()


if __name__ == "__main__":
    main()
