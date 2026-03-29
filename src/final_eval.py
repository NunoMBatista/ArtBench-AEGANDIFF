"""
final_eval.py — Multi-seed final evaluation for ArtBench generative models.

Loads a trained checkpoint and runs the evaluation protocol (5000 generated vs
5000 real images, FID + KID) N times with different random seeds, then reports
mean ± std across seeds as required by the project protocol.

Usage:
    python -m src.final_eval <model_type> --checkpoint <path/to/model.pt> [--seeds 0,1,2,3,4,5,6,7,8,9]
    python -m src.final_eval diffusion --checkpoint outputs/run_diffusion_XYZ/model.pt
    python -m src.final_eval vae  # uses latest checkpoint found in outputs/
"""

import argparse
import os
import statistics
import sys

import yaml

from globals import ensure_repo_root
ensure_repo_root()

from src.evaluate import (
    EvalConfig,
    _find_latest_checkpoint,
    _init_wandb_eval,
    _load_model,
    evaluate,
)

try:
    import wandb
except ImportError:
    wandb = None


def _parse_args():
    from src.evaluate import _MODEL_REGISTRY
    parser = argparse.ArgumentParser(
        description="Run multi-seed final evaluation (FID / KID mean ± std)."
    )
    parser.add_argument(
        "model_type",
        type=str,
        choices=list(_MODEL_REGISTRY.keys()),
        help="Model architecture (e.g. vae, dcgan, diffusion).",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="",
        help="Path to model.pt. If omitted, the latest checkpoint is used.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9",
        help="Comma-separated list of seeds (default: 0..9).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="Generated and real images per seed (default: 5000).",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default="",
        help="W&B group name for all seed runs. Defaults to 'final_eval_<model_type>'.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    model_type = args.model_type.lower()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    ckpt_path = args.checkpoint or _find_latest_checkpoint(model_type)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Seeds: {seeds}")
    print(f"Samples per seed: {args.num_samples}")

    config = EvalConfig()
    config.num_samples = args.num_samples
    config.update(ckpt_path)

    import torch
    device = torch.device(config.device)

    # Build extra kwargs for model construction
    extra_kwargs = {}
    if model_type == "dcgan":
        extra_kwargs["use_spectral_norm"] = config.use_spectral_norm
    elif model_type == "cgan":
        extra_kwargs["num_classes"] = config.num_classes
        extra_kwargs["use_spectral_norm"] = config.use_spectral_norm
    elif model_type == "diffusion":
        extra_kwargs.update({
            "image_size": config.image_size,
            "img_channels": config.img_channels,
            "num_classes": config.num_classes,
            "num_diffusion_steps": config.num_diffusion_steps,
            "cfg_dropout": config.cfg_dropout,
            "sample_steps": config.sample_steps,
            "guidance_scale": config.guidance_scale,
            "class_conditional": config.class_conditional,
            "use_attention": config.use_attention,
        })
    elif model_type == "google_ddpm":
        extra_kwargs.update({
            "pretrained_model_id": config.pretrained_model_id,
            "num_diffusion_steps": config.num_diffusion_steps,
            "sample_steps": config.sample_steps,
            "disable_attention_on_cpu": config.disable_attention_on_cpu,
        })

    model = _load_model(
        model_type, ckpt_path, config.latent_dim, config.base_channels, device, **extra_kwargs
    )

    def sampler(num_samples, dev):
        return model.sample(num_samples, device=dev)

    wandb_group = args.wandb_group or f"final_eval_{model_type}"
    wandb_cfg = config.wandb or {}

    fids, kid_means, kid_stds = [], [], []

    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] seed={seed}")
        run_config = EvalConfig()
        run_config.__dict__.update(config.__dict__)
        run_config.seed = seed

        # Optional per-seed wandb run
        wandb_run = None
        if wandb is not None and wandb_cfg.get("enabled", False):
            try:
                entity = wandb_cfg.get("entity") or os.getenv("WANDB_ENTITY")
                project = wandb_cfg.get("project") or os.getenv("WANDB_PROJECT") or "ArtBench-AEGANDIFF"
                run_name = f"final_eval_{model_type}_seed{seed}"
                wandb_run = wandb.init(
                    project=project,
                    entity=entity,
                    name=run_name,
                    group=wandb_group,
                    config={
                        "model_type": model_type,
                        "checkpoint_path": ckpt_path,
                        "seed": seed,
                        "num_samples": args.num_samples,
                    },
                    tags=wandb_cfg.get("tags", []) + ["final_eval", model_type],
                    reinit=True,
                )
            except Exception as e:
                print(f"WARNING: wandb init failed ({e}), continuing without wandb for this seed.")

        fid, kid_mean, kid_std = evaluate(run_config, sampler)
        fids.append(fid)
        kid_means.append(kid_mean)
        kid_stds.append(kid_std)
        print(f"  FID={fid:.4f}  KID_mean={kid_mean:.6f}  KID_std={kid_std:.6f}")

        if wandb_run is not None:
            try:
                wandb_run.log({"fid": fid, "kid_mean": kid_mean, "kid_std": kid_std, "seed": seed})
            finally:
                wandb_run.finish()

    # Aggregate
    mean_fid = statistics.mean(fids)
    std_fid = statistics.stdev(fids) if len(fids) > 1 else 0.0
    mean_kid = statistics.mean(kid_means)
    std_kid = statistics.stdev(kid_means) if len(kid_means) > 1 else 0.0

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS  ({len(seeds)} seeds: {seeds})")
    print(f"  FID : {mean_fid:.4f} ± {std_fid:.4f}")
    print(f"  KID : {mean_kid:.6f} ± {std_kid:.6f}")
    print("=" * 60)

    summary = {
        "model_type": model_type,
        "checkpoint": ckpt_path,
        "seeds": seeds,
        "num_samples": args.num_samples,
        "fid_mean": mean_fid,
        "fid_std": std_fid,
        "kid_mean": mean_kid,
        "kid_std": std_kid,
        "fid_per_seed": fids,
        "kid_per_seed": kid_means,
    }

    # Save summary next to checkpoint
    summary_path = os.path.join(os.path.dirname(ckpt_path), "final_eval_summary.yml")
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(f"Summary saved to {summary_path}")

    # Log aggregate summary run to wandb
    if wandb is not None and wandb_cfg.get("enabled", False):
        try:
            entity = wandb_cfg.get("entity") or os.getenv("WANDB_ENTITY")
            project = wandb_cfg.get("project") or os.getenv("WANDB_PROJECT") or "ArtBench-AEGANDIFF"
            summary_run = wandb.init(
                project=project,
                entity=entity,
                name=f"final_eval_{model_type}_summary",
                group=wandb_group,
                config=summary,
                tags=wandb_cfg.get("tags", []) + ["final_eval", "summary", model_type],
                reinit=True,
            )
            summary_run.log({
                "fid_mean": mean_fid,
                "fid_std": std_fid,
                "kid_mean": mean_kid,
                "kid_std": std_kid,
            })
            summary_run.finish()
        except Exception as e:
            print(f"WARNING: wandb summary run failed ({e}).")


if __name__ == "__main__":
    main()
