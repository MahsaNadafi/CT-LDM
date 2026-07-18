"""Evaluate CT super-resolution checkpoints with a reproducible protocol."""

import argparse
from contextlib import nullcontext
import copy
import os

import numpy as np
from PIL import Image
import torch
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

from ldm.modules.evaluate.ssim import ssim
from ldm.util import instantiate_from_config


PAPER_TARGETS = {
    2.0: {"PSNR": 40.385, "SSIM": 0.959},
    4.0: {"PSNR": 33.073, "SSIM": 0.907},
    8.0: {"PSNR": 29.811, "SSIM": 0.841},
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


class Averager:
    def __init__(self):
        self.n = 0
        self.total = 0.0

    def add(self, values):
        values = values.detach().double().flatten()
        self.total += values.sum().item()
        self.n += values.numel()

    def item(self):
        return self.total / self.n if self.n else float("nan")


def calc_psnr_per_sample(sr, hr, data_range=1.0):
    mse = (sr - hr).pow(2).flatten(1).mean(dim=1)
    eps = torch.finfo(mse.dtype).eps
    result = 10 * torch.log10((data_range ** 2) / mse.clamp_min(eps))
    result[mse == 0] = float("inf")
    return result


def calc_ssim_per_sample(sr, hr):
    return ssim(sr, hr, size_average=False)


def calc_mae_per_sample(sr, hr):
    return (sr - hr).abs().flatten(1).mean(dim=1)


def infer_config_path(exp):
    exp_name = os.path.basename(os.path.normpath(exp)).split("_")[0]
    expected = os.path.join(exp, "configs", f"{exp_name}-project.yaml")
    if os.path.isfile(expected):
        return expected

    config_dir = os.path.join(exp, "configs")
    candidates = sorted(
        os.path.join(config_dir, name)
        for name in os.listdir(config_dir)
        if name.endswith("-project.yaml") and not name.startswith("-")
    )
    if not candidates:
        raise FileNotFoundError(f"No project config found under {config_dir}")
    return candidates[-1]


def load_model(config, checkpoint, device):
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    print(f"Loading model from {checkpoint}")
    config.model.params.ckpt_path = checkpoint
    ignore_keys = list(config.model.params.get("ignore_keys", []))
    if "loss_fn" not in ignore_keys:
        ignore_keys.append("loss_fn")
    config.model.params.ignore_keys = ignore_keys
    model = instantiate_from_config(config.model).to(device)
    model.eval()
    return model


def paper_comparison(scale_ratio, metrics):
    target = PAPER_TARGETS.get(float(scale_ratio))
    if target is None:
        return None
    return {
        "target_PSNR": target["PSNR"],
        "target_SSIM": target["SSIM"],
        "delta_PSNR": metrics["PSNR"] - target["PSNR"],
        "delta_SSIM": metrics["SSIM"] - target["SSIM"],
    }


@torch.no_grad()
def evaluate(args):
    if args.steps < 1 or args.steps > 1000:
        raise ValueError("--steps must be between 1 and 1000")
    if args.lr_size < 1 or args.scale_ratio <= 0:
        raise ValueError("--lr_size and --scale_ratio must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config_path = args.config or infer_config_path(args.exp)
    checkpoint = args.ckpt or os.path.join(args.exp, "checkpoints", "last.ckpt")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = OmegaConf.load(config_path)
    model = load_model(config, checkpoint, device)

    output_size = round(args.lr_size * args.scale_ratio)
    config.data.params.batch_size = args.batch_size
    config.data.params.train.params.first_k = 1
    eval_data_cfg = config.data.params.validation
    eval_data_cfg.params.size = output_size
    eval_data_cfg.params.lr_size = args.lr_size
    eval_data_cfg.params.first_k = None if args.first_k is None or args.first_k <= 0 else args.first_k

    if args.split == "test":
        config.data.params.test = OmegaConf.create(OmegaConf.to_container(eval_data_cfg, resolve=False))
        config.data.params.test.target = "ldm.data.datasets.CTSRTest"

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    loader = data._test_dataloader() if args.split == "test" else data._val_dataloader()
    print(
        f"Evaluating {len(loader.dataset)} {args.split} samples at "
        f"{args.lr_size} -> {output_size} (x{args.scale_ratio:g}) in {len(loader)} batches"
    )

    lpips_model = None
    if args.lpips:
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError("Install lpips or evaluate with --lpips false") from exc
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    psnr_result = Averager()
    ssim_result = Averager()
    mae_result = Averager()
    lpips_result = Averager()
    num_samples = 0
    image_dir = None
    if args.save_image:
        suffix = f"_{args.output_tag}" if args.output_tag else ""
        image_dir = os.path.join(args.exp, f"eval_{args.split}_imgs{suffix}")
        os.makedirs(image_dir, exist_ok=True)

    ema_context = model.ema_scope("evaluation") if args.use_ema and hasattr(model, "ema_scope") else nullcontext()
    with ema_context:
        pbar = tqdm(loader, leave=False, desc=args.split)
        for batch in pbar:
            batch = {
                key: rearrange(value.to(device), "b h w c -> b c h w")
                for key, value in batch.items()
            }
            cond = batch["image_lr"]
            batch_size = cond.shape[0]
            if hasattr(model, "get_cond"):
                cond = model.get_cond(cond)

            samples, _ = model.sample_log(
                cond=cond,
                batch_size=batch_size,
                ddim=True,
                ddim_steps=args.steps,
                eta=args.eta,
                log_every_t=20,
                print_bar="final" if args.verbose else False,
            )
            pred = (model.decode_first_stage(samples, output_size=output_size) * 0.5 + 0.5).clamp(0, 1)
            gt = (batch["image_hr"] * 0.5 + 0.5).clamp(0, 1)
            lr = (batch["image_lr"] * 0.5 + 0.5).clamp(0, 1)

            psnr_result.add(calc_psnr_per_sample(pred, gt))
            ssim_result.add(calc_ssim_per_sample(pred, gt))
            mae_result.add(calc_mae_per_sample(pred, gt))
            if lpips_model is not None:
                lpips_result.add(lpips_model(pred * 2 - 1, gt * 2 - 1).flatten())

            if image_dir:
                for offset in range(batch_size):
                    index = num_samples + offset
                    for label, image in (("pred", pred[offset]), ("gt", gt[offset]), ("lr", lr[offset])):
                        array = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
                        Image.fromarray(array).save(os.path.join(image_dir, f"{index:06d}_{label}.png"))

            num_samples += batch_size
            pbar.set_postfix(
                PSNR=f"{psnr_result.item():.4f}",
                SSIM=f"{ssim_result.item():.4f}",
                MAE=f"{mae_result.item():.6f}",
            )

    metrics = {
        "split": args.split,
        "num_samples": num_samples,
        "checkpoint": checkpoint,
        "config": config_path,
        "lr_size": args.lr_size,
        "output_size": output_size,
        "scale_ratio": float(args.scale_ratio),
        "ddim_steps": args.steps,
        "eta": args.eta,
        "seed": args.seed,
        "use_ema": bool(args.use_ema),
        "PSNR": psnr_result.item(),
        "SSIM": ssim_result.item(),
        "MAE": mae_result.item(),
    }
    if lpips_model is not None:
        metrics["LPIPS"] = lpips_result.item()
    comparison = paper_comparison(args.scale_ratio, metrics)
    if comparison:
        metrics["paper_comparison"] = comparison
    return metrics


def load_eval_config(path):
    if path is None:
        return {}
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    experiment = raw.get("experiment", {})
    evaluation = raw.get("evaluation", {})
    runtime = raw.get("runtime", {})
    output = raw.get("output", {})
    return {
        "exp": experiment.get("path"),
        "config": experiment.get("project_config"),
        "checkpoints": experiment.get("checkpoints"),
        "split": evaluation.get("split"),
        "lr_size": evaluation.get("lr_size"),
        "scale_ratio": evaluation.get("scale_ratio"),
        "first_k": evaluation.get("sample_counts"),
        "steps": evaluation.get("ddim_steps"),
        "eta": evaluation.get("eta"),
        "seed": evaluation.get("seed"),
        "batch_size": runtime.get("batch_size"),
        "device": runtime.get("device"),
        "use_ema": runtime.get("use_ema"),
        "lpips": runtime.get("lpips"),
        "verbose": output.get("verbose"),
        "save_image": output.get("save_images"),
        "save_metrics": output.get("save_metrics"),
        "metrics_dir": output.get("metrics_dir"),
        "output_tag": output.get("tag"),
    }


def configured_default(defaults, key, fallback=None):
    value = defaults.get(key)
    return fallback if value is None else value


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--eval_config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args()
    defaults = load_eval_config(config_args.eval_config)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_config", type=str, default=config_args.eval_config, help="Evaluation sweep YAML")
    parser.add_argument(
        "--exp",
        type=str,
        required=defaults.get("exp") is None,
        default=defaults.get("exp"),
        help="Experiment directory",
    )
    parser.add_argument("--config", type=str, default=defaults.get("config"), help="Override project config")
    parser.add_argument("--ckpt", type=str, default=None, help="Override checkpoint")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=defaults.get("checkpoints"),
        help="Checkpoint paths or names under EXP/checkpoints",
    )
    parser.add_argument(
        "--lr_size",
        type=int,
        required=defaults.get("lr_size") is None,
        default=defaults.get("lr_size"),
        help="Low-resolution input size",
    )
    parser.add_argument(
        "--scale_ratio",
        type=float,
        required=defaults.get("scale_ratio") is None,
        default=defaults.get("scale_ratio"),
        help="Requested upscaling ratio",
    )
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument(
        "--num_samples",
        dest="first_k",
        type=int,
        nargs="+",
        default=defaults.get("first_k"),
        help="Number of existing split samples to evaluate; <=0 means all",
    )
    sample_group.add_argument(
        "--first_k",
        dest="first_k",
        type=int,
        nargs="+",
        help="Legacy alias for --num_samples",
    )
    parser.add_argument("--batch_size", type=int, default=configured_default(defaults, "batch_size", 1))
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=configured_default(defaults, "steps", [100]),
        help="One or more DDIM step counts",
    )
    parser.add_argument("--eta", type=float, default=configured_default(defaults, "eta", 0.0))
    parser.add_argument("--seed", type=int, default=configured_default(defaults, "seed", 23))
    parser.add_argument("--device", type=str, default=defaults.get("device"), help="Example: cuda, cuda:0, or cpu")
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default=configured_default(defaults, "split", "test"),
    )
    parser.add_argument("--use_ema", type=str2bool, default=configured_default(defaults, "use_ema", True))
    parser.add_argument("--lpips", type=str2bool, default=configured_default(defaults, "lpips", False))
    parser.add_argument("--verbose", type=str2bool, default=configured_default(defaults, "verbose", False))
    parser.add_argument("--save_image", type=str2bool, default=configured_default(defaults, "save_image", False))
    parser.add_argument("--save_metrics", type=str2bool, default=configured_default(defaults, "save_metrics", True))
    parser.add_argument("--metrics_dir", type=str, default=defaults.get("metrics_dir"))
    parser.add_argument("--output_tag", type=str, default=defaults.get("output_tag"))
    return parser.parse_args()


def as_list(value, fallback):
    if value is None:
        return list(fallback)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def resolve_checkpoint(exp, checkpoint):
    if checkpoint is None:
        return os.path.join(exp, "checkpoints", "last.ckpt")
    if os.path.isabs(checkpoint) or os.path.dirname(checkpoint):
        return checkpoint
    return os.path.join(exp, "checkpoints", checkpoint)


def combination_tag(base_tag, checkpoint, steps, sample_count):
    checkpoint_tag = os.path.splitext(os.path.basename(checkpoint))[0].replace("=", "-")
    sample_tag = "all" if sample_count is None or sample_count <= 0 else str(sample_count)
    generated = f"{checkpoint_tag}_steps{steps}_samples{sample_tag}"
    return f"{base_tag}_{generated}" if base_tag else generated


def run_evaluations(args):
    configured_checkpoints = [args.ckpt] if args.ckpt else as_list(args.checkpoints, [None])
    checkpoints = [resolve_checkpoint(args.exp, value) for value in configured_checkpoints]
    steps_values = as_list(args.steps, [100])
    sample_values = as_list(args.first_k, [None])

    all_results = []
    for checkpoint in checkpoints:
        for steps in steps_values:
            for sample_count in sample_values:
                run_args = copy.copy(args)
                run_args.ckpt = checkpoint
                run_args.steps = int(steps)
                run_args.first_k = None if sample_count is None else int(sample_count)
                run_args.output_tag = combination_tag(args.output_tag, checkpoint, run_args.steps, run_args.first_k)

                results = evaluate(run_args)
                if run_args.save_metrics:
                    metrics_dir = run_args.metrics_dir or run_args.exp
                    os.makedirs(metrics_dir, exist_ok=True)
                    metrics_path = os.path.join(
                        metrics_dir,
                        f"eval_{run_args.split}_metrics_{run_args.output_tag}.yaml",
                    )
                    OmegaConf.save(config=OmegaConf.create(results), f=metrics_path)
                    print(f"Saved metrics to {metrics_path}")
                print(OmegaConf.to_yaml(OmegaConf.create(results)))
                all_results.append(results)
    return all_results


if __name__ == "__main__":
    run_evaluations(parse_args())
