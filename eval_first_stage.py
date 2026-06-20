import argparse
import os
import time

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ldm.util import instantiate_from_config
from ldm.modules.evaluate.ssim import ssim


class Averager:
    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, value, n=1.0):
        self.v = (self.v * self.n + value * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


def infer_config_from_exp(exp):
    exp_name = os.path.basename(os.path.normpath(exp)).split("_")[0]
    return os.path.join(exp, "configs", f"{exp_name}-project.yaml")


def calc_psnr(pred, gt):
    mse = (pred - gt).pow(2).mean()
    if mse.item() == 0:
        return torch.tensor(float("inf"), device=pred.device)
    return -10 * torch.log10(mse)


def move_fconfig_to_device(fconfig, device):
    moved = {}
    for key, value in fconfig.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


@torch.no_grad()
def evaluate(args):
    if args.exp:
        config_path = args.config or infer_config_from_exp(args.exp)
        ckpt_path = args.ckpt or os.path.join(args.exp, "checkpoints", "last.ckpt")
        output_dir = args.exp
    else:
        if not args.config or not args.ckpt:
            raise ValueError("Use --exp, or pass both --config and --ckpt.")
        config_path = args.config
        ckpt_path = args.ckpt
        output_dir = os.path.dirname(os.path.abspath(ckpt_path))

    config = OmegaConf.load(config_path)
    config.model.params.ckpt_path = ckpt_path

    if args.batch_size is not None:
        config.data.params.batch_size = args.batch_size

    eval_data_cfg = config.data.params.validation
    if args.first_k is not None:
        eval_data_cfg.params.first_k = args.first_k
    if args.size is not None:
        eval_data_cfg.params.size = args.size

    if args.split == "test":
        config.data.params.test = OmegaConf.create(OmegaConf.to_container(eval_data_cfg, resolve=False))
        config.data.params.test.target = "ldm.data.datasets.CTSRTest"

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = instantiate_from_config(config.model).to(device)
    model.eval()

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    loader = data._test_dataloader() if args.split == "test" else data._val_dataloader()

    psnr_res = Averager()
    ssim_res = Averager()
    mae_res = Averager()
    rec_loss_res = Averager()
    num_samples = 0

    pbar = tqdm(loader, desc=f"first-stage {args.split}", leave=False)
    for batch in pbar:
        inp, gt, fconfig = model.get_input(batch, model.valconfig)
        inp = inp.to(device)
        gt = gt.to(device)
        fconfig = move_fconfig_to_device(fconfig, device)

        pred, _ = model(inp, sample_posterior=False, **fconfig)
        rec_loss, log_dict = model.loss(gt, pred, split="val")

        pred = (pred * 0.5 + 0.5).clamp(0, 1)
        gt = (gt * 0.5 + 0.5).clamp(0, 1)

        b_size = pred.shape[0]
        num_samples += b_size

        psnr_res.add(calc_psnr(pred, gt).item(), b_size)
        ssim_res.add(ssim(pred, gt).item(), b_size)
        mae_res.add(torch.mean(torch.abs(pred - gt)).item(), b_size)
        rec_loss_res.add(rec_loss.item(), b_size)

        pbar.set_description(
            f"samples: {num_samples}, psnr: {psnr_res.item():.4f}, "
            f"ssim: {ssim_res.item():.4f}, mae: {mae_res.item():.6f}"
        )

    metrics = {
        "split": args.split,
        "num_samples": num_samples,
        "checkpoint": ckpt_path,
        "PSNR": psnr_res.item(),
        "SSIM": ssim_res.item(),
        "MAE": mae_res.item(),
        "rec_loss": rec_loss_res.item(),
    }

    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    metrics_path = args.output or os.path.join(output_dir, f"first_stage_eval_metrics_{timestamp}.yaml")
    os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(metrics), f=metrics_path)

    print(f"Saved metrics to {metrics_path}")
    print(OmegaConf.to_yaml(OmegaConf.create(metrics)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default=None, help="Experiment directory under logs/")
    parser.add_argument("--config", type=str, default=None, help="First-stage config yaml")
    parser.add_argument("--ckpt", type=str, default=None, help="First-stage checkpoint")
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "test"])
    parser.add_argument("--first_k", type=int, default=None, help="Limit number of evaluated samples")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--size", type=int, default=None, help="Override dataset image size")
    parser.add_argument("--device", type=str, default=None, help="Example: cuda, cuda:0, or cpu")
    parser.add_argument("--output", type=str, default=None, help="Metrics yaml output path")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
