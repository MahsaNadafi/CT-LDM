import torch
import torch.nn as nn

class EncoderLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_origin, x_recon, split="train"):
        rec_loss = torch.abs(x_origin.contiguous() - x_recon.contiguous()).mean(dim=[1,2,3]).mean()

        log = {
            "{}/rec_loss".format(split): rec_loss.detach().mean(),
        }

        if split != "train":
            gt, pred = [(x * 0.5 + 0.5).clamp(0, 1) for x in [x_origin, x_recon]]
            mse = (gt - pred).pow(2).mean()
            psnr = -10 * torch.log10(mse)
            log.update(
                **{"{}/psnr".format(split): psnr.detach()}
            )

        return rec_loss, log


class FrequencyAwareEncoderLoss(nn.Module):
    """Pixel-fidelity loss with explicit high-frequency reconstruction terms."""

    def __init__(self, l1_weight=1.0, gradient_weight=0.1, laplacian_weight=0.05):
        super().__init__()
        weights = {
            "l1_weight": l1_weight,
            "gradient_weight": gradient_weight,
            "laplacian_weight": laplacian_weight,
        }
        for name, value in weights.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        self.l1_weight = l1_weight
        self.gradient_weight = gradient_weight
        self.laplacian_weight = laplacian_weight

    @staticmethod
    def _gradient_loss(target, prediction):
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        prediction_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        prediction_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
        return 0.5 * (
            torch.abs(target_dx - prediction_dx).mean()
            + torch.abs(target_dy - prediction_dy).mean()
        )

    @staticmethod
    def _laplacian_loss(target, prediction):
        target_dxx = target[..., :, 2:] - 2 * target[..., :, 1:-1] + target[..., :, :-2]
        prediction_dxx = (
            prediction[..., :, 2:]
            - 2 * prediction[..., :, 1:-1]
            + prediction[..., :, :-2]
        )
        target_dyy = target[..., 2:, :] - 2 * target[..., 1:-1, :] + target[..., :-2, :]
        prediction_dyy = (
            prediction[..., 2:, :]
            - 2 * prediction[..., 1:-1, :]
            + prediction[..., :-2, :]
        )
        return 0.5 * (
            torch.abs(target_dxx - prediction_dxx).mean()
            + torch.abs(target_dyy - prediction_dyy).mean()
        )

    def forward(self, x_origin, x_recon, split="train"):
        target = x_origin.contiguous()
        prediction = x_recon.contiguous()
        l1_loss = torch.abs(target - prediction).mean()
        gradient_loss = self._gradient_loss(target, prediction)
        laplacian_loss = self._laplacian_loss(target, prediction)
        rec_loss = (
            self.l1_weight * l1_loss
            + self.gradient_weight * gradient_loss
            + self.laplacian_weight * laplacian_loss
        )

        log = {
            f"{split}/rec_loss": rec_loss.detach(),
            f"{split}/l1_loss": l1_loss.detach(),
            f"{split}/gradient_loss": gradient_loss.detach(),
            f"{split}/laplacian_loss": laplacian_loss.detach(),
        }

        if split != "train":
            gt, pred = [(x * 0.5 + 0.5).clamp(0, 1) for x in [target, prediction]]
            mse = (gt - pred).pow(2).mean()
            log[f"{split}/psnr"] = (-10 * torch.log10(mse)).detach()

        return rec_loss, log
