#!/usr/bin/env python3
"""Use SimpleNet's feature extractor as encoder, train a decoder to reconstruct images,
and save latent traversal grids showing how reconstructions change when varying latent dims.

This script does NOT modify SimpleNet internals; it freezes the feature extractor and
trains a lightweight decoder that maps SimpleNet's aggregated features back to image space.

Example:
  python simplenet_feature_reconstruct_traversal.py --dataset screw --data_root ./MVTec_ad --gpu 0 --epochs 20
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import backbones
import simplenet
import utils
import datasets.mvtec as mvtec


def safe_set_eval(module: torch.nn.Module):
    """Safely set module and its submodules to eval mode without calling overridden train().

    Some classes (like SimpleNet) override `train()` with a different signature,
    so calling the base Module.eval()/train() may dispatch to that override and
    raise errors. This function sets the `training` flag and calls eval() on
    standard child modules to avoid invoking overridden train().
    """
    module.training = False
    for m in module.children():
        try:
            m.eval()
        except Exception:
            # best-effort: if child module has odd train signature, set flag directly
            m.training = False


def safe_set_train(module: torch.nn.Module):
    module.training = True
    for m in module.children():
        try:
            m.train()
        except Exception:
            m.training = True



class FeatureDecoder(nn.Module):
    def __init__(self, z_dim, imagesize):
        super().__init__()
        self.imagesize = imagesize
        hidden = max(1024, z_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, 3 * imagesize * imagesize),
        )

    def forward(self, z):
        x = self.net(z)
        x = x.view(-1, 3, self.imagesize, self.imagesize)
        return x


def denormalize(tensor, mean, std):
    """Denormalize a torch tensor normalized by mean/std (channel-first)."""
    mean = torch.tensor(mean, device=tensor.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(1, -1, 1, 1)
    return tensor * std + mean


def compute_latent_stats(encoder, dataloader, device, max_samples=2000):
    safe_set_eval(encoder)
    zs = []
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["image"].to(device)
            feats, _ = encoder._embed(imgs, evaluation=True)
            # If feats are per-patch (N*p, C), average patches per image
            if isinstance(feats, torch.Tensor) and feats.ndim == 2 and feats.shape[0] > imgs.size(0):
                patches_per_image = feats.shape[0] // imgs.size(0)
                feats = feats.view(imgs.size(0), patches_per_image, -1).mean(1)
            zs.append(feats.detach().cpu().numpy())
            if sum(x.shape[0] for x in zs) >= max_samples:
                break
    zs = np.concatenate(zs, axis=0)
    return zs.mean(0), zs.std(0)


def latent_traversal(encoder, decoder, device, img, z_mean, z_std, out_dir, dims_to_vary=6, steps=7, scale=3.0):
    safe_set_eval(encoder)
    safe_set_eval(decoder)
    img = img.to(device)
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        z, _ = encoder._embed(img, evaluation=True)
        # aggregate per-patch to per-image if necessary
        if isinstance(z, torch.Tensor) and z.ndim == 2 and z.shape[0] > img.size(0):
            patches_per_image = z.shape[0] // img.size(0)
            z = z.view(img.size(0), patches_per_image, -1).mean(1)
    z = z.cpu().numpy()[0]
    import torch as _t
    for d in range(min(dims_to_vary, z.shape[0])):
        vals = np.linspace(z[d] - scale * z_std[d], z[d] + scale * z_std[d], steps)
        recon_imgs = []
        for v in vals:
            z_new = z.copy()
            z_new[d] = v
            z_t = _t.from_numpy(z_new).unsqueeze(0).to(device).float()
            with torch.no_grad():
                recon = decoder(z_t)
            recon_imgs.append(recon.cpu())
        grid = torch.cat([img.cpu()] + recon_imgs, dim=0)
        save_image(grid, os.path.join(out_dir, f"dim{d}.png"), nrow=steps + 1)


def train_decoder(encoder, decoder, train_loader, device, epochs=10, lr=1e-3):
    safe_set_eval(encoder)
    safe_set_train(decoder)
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    for ep in range(epochs):
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            with torch.no_grad():
                z, _ = encoder._embed(imgs, evaluation=True)
                # aggregate per-patch features into per-image latent
                if isinstance(z, torch.Tensor) and z.ndim == 2 and z.shape[0] > imgs.size(0):
                    patches_per_image = z.shape[0] // imgs.size(0)
                    z = z.view(imgs.size(0), patches_per_image, -1).mean(1)
            recon = decoder(z)
            loss = F.mse_loss(recon, imgs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        print(f"Decoder Train Epoch {ep+1}/{epochs}  MSE: {total_loss / n:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", default="./MVTec_ad")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--backbone", default="wideresnet50")
    parser.add_argument("--layers", default="layer2,layer3")
    parser.add_argument("--imagesize", type=int, default=128)
    parser.add_argument("--resize", type=int, default=140)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--results", default="results")
    parser.add_argument("--dims_to_vary", type=int, default=6)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--noise_std", type=float, default=0.0, help="Stddev of Gaussian noise added to latent before decoding")
    parser.add_argument("--mix_noise", type=int, default=1, help="If >1, mixes several noise scales like SimpleNet")
    args = parser.parse_args()

    device = utils.set_torch_device([args.gpu])

    # datasets
    train_ds = mvtec.MVTecDataset(args.data_root, classname=args.dataset, resize=args.resize, imagesize=args.imagesize, split=mvtec.DatasetSplit.TRAIN)
    test_ds = mvtec.MVTecDataset(args.data_root, classname=args.dataset, resize=args.resize, imagesize=args.imagesize, split=mvtec.DatasetSplit.TEST)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # build SimpleNet encoder (feature extractor)
    backbone = backbones.load(args.backbone)
    layers = args.layers.split(",") if isinstance(args.layers, str) else list(args.layers)
    encoder = simplenet.SimpleNet(device)
    encoder.load(
        backbone=backbone,
        layers_to_extract_from=layers,
        device=device,
        input_shape=(3, args.imagesize, args.imagesize),
        pretrain_embed_dimension=1536,
        target_embed_dimension=1536,
        patchsize=3,
        embedding_size=256,
        meta_epochs=1,
        gan_epochs=1,
        noise_std=0.05,
        dsc_layers=2,
        dsc_hidden=1024,
        dsc_margin=0.5,
        train_backbone=False,
        pre_proj=0,
    )

    # freeze encoder (we'll not train backbone)
    for p in encoder.parameters():
        p.requires_grad = False

    # determine z-dim from forward_modules
    try:
        z_dim = encoder.forward_modules["preadapt_aggregator"].target_dim
    except Exception:
        # fallback
        z_dim = 1536

    decoder = FeatureDecoder(z_dim=z_dim, imagesize=args.imagesize).to(device)

    # train decoder
    train_decoder(encoder, decoder, train_loader, device, epochs=args.epochs, lr=args.lr)

    # save decoder
    out_dir = Path(args.results) / "simplenet_feature_decoder" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.state_dict(), out_dir / "decoder.pth")
    print(f"Saved decoder to {out_dir / 'decoder.pth'}")

    # compute latent stats
    z_mean, z_std = compute_latent_stats(encoder, train_loader, device)

    # take a few test images
    n_images = 4
    imgs_batch = None
    for batch in test_loader:
        imgs_batch = batch["image"][:n_images]
        break

    save_root = Path("output") / "simplenet_feature_traversal" / args.dataset
    save_root.mkdir(parents=True, exist_ok=True)

    # save original vs reconstructed
    safe_set_eval(encoder)
    safe_set_eval(decoder)
    with torch.no_grad():
        imgs = imgs_batch.to(device)
        z, _ = encoder._embed(imgs, evaluation=True)
        # add noise to decoder input if requested
        if args.noise_std > 0:
            if args.mix_noise <= 1:
                noise = torch.normal(0, args.noise_std, size=z.shape, device=device)
            else:
                noises = torch.stack([
                    torch.normal(0, args.noise_std * (1.1 ** k), size=z.shape, device=device)
                    for k in range(args.mix_noise)
                ], dim=1)  # (N, K, D)
                idxs = torch.randint(0, args.mix_noise, (z.shape[0],), device=device)
                one_hot = F.one_hot(idxs, num_classes=args.mix_noise).float().to(device)
                noise = (noises * one_hot.unsqueeze(-1)).sum(1)
            z_noisy = z + noise
        else:
            z_noisy = z
        recon = decoder(z_noisy)
    # denormalize for saving
    imgs_vis = denormalize(imgs.cpu(), mvtec.IMAGENET_MEAN, mvtec.IMAGENET_STD).clamp(0, 1)
    recon_vis = denormalize(recon.cpu(), mvtec.IMAGENET_MEAN, mvtec.IMAGENET_STD).clamp(0, 1)
    save_image(torch.cat([imgs_vis, recon_vis], dim=0), save_root / "orig_vs_recon.png", nrow=n_images)

    # produce traversal for each image
    for i in range(n_images):
        img = imgs_batch[i].unsqueeze(0)
        out_i = save_root / f"img_{i}"
        latent_traversal(encoder, decoder, device, img, z_mean, z_std, str(out_i), dims_to_vary=args.dims_to_vary, steps=args.steps, scale=args.scale)

    print(f"Saved traversal images under {save_root}")
