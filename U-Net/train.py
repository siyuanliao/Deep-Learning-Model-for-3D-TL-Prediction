import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    DATA, TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR, DEFAULT_WEIGHT_DECAY,
    DEFAULT_PATIENCE, DEFAULT_SEED, DEFAULT_BASE_CHANNELS, DEFAULT_COND_DIM,
    DEFAULT_DROPOUT, DEFAULT_CLUSTERS,
)
from synthetic_data import SyntheticOceanData
from eofk import build_eofk_partition, fit_training_statistics_and_background
from dataset import SoundFieldDataset
from model import SFUNet
from losses import MixedLoss
from train_utils import Metrics, set_seed, fit_environment_normalization, calculate_model_complexity


def parse_args():
    parser = argparse.ArgumentParser(description="Train a U-Net for 3-D underwater acoustic field prediction")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--clusters", type=int, default=DEFAULT_CLUSTERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=str, default="./outputs_unet_eofk")
    return parser.parse_args()


def evaluate(model, loader, criterion, mse_criterion, l1_criterion, device, tl_mean, tl_std):
    model.eval()
    normal = {"loss": 0.0, "l1": 0.0, "grad": 0.0, "mse": 0.0}
    real = {"loss": 0.0, "l1": 0.0, "grad": 0.0, "mse": 0.0}
    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pred = model(x1, x2)
                pred_real = pred * tl_std + tl_mean
                y_real = y * tl_std + tl_mean

                loss = criterion(pred, y)
                l1 = l1_criterion(pred, y)
                grad = criterion.gradient_loss(pred, y)
                mse = mse_criterion(pred, y)

                loss_real = criterion(pred_real, y_real)
                l1_real = l1_criterion(pred_real, y_real)
                grad_real = criterion.gradient_loss(pred_real, y_real)
                mse_real = mse_criterion(pred_real, y_real)

            normal["loss"] += loss.item()
            normal["l1"] += l1.item()
            normal["grad"] += grad.item()
            normal["mse"] += mse.item()
            real["loss"] += loss_real.item()
            real["l1"] += l1_real.item()
            real["grad"] += grad_real.item()
            real["mse"] += mse_real.item()

    n = len(loader)
    normal_m = Metrics(normal["loss"] / n, normal["l1"] / n,
                       normal["grad"] / n, np.sqrt(normal["mse"] / n))
    real_m = Metrics(real["loss"] / n, real["l1"] / n,
                     real["grad"] / n, np.sqrt(real["mse"] / n))
    return normal_m, real_m


def save_preprocessing(out_dir, data, eofk, geo_steady, cluster_counts,
                       x1_mean, x1_std, tl_mean, tl_std):
    np.save(os.path.join(out_dir, "grid_lonlat.npy"), data.grid_lonlat)
    np.save(os.path.join(out_dir, "grid_cluster_labels.npy"), eofk["labels"])
    np.save(os.path.join(out_dir, "long_term_ssp.npy"), eofk["long_term_ssp"])
    np.save(os.path.join(out_dir, "eof_mean.npy"), eofk["mean"])
    np.save(os.path.join(out_dir, "eof_modes.npy"), eofk["modes"])
    np.save(os.path.join(out_dir, "eof_explained_ratio.npy"), eofk["explained_ratio"])
    np.save(os.path.join(out_dir, "eof_score_mean.npy"), eofk["score_mean"])
    np.save(os.path.join(out_dir, "eof_score_std.npy"), eofk["score_std"])
    np.save(os.path.join(out_dir, "kmeans_centers.npy"), eofk["centers"])
    np.save(os.path.join(out_dir, "geo_steady_fields.npy"), geo_steady)
    np.save(os.path.join(out_dir, "geo_steady_cluster_counts.npy"), cluster_counts)
    np.save(os.path.join(out_dir, "x1_mean.npy"), x1_mean)
    np.save(os.path.join(out_dir, "x1_std.npy"), x1_std)
    np.save(os.path.join(out_dir, "t_mean.npy"), np.asarray(tl_mean))
    np.save(os.path.join(out_dir, "t_std.npy"), np.asarray(tl_std))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print("U-Net training for 3-D underwater acoustic field prediction")
    print(f"Device: {device}")
    print(f"Total samples: {DATA.n_samples}")

    data = SyntheticOceanData(DATA)
    train_idx = data.indices_for_years(TRAIN_YEARS)
    val_idx = data.indices_for_years(VAL_YEARS)
    test_idx = data.indices_for_years(TEST_YEARS)
    print(f"Data split: train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}")
    print(f"Year split: train={TRAIN_YEARS}; validation={VAL_YEARS}; test={TEST_YEARS}")

    # EOF-K is fitted using SSPs from the training years only. Bathymetry is unavailable, so clustering uses EOF coefficients only.
    eofk = build_eofk_partition(
        data, TRAIN_YEARS, n_clusters=args.clusters,
        variance_threshold=0.99, seed=args.seed,
    )
    print(f"EOF retained modes: K={eofk['k']}, cumulative explained variance={eofk['explained_ratio'].sum():.6f}")
    print("Grid points per subregion:", np.bincount(eofk["labels"], minlength=args.clusters).tolist())

    # Geo-steady fields and TL normalization statistics are computed from the training set only.
    geo_steady, cluster_counts, tl_mean, tl_std = fit_training_statistics_and_background(
        data, train_idx, eofk["labels"], args.clusters, batch_size=args.batch_size,
    )
    x1_mean, x1_std = fit_environment_normalization(data, train_idx)
    print(f"Training TL statistics: mean={float(tl_mean):.4f} dB, std={float(tl_std):.4f} dB")

    save_preprocessing(
        args.out_dir, data, eofk, geo_steady, cluster_counts,
        x1_mean, x1_std, tl_mean, tl_std,
    )

    train_set = SoundFieldDataset(data, train_idx, eofk["labels"], geo_steady,
                                  x1_mean, x1_std, tl_mean, tl_std)
    val_set = SoundFieldDataset(data, val_idx, eofk["labels"], geo_steady,
                                x1_mean, x1_std, tl_mean, tl_std)
    test_set = SoundFieldDataset(data, test_idx, eofk["labels"], geo_steady,
                                 x1_mean, x1_std, tl_mean, tl_std)

    loader_kwargs = dict(batch_size=args.batch_size, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    # U-Net configuration used for the baseline experiment.
    model = SFUNet(
        x1_dim=52,
        in_ch=4,
        base_ch=DEFAULT_BASE_CHANNELS,
        cond_dim=DEFAULT_COND_DIM,
        dropout=DEFAULT_DROPOUT,
    ).to(device)
    calculate_model_complexity(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )
    criterion = MixedLoss(alpha=1.0, beta=0.5, split_idx=125).to(device)
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history = {
        "train_loss": [], "train_loss_l1": [], "train_loss_grad": [],
        "val_loss": [], "val_loss_l1": [], "val_loss_grad": [],
    }
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        totals = {"loss": 0.0, "l1": 0.0, "grad": 0.0, "mse": 0.0}

        for batch_idx, (x1, x2, y) in enumerate(train_loader):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pred = model(x1, x2)
                loss = criterion(pred, y)
                l1 = l1_criterion(pred, y)
                grad = criterion.gradient_loss(pred, y)
                mse = mse_criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            totals["loss"] += loss.item()
            totals["l1"] += l1.item()
            totals["grad"] += grad.item()
            totals["mse"] += mse.item()
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx:3d}/{len(train_loader):3d}: loss={loss.item():.5f}, l1={l1.item():.5f}")

        n_train = len(train_loader)
        train_m = Metrics(
            totals["loss"] / n_train,
            totals["l1"] / n_train,
            totals["grad"] / n_train,
            np.sqrt(totals["mse"] / n_train),
        )
        val_m, val_real = evaluate(
            model, val_loader, criterion, mse_criterion, l1_criterion,
            device, tl_mean, tl_std,
        )
        scheduler.step(val_m.loss)

        print(f"\n[Epoch {epoch:03d}] lr={optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Train: loss={train_m.loss:.5f}, l1={train_m.l1:.5f}, grad={train_m.grad:.5f}, rmse={train_m.rmse:.5f}")
        print(f"  Val:   loss={val_m.loss:.5f}, l1={val_m.l1:.5f}, grad={val_m.grad:.5f}, rmse={val_m.rmse:.5f}")
        print(f"  Val(real): loss={val_real.loss:.5f}, l1={val_real.l1:.5f}, grad={val_real.grad:.5f}, rmse={val_real.rmse:.5f}")
        print(f"  Time: {time.time() - t0:.1f}s")

        history["train_loss"].append(train_m.loss)
        history["train_loss_l1"].append(train_m.l1)
        history["train_loss_grad"].append(train_m.grad)
        history["val_loss"].append(val_m.loss)
        history["val_loss_l1"].append(val_m.l1)
        history["val_loss_grad"].append(val_m.grad)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val": best_val,
            "history": history,
        }
        torch.save(checkpoint, os.path.join(args.out_dir, "last.pt"))

        if val_m.loss < best_val - 1e-6:
            best_val = val_m.loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))
            print(f"New best model: val_loss={best_val:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= DEFAULT_PATIENCE:
                print(f"Early stopping triggered (patience={DEFAULT_PATIENCE})")
                break

    print("\n" + "=" * 50)
    print("Testing the best model")
    print("=" * 50)
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device))
    test_m, test_real = evaluate(
        model, test_loader, criterion, mse_criterion, l1_criterion,
        device, tl_mean, tl_std,
    )
    print(f"TEST normalized scale: Loss={test_m.loss:.5f}, L1={test_m.l1:.5f}, RMSE={test_m.rmse:.5f}")
    print(f"TEST physical scale:   Loss={test_real.loss:.5f}, L1={test_real.l1:.5f}, RMSE={test_real.rmse:.5f}")

    np.save(os.path.join(args.out_dir, "training_history.npy"), history)
    print("Training complete. Results saved to:", args.out_dir)


if __name__ == "__main__":
    main()
