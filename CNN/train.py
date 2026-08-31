# 3-D transmission-loss field prediction with EOF-K geo-steady backgrounds and synthetic data
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import DATA, build_source_grid
from synthetic_data import split_indices_by_year
from eofk import fit_eof_k, build_geo_steady_fields
from dataset import SoundFieldDataset
from model import SimpleConditionalCNN
from losses import MixedLoss
from train_utils import Metrics, set_seed, compute_training_statistics
from my_functions import calculate_model_complexity


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clusters", type=int, default=4,
                   help="EOF-K subregion number M")
    p.add_argument("--variance-threshold", type=float, default=0.99)
    p.add_argument("--prep-batch-size", type=int, default=32,
                   help="Batch size used for streaming TL statistics/background construction")
    p.add_argument("--num-workers", type=int, default=0,
                   help="0 is the most portable default for the synthetic on-the-fly dataset")
    p.add_argument("--out-dir", type=str, default="./baseline_cnn_outputs")
    return p.parse_args()


def main():
    args = parse_args()
    print("Baseline CNN training: EOF-K geo-steady field + synthetic data")
    print(f"Total samples: {DATA.n_samples} = {DATA.years} years x {DATA.months} months x {DATA.points} grid points")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Chronological split: years 1-11 train, year 12 val, year 13 test.
    # ------------------------------------------------------------------
    train_idx, val_idx, test_idx = split_indices_by_year(DATA)
    print(f"Data split: train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}")
    print("Year split: Train=1-11, Validation=12, Test=13")

    # ------------------------------------------------------------------
    # 2. EOF-K fitted from TRAINING SSP ONLY.
    #    Bathymetry is unavailable, so only EOF coefficients are clustered.
    # ------------------------------------------------------------------
    print("\n[1/3] Fitting EOF-K from training SSPs...")
    eofk = fit_eof_k(
        train_idx,
        n_clusters=args.clusters,
        variance_threshold=args.variance_threshold,
        seed=args.seed,
        cfg=DATA,
    )
    cumulative = float(eofk.explained_ratio.sum())
    print(f"EOF retained K={eofk.eof_modes.shape[1]}, cumulative variance={cumulative:.6f}")
    unique, cluster_sizes = np.unique(eofk.labels, return_counts=True)
    print("Grid-point counts per cluster:", dict(zip(unique.tolist(), cluster_sizes.tolist())))

    # ------------------------------------------------------------------
    # 3. Geo-steady TL field for each acoustic subregion, TRAINING ONLY.
    # ------------------------------------------------------------------
    print("\n[2/3] Computing training-only geo-steady TL fields for EOF-K subregions...")
    geo_steady, cluster_sample_counts = build_geo_steady_fields(
        train_idx,
        labels=eofk.labels,
        n_clusters=args.clusters,
        batch_size=args.prep_batch_size,
        cfg=DATA,
    )
    print("Training samples used in each subregion average:", cluster_sample_counts.tolist())

    # ------------------------------------------------------------------
    # 4. Normalization also uses training samples only.
    # ------------------------------------------------------------------
    print("\n[3/3] Computing training-only normalization statistics...")
    x1_mean, x1_std, t_mean, t_std = compute_training_statistics(
        train_idx, batch_size=args.prep_batch_size, cfg=DATA
    )
    print(f"Training TL mean={float(t_mean):.3f} dB, std={float(t_std):.3f} dB")

    # Save EOF-K/background metadata for reproducibility and plotting.
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "grid_lonlat.npy"), build_source_grid(DATA))
    np.save(os.path.join(args.out_dir, "grid_cluster_labels.npy"), eofk.labels)
    np.save(os.path.join(args.out_dir, "eof_mean.npy"), eofk.eof_mean)
    np.save(os.path.join(args.out_dir, "eof_modes.npy"), eofk.eof_modes)
    np.save(os.path.join(args.out_dir, "eof_explained_ratio.npy"), eofk.explained_ratio)
    np.save(os.path.join(args.out_dir, "eof_score_mean.npy"), eofk.scores_mean)
    np.save(os.path.join(args.out_dir, "eof_score_std.npy"), eofk.scores_std)
    np.save(os.path.join(args.out_dir, "kmeans_centers.npy"), eofk.centers)
    np.save(os.path.join(args.out_dir, "geo_steady_fields.npy"), geo_steady)
    np.save(os.path.join(args.out_dir, "geo_steady_cluster_counts.npy"), cluster_sample_counts)

    # ------------------------------------------------------------------
    # 5. DataLoaders. Targets are synthesized deterministically on demand.
    # ------------------------------------------------------------------
    common = dict(
        cluster_labels=eofk.labels,
        geo_steady_fields=geo_steady,
        x1_mean=x1_mean,
        x1_std=x1_std,
        t_mean=t_mean,
        t_std=t_std,
        cfg=DATA,
    )
    train_loader = DataLoader(
        SoundFieldDataset(train_idx, **common),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        SoundFieldDataset(val_idx, **common),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        SoundFieldDataset(test_idx, **common),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # 6. Network: preserve the original architecture and default parameter scale.
    # ------------------------------------------------------------------
    model = SimpleConditionalCNN(
        x1_dim=52,
        in_ch=4,
        base_ch=128,
        cond_dim=128,
        num_blocks=8,
    ).to(device)
    calculate_model_complexity(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    criterion = MixedLoss(alpha=1.0, beta=0.5, split_idx=125)
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_val, patience_counter = float("inf"), 0
    history = {
        "train_loss": [], "train_loss_l1": [], "train_loss_grad": [],
        "val_loss": [], "val_loss_l1": [], "val_loss_grad": []
    }
    default_weights = torch.ones(1, 1, 36, 250, device=device)

    # ------------------------------------------------------------------
    # 7. Training loop using the baseline optimization settings.
    # ------------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_metrics = {"loss": 0, "l1": 0, "grad": 0, "rmse": 0}

        for batch_idx, (x1, x2, y) in enumerate(train_loader):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pred = model(x1, x2)
                loss = criterion(pred, y)
                l1_loss = l1_criterion(pred, y)
                grad_loss = criterion.gradient_loss(pred, y, default_weights)
                mse_loss = mse_criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                train_metrics["loss"] += loss.item()
                train_metrics["l1"] += l1_loss.item()
                train_metrics["grad"] += grad_loss.item()
                train_metrics["rmse"] += mse_loss.item()

            if batch_idx % 50 == 0:
                print(
                    f"  Batch {batch_idx:3d}/{len(train_loader):3d}: "
                    f"loss={loss.item():.5f}, l1={l1_loss.item():.5f}"
                )

        model.eval()
        val_metrics = {"loss": 0, "l1": 0, "grad": 0, "rmse": 0}
        val_metrics_real = {"loss": 0, "l1": 0, "grad": 0, "rmse": 0}
        with torch.no_grad():
            for x1, x2, y in val_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    pred = model(x1, x2)
                    pred_real = pred * t_std + t_mean
                    y_real = y * t_std + t_mean

                    loss = criterion(pred, y)
                    loss_real = criterion(pred_real, y_real)
                    l1_loss = l1_criterion(pred, y)
                    l1_loss_real = l1_criterion(pred_real, y_real)
                    grad_loss = criterion.gradient_loss(pred, y, default_weights)
                    grad_loss_real = criterion.gradient_loss(pred_real, y_real, default_weights)
                    mse_loss = mse_criterion(pred, y)
                    mse_loss_real = mse_criterion(pred_real, y_real)

                val_metrics["loss"] += loss.item()
                val_metrics["l1"] += l1_loss.item()
                val_metrics["grad"] += grad_loss.item()
                val_metrics["rmse"] += mse_loss.item()

                val_metrics_real["loss"] += loss_real.item()
                val_metrics_real["l1"] += l1_loss_real.item()
                val_metrics_real["grad"] += grad_loss_real.item()
                val_metrics_real["rmse"] += mse_loss_real.item()

        n_train = len(train_loader)
        n_val = len(val_loader)
        train_m = Metrics(
            loss=train_metrics["loss"] / n_train,
            l1=train_metrics["l1"] / n_train,
            grad=train_metrics["grad"] / n_train,
            rmse=np.sqrt(train_metrics["rmse"] / n_train),
        )
        val_m = Metrics(
            loss=val_metrics["loss"] / n_val,
            l1=val_metrics["l1"] / n_val,
            grad=val_metrics["grad"] / n_val,
            rmse=np.sqrt(val_metrics["rmse"] / n_val),
        )
        val_m_real = Metrics(
            loss=val_metrics_real["loss"] / n_val,
            l1=val_metrics_real["l1"] / n_val,
            grad=val_metrics_real["grad"] / n_val,
            rmse=np.sqrt(val_metrics_real["rmse"] / n_val),
        )

        scheduler.step(val_m.loss)
        print(f"\n[Epoch {epoch:03d}] lr={optimizer.param_groups[0]['lr']:.2e}")
        print(
            f"  Train: loss={train_m.loss:.5f}, l1={train_m.l1:.5f}, "
            f"grad={train_m.grad:.5f}, rmse={train_m.rmse:.5f}"
        )
        print(
            f"  Val:   loss={val_m.loss:.5f}, l1={val_m.l1:.5f}, "
            f"grad={val_m.grad:.5f}, rmse={val_m.rmse:.5f}"
        )
        print(
            f"Validation metrics in physical TL scale: loss={val_m_real.loss:.5f}, "
            f"l1={val_m_real.l1:.5f}, grad={val_m_real.grad:.5f}, "
            f"rmse={val_m_real.rmse:.5f}"
        )
        print(f"  Time: {time.time() - t0:.1f}s")

        history["train_loss"].append(train_m.loss)
        history["train_loss_l1"].append(train_m.l1)
        history["train_loss_grad"].append(train_m.grad)
        history["val_loss"].append(val_m.loss)
        history["val_loss_l1"].append(val_m.l1)
        history["val_loss_grad"].append(val_m.grad)

        checkpoint = {"epoch": epoch, "best_val": best_val, "history": history}
        torch.save(checkpoint, os.path.join(args.out_dir, "last.pt"))

        if val_m.loss < best_val - 1e-6:
            best_val = val_m.loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))
            print(f"New best model: val_loss={best_val:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered (patience={args.patience})")
                break

    # ------------------------------------------------------------------
    # 8. Test best model.
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Testing the best model...")
    print("=" * 50)
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device))
    model.eval()

    test_metrics = {"loss": 0, "l1": 0, "rmse": 0}
    test_metrics_real = {"loss": 0, "l1": 0, "rmse": 0}
    with torch.no_grad():
        for x1, x2, y in test_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pred = model(x1, x2)
                loss = criterion(pred, y)
                l1_loss = l1_criterion(pred, y)
                mse_loss = mse_criterion(pred, y)

                pred_real = pred * t_std + t_mean
                y_real = y * t_std + t_mean
                loss_real = criterion(pred_real, y_real)
                l1_loss_real = l1_criterion(pred_real, y_real)
                mse_loss_real = mse_criterion(pred_real, y_real)

            test_metrics["loss"] += loss.item()
            test_metrics["l1"] += l1_loss.item()
            test_metrics["rmse"] += torch.sqrt(mse_loss).item()
            test_metrics_real["loss"] += loss_real.item()
            test_metrics_real["l1"] += l1_loss_real.item()
            test_metrics_real["rmse"] += torch.sqrt(mse_loss_real).item()

    n_test = len(test_loader)
    print("\n[TEST - normalized scale]")
    print(f"  Loss: {test_metrics['loss'] / n_test:.5f}")
    print(f"  L1:   {test_metrics['l1'] / n_test:.5f}")
    print(f"  RMSE: {test_metrics['rmse'] / n_test:.5f}")
    print("\n[TEST - physical TL scale]")
    print(f"  Loss: {test_metrics_real['loss'] / n_test:.5f}")
    print(f"  L1:   {test_metrics_real['l1'] / n_test:.5f}")
    print(f"  RMSE: {test_metrics_real['rmse'] / n_test:.5f}")

    np.save(os.path.join(args.out_dir, "x1_mean.npy"), x1_mean)
    np.save(os.path.join(args.out_dir, "x1_std.npy"), x1_std)
    np.save(os.path.join(args.out_dir, "t_mean.npy"), t_mean)
    np.save(os.path.join(args.out_dir, "t_std.npy"), t_std)
    np.save(os.path.join(args.out_dir, "training_history.npy"), history)
    print("\nTraining complete. Model and parameters saved to:", args.out_dir)


if __name__ == "__main__":
    main()
