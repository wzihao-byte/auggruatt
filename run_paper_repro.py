import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from DataLoader.tensordata import TensorData
from augmentation import augmentation
from dual_domain_model import MODEL_VARIANTS, build_model, model_metadata
from paper_repro.aril_official_split import load_aril_official_split
from paper_repro.plot_confusion_matrix import save_confusion_matrix_set
from protocol_utils import (
    classification_metrics,
    ensure_dir,
    parse_float_list,
    parse_int_list,
    set_global_seed,
    timestamp_now,
    to_serializable,
    write_csv,
    write_json,
)
from tools.mixup import mixup


class SoftTargetCrossEntropy(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def resolve_device(device_arg: str) -> torch.device:
    if str(device_arg).strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper reproduction runner (official ARIL split only).")
    parser.add_argument("--dataset", type=str, default="aril", choices=["aril"])
    parser.add_argument("--task", type=str, default="activity", choices=["activity", "location"])

    parser.add_argument("--model_variant", type=str, default="baseline", choices=list(MODEL_VARIANTS))
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--attention_dim", type=int, default=32)
    parser.add_argument("--freq_feature_dim", type=int, default=64)
    parser.add_argument("--fusion_hidden_dim", type=int, default=64)
    parser.add_argument("--freq_use_abs", type=str, default="false")
    parser.add_argument("--freq_eps", type=float, default=1e-8)

    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--learningrate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="42,52,62")
    parser.add_argument("--mixup_probs", type=str, default="0.3,0.7")
    parser.add_argument("--augment_train", type=str, default="true")

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_model", type=str, default="true")

    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="results/paper_repro")
    parser.add_argument("--aril_dir", type=str, default="ARIL")
    return parser


def normalize_args(raw: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(raw)
    cfg["dataset"] = str(cfg["dataset"]).strip().lower()
    if cfg["dataset"] != "aril":
        raise ValueError("paper reproduction runner currently supports dataset='aril' only")

    cfg["model_variant"] = str(cfg["model_variant"]).strip().lower()
    if cfg["model_variant"] not in MODEL_VARIANTS:
        raise ValueError(f"unsupported model_variant={cfg['model_variant']}")

    cfg["freq_use_abs"] = str2bool(cfg["freq_use_abs"])
    cfg["augment_train"] = str2bool(cfg["augment_train"])
    cfg["save_model"] = str2bool(cfg["save_model"])
    cfg["seeds"] = parse_int_list(cfg["seeds"])
    cfg["mixup_probs"] = parse_float_list(cfg["mixup_probs"])

    if len(cfg["mixup_probs"]) != 2:
        raise ValueError("mixup_probs must contain exactly two comma-separated values")
    if min(cfg["mixup_probs"]) < 0:
        raise ValueError("mixup_probs must be non-negative")
    if sum(cfg["mixup_probs"]) <= 0:
        raise ValueError("mixup_probs sum must be > 0")

    run_name = str(cfg["run_name"]).strip()
    if not run_name:
        run_name = f"{cfg['dataset']}_{cfg['model_variant']}_paper_repro_{timestamp_now()}"
    cfg["run_name"] = run_name
    return cfg


def build_seed_dataloaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    num_workers: int,
    seed: int,
    augment_train_enabled: bool,
) -> tuple[DataLoader, DataLoader]:
    if augment_train_enabled:
        _, _, x_train_aug, _, _, y_train_aug = augmentation(
            torch.FloatTensor(x_train),
            torch.FloatTensor(y_train),
        )
        x_train_use = x_train_aug.detach().cpu().numpy().astype(np.float32)
        y_train_use = y_train_aug.detach().cpu().numpy().astype(np.float32)
    else:
        x_train_use = np.asarray(x_train, dtype=np.float32)
        y_train_use = np.asarray(y_train, dtype=np.float32)

    x_test_use = np.asarray(x_test, dtype=np.float32)
    y_test_use = np.asarray(y_test, dtype=np.float32)

    train_ds = TensorData(torch.FloatTensor(x_train_use), torch.FloatTensor(y_train_use))
    test_ds = TensorData(torch.FloatTensor(x_test_use), torch.FloatTensor(y_test_use))

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    loader_kwargs: Dict[str, Any] = {"num_workers": int(num_workers)}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(batch_size),
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, test_loader


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            logits = model(inputs)
            if isinstance(logits, tuple):
                logits = logits[0]

            loss = criterion(logits, labels)
            total_loss += float(loss.item() * inputs.shape[0])

            pred_idx = torch.argmax(logits, dim=1)
            true_idx = torch.argmax(labels, dim=1)
            y_pred.extend(pred_idx.detach().cpu().numpy().tolist())
            y_true.extend(true_idx.detach().cpu().numpy().tolist())

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    metrics = classification_metrics(
        y_true=y_true_np,
        y_pred=y_pred_np,
        num_classes=num_classes,
        class_names=class_names,
    )
    metrics["loss"] = float(total_loss / max(len(y_true), 1))
    return metrics


def train_one_seed(
    seed: int,
    cfg: Dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
    device: torch.device,
    run_dir: Path,
) -> Dict[str, Any]:
    seed_dir = ensure_dir(run_dir / f"seed_{seed}")

    set_global_seed(int(seed), deterministic=True)
    train_loader, test_loader = build_seed_dataloaders(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        batch_size=cfg["batchsize"],
        num_workers=cfg["num_workers"],
        seed=seed,
        augment_train_enabled=cfg["augment_train"],
    )

    input_size = int(x_train.shape[-1])
    num_classes = int(y_train.shape[1])

    model = build_model(
        model_variant=cfg["model_variant"],
        input_dim=input_size,
        hidden_dim=cfg["hidden_size"],
        attention_dim=cfg["attention_dim"],
        output_dim=num_classes,
        freq_feature_dim=cfg["freq_feature_dim"],
        fusion_hidden_dim=cfg["fusion_hidden_dim"],
        freq_use_abs=cfg["freq_use_abs"],
        freq_eps=cfg["freq_eps"],
    ).to(device)

    criterion = SoftTargetCrossEntropy()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learningrate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    mixup_probs = cfg["mixup_probs"]
    p_mixup = float(mixup_probs[0] / max(sum(mixup_probs), 1e-12))

    history_rows: List[Dict[str, Any]] = []
    best_eval_macro_f1 = -1.0
    best_state = None

    train_start = time.perf_counter()
    for epoch in range(cfg["epochs"]):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            use_mixup = np.random.rand() < p_mixup
            if use_mixup:
                mixed_inputs, label_a, label_b, lam = mixup(inputs, labels, 1.0)
                lam_value = float(lam.item())
                inputs_train = mixed_inputs
                targets = lam_value * label_a + (1.0 - lam_value) * label_b
            else:
                inputs_train = inputs
                targets = labels

            logits = model(inputs_train)
            if isinstance(logits, tuple):
                logits = logits[0]

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            pred_idx = torch.argmax(logits, dim=1)
            true_idx = torch.argmax(targets, dim=1)
            running_correct += int((pred_idx == true_idx).sum().item())
            running_total += int(inputs_train.shape[0])
            running_loss += float(loss.item() * inputs_train.shape[0])

        scheduler.step()

        train_loss = float(running_loss / max(running_total, 1))
        train_acc = float(running_correct / max(running_total, 1))

        eval_metrics = evaluate_model(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
        )

        history_rows.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "eval_loss": float(eval_metrics["loss"]),
                "eval_accuracy": float(eval_metrics["accuracy"]),
                "eval_macro_f1": float(eval_metrics["macro_f1"]),
            }
        )

        if float(eval_metrics["macro_f1"]) > best_eval_macro_f1:
            best_eval_macro_f1 = float(eval_metrics["macro_f1"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if cfg["log_every"] > 0 and (
            (epoch + 1) % cfg["log_every"] == 0 or epoch == 0 or (epoch + 1) == cfg["epochs"]
        ):
            print(
                f"[seed={seed}] Epoch {epoch + 1}/{cfg['epochs']} "
                f"train_acc={train_acc:.4f} train_loss={train_loss:.4f} "
                f"test_acc={eval_metrics['accuracy']:.4f} test_macro_f1={eval_metrics['macro_f1']:.4f}"
            )

    train_seconds = float(time.perf_counter() - train_start)
    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        class_names=class_names,
    )

    confusion = final_metrics["confusion_matrix"]
    np.save(seed_dir / "confusion_matrix.npy", confusion)
    save_confusion_matrix_set(confusion=confusion, class_names=class_names, out_dir=seed_dir)

    model_info = dict(model_metadata(model))
    if cfg["save_model"]:
        checkpoint_path = seed_dir / f"best_model_{cfg['model_variant']}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "dataset": cfg["dataset"],
                "task": cfg["task"],
                "model_variant": cfg["model_variant"],
                "hidden_size": cfg["hidden_size"],
                "attention_dim": cfg["attention_dim"],
                "freq_feature_dim": cfg["freq_feature_dim"],
                "fusion_hidden_dim": cfg["fusion_hidden_dim"],
                "freq_use_abs": cfg["freq_use_abs"],
                "freq_eps": cfg["freq_eps"],
                "input_size": input_size,
                "num_classes": num_classes,
                "seed": int(seed),
            },
            checkpoint_path,
        )
    else:
        checkpoint_path = None

    metrics_row = {
        "seed": int(seed),
        "run_name": cfg["run_name"],
        "dataset": cfg["dataset"],
        "task": cfg["task"],
        "model_variant": cfg["model_variant"],
        "fusion_type": str(model_info.get("fusion_type", "none")),
        "accuracy": float(final_metrics["accuracy"]),
        "acc": float(final_metrics["accuracy"]),
        "macro_f1": float(final_metrics["macro_f1"]),
        "loss": float(final_metrics["loss"]),
        "num_samples": int(final_metrics["num_samples"]),
        "train_time_sec": train_seconds,
        "hidden_size": int(cfg["hidden_size"]),
        "attention_dim": int(cfg["attention_dim"]),
        "freq_feature_dim": int(cfg["freq_feature_dim"]),
        "fusion_hidden_dim": int(cfg["fusion_hidden_dim"]),
        "frequency_use_abs": bool(cfg["freq_use_abs"]),
        "raw_frequency_signature_dim": model_info.get("raw_frequency_signature_dim"),
        "projected_frequency_feature_dim": model_info.get("projected_frequency_feature_dim"),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else "",
    }

    write_json(seed_dir / "metrics.json", metrics_row)
    write_csv(seed_dir / "metrics.csv", [metrics_row], fieldnames=list(metrics_row.keys()))
    write_csv(
        seed_dir / "per_class_f1.csv",
        final_metrics["per_class_rows"],
        fieldnames=["class_index", "class_name", "f1", "support"],
    )
    write_csv(
        seed_dir / "training_history.csv",
        history_rows,
        fieldnames=["epoch", "train_loss", "train_accuracy", "eval_loss", "eval_accuracy", "eval_macro_f1"],
    )
    write_json(seed_dir / "run_config_snapshot.json", to_serializable({**cfg, "seed": int(seed)}))

    return {
        "seed_metrics": metrics_row,
        "per_class_rows": final_metrics["per_class_rows"],
    }


def aggregate_run_outputs(
    cfg: Dict[str, Any],
    run_dir: Path,
    seed_outputs: Sequence[Dict[str, Any]],
    class_names: Sequence[str],
) -> None:
    seed_rows = [entry["seed_metrics"] for entry in seed_outputs]
    write_csv(run_dir / "seed_stats.csv", seed_rows, fieldnames=list(seed_rows[0].keys()))

    acc = np.asarray([float(row["accuracy"]) for row in seed_rows], dtype=np.float64)
    macro_f1 = np.asarray([float(row["macro_f1"]) for row in seed_rows], dtype=np.float64)
    loss = np.asarray([float(row["loss"]) for row in seed_rows], dtype=np.float64)
    train_sec = np.asarray([float(row["train_time_sec"]) for row in seed_rows], dtype=np.float64)

    summary_row = {
        "run_name": cfg["run_name"],
        "dataset": cfg["dataset"],
        "task": cfg["task"],
        "model_variant": cfg["model_variant"],
        "fusion_type": str(seed_rows[0].get("fusion_type", "none")),
        "seeds": ",".join(str(s) for s in cfg["seeds"]),
        "num_seeds": int(len(cfg["seeds"])),
        "hidden_size": int(cfg["hidden_size"]),
        "attention_dim": int(cfg["attention_dim"]),
        "freq_feature_dim": int(cfg["freq_feature_dim"]),
        "fusion_hidden_dim": int(cfg["fusion_hidden_dim"]),
        "batchsize": int(cfg["batchsize"]),
        "learningrate": float(cfg["learningrate"]),
        "epochs": int(cfg["epochs"]),
        "augment_train": bool(cfg["augment_train"]),
        "mixup_probs": ",".join(str(v) for v in cfg["mixup_probs"]),
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std(ddof=0)),
        "macro_f1_mean": float(macro_f1.mean()),
        "macro_f1_std": float(macro_f1.std(ddof=0)),
        "loss_mean": float(loss.mean()),
        "loss_std": float(loss.std(ddof=0)),
        "train_time_sec_mean": float(train_sec.mean()),
        "train_time_sec_std": float(train_sec.std(ddof=0)),
    }
    write_csv(run_dir / "summary.csv", [summary_row], fieldnames=list(summary_row.keys()))
    write_json(run_dir / "summary.json", summary_row)

    per_class_values = []
    for entry in seed_outputs:
        ordered = sorted(entry["per_class_rows"], key=lambda row: int(row["class_index"]))
        per_class_values.append([float(row["f1"]) for row in ordered])
    per_class_values_np = np.asarray(per_class_values, dtype=np.float64)

    per_class_rows = []
    for idx, class_name in enumerate(class_names):
        vals = per_class_values_np[:, idx]
        per_class_rows.append(
            {
                "class_index": int(idx),
                "class_name": class_name,
                "f1_mean": float(vals.mean()),
                "f1_std": float(vals.std(ddof=0)),
            }
        )
    write_csv(
        run_dir / "per_class_f1_summary.csv",
        per_class_rows,
        fieldnames=["class_index", "class_name", "f1_mean", "f1_std"],
    )


def main() -> None:
    parser = build_parser()
    args = normalize_args(vars(parser.parse_args()))
    device = resolve_device(args["device"])

    run_dir = ensure_dir(Path(args["out_dir"]).resolve() / args["run_name"])
    print(f"Using device: {device}")
    print(f"Run directory: {run_dir}")

    x_train, y_train, x_test, y_test, class_names = load_aril_official_split(
        aril_dir=args["aril_dir"],
        task=args["task"],
    )
    dataset_summary = {
        "dataset": args["dataset"],
        "task": args["task"],
        "split_type": "official_train_test",
        "x_train_shape": list(x_train.shape),
        "x_test_shape": list(x_test.shape),
        "num_classes": int(y_train.shape[1]),
        "class_names": list(class_names),
    }

    write_json(run_dir / "run_config_snapshot.json", to_serializable(args))
    write_json(run_dir / "dataset_summary.json", dataset_summary)

    seed_outputs = []
    for seed in args["seeds"]:
        print(f"=== Seed {seed} ===")
        seed_result = train_one_seed(
            seed=int(seed),
            cfg=args,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            class_names=class_names,
            device=device,
            run_dir=run_dir,
        )
        seed_outputs.append(seed_result)

    aggregate_run_outputs(cfg=args, run_dir=run_dir, seed_outputs=seed_outputs, class_names=class_names)
    print(f"Paper reproduction run completed: {run_dir}")


if __name__ == "__main__":
    main()

