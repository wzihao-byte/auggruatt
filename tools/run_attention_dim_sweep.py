import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PrunedAttentionGRU import TEMPORAL_POOLING_MODES, prunedAttentionGRU
from dual_domain_model import build_model
from protocol_data import load_dataset_bundle
from protocol_utils import ensure_dir, write_csv, write_json


def _str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def _parse_int_csv(text: str) -> List[int]:
    out: List[int] = []
    for token in str(text).split(","):
        item = token.strip()
        if item:
            out.append(int(item))
    if not out:
        raise ValueError("integer list is empty")
    return out


def _parse_float_csv(text: str) -> List[float]:
    out: List[float] = []
    for token in str(text).split(","):
        item = token.strip()
        if item:
            out.append(float(item))
    if not out:
        raise ValueError("float list is empty")
    return out


def _parse_poolings(text: str) -> List[str]:
    values = [token.strip().lower() for token in str(text).split(",") if token.strip()]
    if not values:
        raise ValueError("pooling list is empty")
    unknown = [name for name in values if name not in TEMPORAL_POOLING_MODES]
    if unknown:
        raise ValueError(f"unsupported pooling(s): {unknown}; expected subset of {TEMPORAL_POOLING_MODES}")
    return values


def _count_params(module: Optional[torch.nn.Module], trainable_only: bool) -> int:
    if module is None:
        return 0
    if trainable_only:
        return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
    return int(sum(p.numel() for p in module.parameters()))


def _resolve_io_dims(
    dataset: str,
    metadata_path: Optional[str],
    input_dim: Optional[int],
    output_dim: Optional[int],
) -> Tuple[int, int]:
    if input_dim is not None and output_dim is not None:
        return int(input_dim), int(output_dim)

    prev_cwd = os.getcwd()
    try:
        os.chdir(REPO_ROOT)
        bundle = load_dataset_bundle(dataset_name=dataset, metadata_path=metadata_path)
    finally:
        os.chdir(prev_cwd)
    resolved_input = int(input_dim) if input_dim is not None else int(bundle.input_size)
    resolved_output = int(output_dim) if output_dim is not None else int(bundle.num_classes)
    return resolved_input, resolved_output


def _pooling_capacity(
    pooling_name: str,
    input_dim: int,
    hidden_size: int,
    attention_dim: int,
    output_dim: int,
) -> Dict[str, Any]:
    model = build_model(
        model_variant="baseline",
        input_dim=input_dim,
        hidden_dim=hidden_size,
        attention_dim=attention_dim,
        output_dim=output_dim,
        temporal_pooling=pooling_name,
    )
    if not isinstance(model, prunedAttentionGRU):
        raise TypeError(f"expected baseline prunedAttentionGRU, got {type(model)!r}")
    if model.fc is None:
        raise RuntimeError("baseline classifier layer is missing")

    if pooling_name == "attn":
        pooling_module = model.attention
        pooling_module_name = "MaskedAttention"
    elif pooling_name in {"mean", "last"}:
        pooling_module = model.temporal_projection
        pooling_module_name = "MaskedLinear(hidden_dim->attention_dim)"
    else:
        raise ValueError(f"unsupported pooling_name={pooling_name!r}")

    fc_trainable = _count_params(model.fc, trainable_only=True)
    pooling_trainable = _count_params(pooling_module, trainable_only=True)
    pooling_total = _count_params(pooling_module, trainable_only=False)

    return {
        "pooling_name": pooling_name,
        "pooling_block_module": pooling_module_name,
        "representation_dim_to_classifier": int(model.fc.in_features),
        "classifier_fc_trainable_params": int(fc_trainable),
        "pooling_block_trainable_params": int(pooling_trainable),
        "pooling_block_total_params": int(pooling_total),
        "classifier_plus_pooling_trainable_params": int(fc_trainable + pooling_trainable),
        "total_params_in_model": _count_params(model, trainable_only=False),
    }


def _make_protocol_command(
    python_exe: str,
    run_protocol_path: Path,
    dataset: str,
    hidden_size: int,
    attention_dim: int,
    epochs: int,
    seed_list_raw: str,
    augment_train: bool,
    mixup_probs_raw: str,
    split_mode: str,
    pooling_name: str,
    output_dir: Path,
    run_name: str,
    device: str,
    num_workers: int,
) -> List[str]:
    return [
        str(python_exe),
        str(run_protocol_path),
        "--dataset",
        str(dataset),
        "--model-variant",
        "baseline",
        "--hidden-size",
        str(hidden_size),
        "--attention-dim",
        str(attention_dim),
        "--epochs",
        str(epochs),
        "--seed-list",
        str(seed_list_raw),
        "--augment-train",
        "true" if augment_train else "false",
        "--mixup-probs",
        str(mixup_probs_raw),
        "--split-mode",
        str(split_mode),
        "--temporal-pooling",
        str(pooling_name),
        "--output-dir",
        str(output_dir),
        "--run-name",
        str(run_name),
        "--device",
        str(device),
        "--num-workers",
        str(num_workers),
    ]


def _run_complete(run_dir: Path, seeds: Sequence[int]) -> bool:
    if not run_dir.exists():
        return False
    for seed in seeds:
        if not (run_dir / f"seed_{int(seed)}" / "metrics.json").exists():
            return False
    return True


def _read_seed_metrics(seed_metrics_path: Path) -> Dict[str, Any]:
    payload = json.loads(seed_metrics_path.read_text(encoding="utf-8"))
    return {
        "accuracy": float(payload["accuracy"]),
        "macro_f1": float(payload["macro_f1"]),
    }


def _aggregate_summary(seed_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for row in seed_rows:
        key = (int(row["attention_dim"]), str(row["pooling"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (attention_dim, pooling), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        acc = np.asarray([float(r["accuracy"]) for r in rows], dtype=np.float64)
        f1 = np.asarray([float(r["macro_f1"]) for r in rows], dtype=np.float64)
        summary_rows.append(
            {
                "attention_dim": int(attention_dim),
                "pooling": str(pooling),
                "num_seeds": int(len(rows)),
                "accuracy_mean": float(acc.mean()),
                "accuracy_std": float(acc.std(ddof=0)),
                "macro_f1_mean": float(f1.mean()),
                "macro_f1_std": float(f1.std(ddof=0)),
            }
        )
    return summary_rows


def _plot_macro_f1(summary_rows: Sequence[Dict[str, Any]], output_path: Path, poolings: Iterable[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required to export attnDim_sweep_macroF1.png") from exc

    grouped: Dict[str, List[Dict[str, Any]]] = {str(pool): [] for pool in poolings}
    for row in summary_rows:
        pool_name = str(row["pooling"])
        grouped.setdefault(pool_name, []).append(row)

    plt.figure(figsize=(8.5, 5.5))
    for pooling_name in poolings:
        rows = sorted(grouped.get(pooling_name, []), key=lambda r: int(r["attention_dim"]))
        if not rows:
            continue
        x = [int(r["attention_dim"]) for r in rows]
        y = [float(r["macro_f1_mean"]) for r in rows]
        e = [float(r["macro_f1_std"]) for r in rows]
        plt.errorbar(x, y, yerr=e, marker="o", capsize=4, linewidth=1.5, label=pooling_name)

    plt.title("ARIL Macro-F1 vs Attention Dimension by Pooling")
    plt.xlabel("attention_dim")
    plt.ylabel("macro_F1")
    plt.grid(alpha=0.3)
    plt.legend(title="pooling")
    ensure_dir(output_path.parent)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ARIL pooling vs attention_dim sweep via run_protocol.py and aggregate metrics."
    )
    parser.add_argument("--dataset", type=str, default="aril")
    parser.add_argument("--metadata-path", type=str, default=None)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed-list", type=str, default="42,52,62")
    parser.add_argument("--augment-train", type=str, default="true")
    parser.add_argument("--mixup-probs", type=str, default="0.0,1.0")
    parser.add_argument("--split-mode", type=str, default="predefined")
    parser.add_argument("--poolings", type=str, default="attn,mean,last")
    parser.add_argument("--attention-dims", type=str, default="32,64,128")
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/protocol/aril_pooling_attnDim_sweep",
        help="Root under which attnDim_{D}/pool_{P}/seed_{S}/... will be written.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="figures/attnDim_sweep_macroF1.png",
    )
    parser.add_argument("--run-protocol-path", type=str, default="run_protocol.py")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--input-dim", type=int, default=None)
    parser.add_argument("--output-dim", type=int, default=None)
    parser.add_argument(
        "--aggregate-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip launching new protocol runs and only aggregate existing run outputs.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip protocol launch when all seed metrics already exist for a combo.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset = str(args.dataset).strip().lower()
    poolings = _parse_poolings(args.poolings)
    attention_dims = _parse_int_csv(args.attention_dims)
    seeds = _parse_int_csv(args.seed_list)
    mixup_probs = _parse_float_csv(args.mixup_probs)
    if len(mixup_probs) != 2:
        raise ValueError("mixup_probs must contain exactly two values")
    augment_train = _str2bool(args.augment_train)

    output_root = Path(args.output_root)
    absolute_output_root = (REPO_ROOT / output_root).resolve()
    ensure_dir(absolute_output_root)

    run_protocol_path = (REPO_ROOT / args.run_protocol_path).resolve()
    if not run_protocol_path.exists():
        raise FileNotFoundError(f"run_protocol.py not found: {run_protocol_path}")

    input_dim, output_dim = _resolve_io_dims(
        dataset=dataset,
        metadata_path=args.metadata_path,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
    )

    seed_rows: List[Dict[str, Any]] = []
    for attention_dim in attention_dims:
        for pooling_name in poolings:
            relative_combo_output = output_root / f"attnDim_{int(attention_dim)}"
            run_name = f"pool_{pooling_name}"
            run_dir = (REPO_ROOT / relative_combo_output / run_name).resolve()
            ensure_dir(run_dir)

            pooling_capacity = _pooling_capacity(
                pooling_name=pooling_name,
                input_dim=input_dim,
                hidden_size=int(args.hidden_size),
                attention_dim=int(attention_dim),
                output_dim=output_dim,
            )

            command = _make_protocol_command(
                python_exe=args.python_exe,
                run_protocol_path=run_protocol_path,
                dataset=dataset,
                hidden_size=int(args.hidden_size),
                attention_dim=int(attention_dim),
                epochs=int(args.epochs),
                seed_list_raw=args.seed_list,
                augment_train=augment_train,
                mixup_probs_raw=args.mixup_probs,
                split_mode=str(args.split_mode),
                pooling_name=pooling_name,
                output_dir=relative_combo_output,
                run_name=run_name,
                device=str(args.device),
                num_workers=int(args.num_workers),
            )
            run_config_payload = {
                "dataset": dataset,
                "model_variant": "baseline",
                "hidden_size": int(args.hidden_size),
                "attention_dim": int(attention_dim),
                "temporal_pooling": pooling_name,
                "epochs": int(args.epochs),
                "seed_list": [int(seed) for seed in seeds],
                "augment_train": bool(augment_train),
                "mixup_probs": [float(v) for v in mixup_probs],
                "split_mode": str(args.split_mode),
                "output_dir": str(relative_combo_output),
                "run_name": run_name,
                "run_dir": str(run_dir),
                "input_dim": int(input_dim),
                "output_dim": int(output_dim),
                "pooling_block": pooling_capacity,
                "command": command,
            }
            write_json(run_dir / "sweep_run_config.json", run_config_payload)

            if not args.aggregate_only:
                if args.skip_existing and _run_complete(run_dir=run_dir, seeds=seeds):
                    print(f"[skip] attnDim={attention_dim} pool={pooling_name} (all seed metrics exist)")
                else:
                    print(f"[run ] attnDim={attention_dim} pool={pooling_name}")
                    subprocess.run(command, cwd=str(REPO_ROOT), check=True)

            for seed in seeds:
                seed_dir = run_dir / f"seed_{int(seed)}"
                seed_cfg = dict(run_config_payload)
                seed_cfg["seed"] = int(seed)
                seed_cfg["seed_dir"] = str(seed_dir)
                write_json(seed_dir / "sweep_run_config.json", seed_cfg)

                seed_metrics_path = seed_dir / "metrics.json"
                if not seed_metrics_path.exists():
                    raise FileNotFoundError(
                        f"missing seed metrics: {seed_metrics_path} "
                        f"(run with --no-aggregate-only to generate)"
                    )
                seed_metrics = _read_seed_metrics(seed_metrics_path)
                seed_rows.append(
                    {
                        "attention_dim": int(attention_dim),
                        "pooling": pooling_name,
                        "seed": int(seed),
                        "accuracy": float(seed_metrics["accuracy"]),
                        "macro_f1": float(seed_metrics["macro_f1"]),
                    }
                )

    if not seed_rows:
        raise RuntimeError("no seed metrics collected")

    seed_rows = sorted(seed_rows, key=lambda row: (int(row["attention_dim"]), str(row["pooling"]), int(row["seed"])))
    seed_metrics_csv = absolute_output_root / "attention_dim_sweep_seed_metrics.csv"
    write_csv(
        seed_metrics_csv,
        seed_rows,
        fieldnames=["attention_dim", "pooling", "seed", "accuracy", "macro_f1"],
    )

    summary_rows = _aggregate_summary(seed_rows)
    summary_csv = absolute_output_root / "attention_dim_sweep_summary.csv"
    write_csv(
        summary_csv,
        summary_rows,
        fieldnames=[
            "attention_dim",
            "pooling",
            "num_seeds",
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
        ],
    )

    plot_path = (REPO_ROOT / args.plot_path).resolve()
    _plot_macro_f1(summary_rows=summary_rows, output_path=plot_path, poolings=poolings)

    print(f"saved seed metrics: {seed_metrics_csv}")
    print(f"saved summary: {summary_csv}")
    print(f"saved plot: {plot_path}")


if __name__ == "__main__":
    main()
