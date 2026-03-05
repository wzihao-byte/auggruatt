import argparse
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _split_arg_list(raw: str) -> List[str]:
    values = []
    for token in str(raw).split(","):
        item = token.strip()
        if item:
            values.append(item)
    if not values:
        raise ValueError("empty path list")
    return values


def _as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _count_seeds(seed_text: str) -> int:
    tokens = [token.strip() for token in str(seed_text).split(",") if token.strip()]
    return len(tokens)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_model_variant_from_run_dir(run_dir: Path) -> str:
    snapshot = run_dir / "run_config_snapshot.json"
    if not snapshot.exists():
        return "unknown"
    try:
        import json

        payload = json.loads(snapshot.read_text(encoding="utf-8"))
        variant = str(payload.get("model_variant", "")).strip().lower()
        return variant if variant else "unknown"
    except Exception:
        return "unknown"


def _standard_row(
    source: str,
    run_name: str,
    model_variant: str,
    accuracy_mean: float,
    accuracy_std: float,
    macro_f1_mean: float,
    macro_f1_std: float,
    loss_mean: float,
    loss_std: float,
    seeds: str,
    num_seeds: int,
    summary_path: Path,
    hidden_size: float = math.nan,
    attention_dim: float = math.nan,
) -> Dict[str, Any]:
    return {
        "source": source,
        "run_name": run_name,
        "model_variant": model_variant,
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "macro_f1_mean": macro_f1_mean,
        "macro_f1_std": macro_f1_std,
        "loss_mean": loss_mean,
        "loss_std": loss_std,
        "seeds": seeds,
        "num_seeds": num_seeds,
        "hidden_size": hidden_size,
        "attention_dim": attention_dim,
        "summary_path": str(summary_path.resolve()),
    }


def _load_rows_from_metric_csv(path: Path, source: str) -> List[Dict[str, Any]]:
    rows = _read_csv(path)
    if not rows:
        return []

    metric_map = {str(r.get("metric", "")).strip().lower(): r for r in rows}
    if not metric_map:
        return []

    run_dir = path.parent
    run_name = run_dir.name
    model_variant = _load_model_variant_from_run_dir(run_dir)
    return [
        _standard_row(
            source=source,
            run_name=run_name,
            model_variant=model_variant,
            accuracy_mean=_as_float(metric_map.get("accuracy", {}).get("mean")),
            accuracy_std=_as_float(metric_map.get("accuracy", {}).get("std"), default=0.0),
            macro_f1_mean=_as_float(metric_map.get("macro_f1", {}).get("mean")),
            macro_f1_std=_as_float(metric_map.get("macro_f1", {}).get("std"), default=0.0),
            loss_mean=_as_float(metric_map.get("loss", {}).get("mean")),
            loss_std=_as_float(metric_map.get("loss", {}).get("std"), default=0.0),
            seeds="",
            num_seeds=0,
            summary_path=path,
            hidden_size=math.nan,
            attention_dim=math.nan,
        )
    ]


def _load_rows_from_seed_stats(path: Path, source: str) -> List[Dict[str, Any]]:
    rows = _read_csv(path)
    if not rows:
        return []

    grouped: Dict[tuple[str, str], List[Dict[str, str]]] = {}
    run_name_default = path.parent.name
    for row in rows:
        run_name = str(row.get("run_name", run_name_default)).strip() or run_name_default
        variant = str(row.get("model_variant", "unknown")).strip().lower() or "unknown"
        grouped.setdefault((run_name, variant), []).append(row)

    out_rows: List[Dict[str, Any]] = []
    for (run_name, variant), items in grouped.items():
        acc_vals = np.asarray([_as_float(r.get("accuracy")) for r in items], dtype=np.float64)
        f1_vals = np.asarray([_as_float(r.get("macro_f1")) for r in items], dtype=np.float64)
        loss_vals = np.asarray([_as_float(r.get("loss")) for r in items], dtype=np.float64)
        seeds = [str(r.get("seed", "")).strip() for r in items if str(r.get("seed", "")).strip()]
        out_rows.append(
            _standard_row(
                source=source,
                run_name=run_name,
                model_variant=variant,
                accuracy_mean=float(np.nanmean(acc_vals)),
                accuracy_std=float(np.nanstd(acc_vals, ddof=0)),
                macro_f1_mean=float(np.nanmean(f1_vals)),
                macro_f1_std=float(np.nanstd(f1_vals, ddof=0)),
                loss_mean=float(np.nanmean(loss_vals)) if np.isfinite(loss_vals).any() else math.nan,
                loss_std=float(np.nanstd(loss_vals, ddof=0)) if np.isfinite(loss_vals).any() else math.nan,
                seeds=",".join(seeds),
                num_seeds=len(seeds),
                summary_path=path,
                hidden_size=math.nan,
                attention_dim=math.nan,
            )
        )
    return out_rows


def _load_rows_from_summary_csv(path: Path, source: str) -> List[Dict[str, Any]]:
    rows = _read_csv(path)
    if not rows:
        return []

    field_names = {name.strip() for name in rows[0].keys()}

    if {"accuracy_mean", "macro_f1_mean"}.issubset(field_names):
        out_rows: List[Dict[str, Any]] = []
        for row in rows:
            run_name = str(row.get("run_name", path.parent.name)).strip() or path.parent.name
            model_variant = str(row.get("model_variant", row.get("tag", "unknown"))).strip().lower() or "unknown"
            seeds = str(row.get("seeds", "")).strip()
            num_seeds = _as_int(row.get("num_seeds"), default=_count_seeds(seeds))
            out_rows.append(
                _standard_row(
                    source=source,
                    run_name=run_name,
                    model_variant=model_variant,
                    accuracy_mean=_as_float(row.get("accuracy_mean")),
                    accuracy_std=_as_float(row.get("accuracy_std"), default=0.0),
                    macro_f1_mean=_as_float(row.get("macro_f1_mean")),
                    macro_f1_std=_as_float(row.get("macro_f1_std"), default=0.0),
                    loss_mean=_as_float(row.get("loss_mean")),
                    loss_std=_as_float(row.get("loss_std"), default=0.0),
                    seeds=seeds,
                    num_seeds=num_seeds,
                    summary_path=path,
                    hidden_size=_as_float(row.get("hidden_size")),
                    attention_dim=_as_float(row.get("attention_dim")),
                )
            )
        return out_rows

    if {"metric", "mean", "std"}.issubset(field_names):
        return _load_rows_from_metric_csv(path=path, source=source)

    if {"seed", "accuracy", "macro_f1"}.issubset(field_names):
        return _load_rows_from_seed_stats(path=path, source=source)

    raise ValueError(
        f"unsupported summary format: {path}\n"
        "Expected columns with either (accuracy_mean, macro_f1_mean) "
        "or (metric, mean, std) or (seed, accuracy, macro_f1)."
    )


def _resolve_paths(raw: str, default_filename_if_dir: str) -> List[Path]:
    out_paths: List[Path] = []
    for token in _split_arg_list(raw):
        candidate = Path(token).expanduser()
        if any(ch in token for ch in ["*", "?"]):
            out_paths.extend(sorted(Path().glob(token)))
            continue
        if candidate.is_dir():
            possible = candidate / default_filename_if_dir
            if not possible.exists():
                raise FileNotFoundError(f"directory provided but missing {default_filename_if_dir}: {candidate}")
            out_paths.append(possible.resolve())
        else:
            if not candidate.exists():
                raise FileNotFoundError(f"path not found: {candidate}")
            out_paths.append(candidate.resolve())
    if not out_paths:
        raise FileNotFoundError("no summary files resolved")
    return out_paths


def _pick_best_variant_rows(rows: Sequence[Dict[str, Any]]) -> Dict[tuple[str, str], Dict[str, Any]]:
    best: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row["source"]), str(row["model_variant"]))
        prev = best.get(key)
        if prev is None or float(row["accuracy_mean"]) > float(prev["accuracy_mean"]):
            best[key] = row
    return best


def _apply_baseline_deltas(rows: List[Dict[str, Any]]) -> None:
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row["source"]), []).append(row)

    for _, source_rows in by_source.items():
        baseline_candidates = [r for r in source_rows if str(r["model_variant"]) == "baseline"]
        global_baseline = max(baseline_candidates, key=lambda r: float(r["accuracy_mean"])) if baseline_candidates else None

        baseline_by_hidden: Dict[float, Dict[str, Any]] = {}
        for base in baseline_candidates:
            hidden = float(base.get("hidden_size", math.nan))
            if not math.isfinite(hidden):
                continue
            prev = baseline_by_hidden.get(hidden)
            if prev is None or float(base["accuracy_mean"]) > float(prev["accuracy_mean"]):
                baseline_by_hidden[hidden] = base

        for row in source_rows:
            hidden = float(row.get("hidden_size", math.nan))
            baseline = baseline_by_hidden.get(hidden) if math.isfinite(hidden) else None
            if baseline is None:
                baseline = global_baseline

            if baseline is None:
                row["delta_accuracy_vs_baseline"] = math.nan
                row["delta_macro_f1_vs_baseline"] = math.nan
                continue

            row["delta_accuracy_vs_baseline"] = float(row["accuracy_mean"]) - float(baseline["accuracy_mean"])
            row["delta_macro_f1_vs_baseline"] = float(row["macro_f1_mean"]) - float(baseline["macro_f1_mean"])


def _fmt_mean_std(mean_val: float, std_val: float) -> str:
    if not (math.isfinite(mean_val) and math.isfinite(std_val)):
        return "NA"
    return f"{mean_val:.4f} +/- {std_val:.4f}"


def _fmt_delta(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.4f}"


def _sort_variant_names(names: Iterable[str]) -> List[str]:
    raw = sorted(set(str(n) for n in names))
    if "baseline" in raw:
        return ["baseline"] + [n for n in raw if n != "baseline"]
    return raw


def build_markdown(rows: Sequence[Dict[str, Any]]) -> str:
    best = _pick_best_variant_rows(rows)
    variants = _sort_variant_names([row["model_variant"] for row in rows])

    lines: List[str] = []
    lines.append("# Paper-Repro vs Protocol Comparison")
    lines.append("")
    lines.append("## Cross-Setting (Best Run per Variant)")
    lines.append("")
    lines.append(
        "| Variant | Paper Repro Accuracy | Protocol Accuracy | Delta Accuracy (Paper-Protocol) | "
        "Paper Repro Macro-F1 | Protocol Macro-F1 | Delta Macro-F1 (Paper-Protocol) |"
    )
    lines.append("|---|---|---|---|---|---|---|")

    for variant in variants:
        paper = best.get(("paper_repro", variant))
        protocol = best.get(("protocol", variant))

        paper_acc = _fmt_mean_std(float(paper["accuracy_mean"]), float(paper["accuracy_std"])) if paper else "NA"
        protocol_acc = _fmt_mean_std(float(protocol["accuracy_mean"]), float(protocol["accuracy_std"])) if protocol else "NA"
        paper_f1 = _fmt_mean_std(float(paper["macro_f1_mean"]), float(paper["macro_f1_std"])) if paper else "NA"
        protocol_f1 = _fmt_mean_std(float(protocol["macro_f1_mean"]), float(protocol["macro_f1_std"])) if protocol else "NA"

        if paper and protocol:
            delta_acc = _fmt_delta(float(paper["accuracy_mean"]) - float(protocol["accuracy_mean"]))
            delta_f1 = _fmt_delta(float(paper["macro_f1_mean"]) - float(protocol["macro_f1_mean"]))
        else:
            delta_acc = "NA"
            delta_f1 = "NA"

        lines.append(
            f"| {variant} | {paper_acc} | {protocol_acc} | {delta_acc} | {paper_f1} | {protocol_f1} | {delta_f1} |"
        )

    lines.append("")
    lines.append("## Baseline vs Improved (Within Setting)")
    lines.append("")
    lines.append(
        "| Setting | Variant | Accuracy | Delta Accuracy vs Baseline | Macro-F1 | Delta Macro-F1 vs Baseline | Run Name |"
    )
    lines.append("|---|---|---|---|---|---|---|")

    for source in ["paper_repro", "protocol"]:
        subset = [row for row in rows if str(row["source"]) == source]
        subset_best = _pick_best_variant_rows(subset)
        source_variants = _sort_variant_names([row["model_variant"] for row in subset])
        for variant in source_variants:
            row = subset_best.get((source, variant))
            if row is None:
                continue
            lines.append(
                "| "
                f"{source} | "
                f"{variant} | "
                f"{_fmt_mean_std(float(row['accuracy_mean']), float(row['accuracy_std']))} | "
                f"{_fmt_delta(float(row.get('delta_accuracy_vs_baseline', math.nan)))} | "
                f"{_fmt_mean_std(float(row['macro_f1_mean']), float(row['macro_f1_std']))} | "
                f"{_fmt_delta(float(row.get('delta_macro_f1_vs_baseline', math.nan)))} | "
                f"{row['run_name']} |"
            )

    lines.append("")
    lines.append(f"_Generated: {timestamp_now()}_")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare paper-repro summary against protocol summary.")
    parser.add_argument(
        "--paper-summary",
        type=str,
        required=True,
        help="Comma-separated paths to paper summary CSV(s) or run directories containing summary.csv.",
    )
    parser.add_argument(
        "--protocol-summary",
        type=str,
        default="results/protocol/aril_capacity_scaling_summary.csv",
        help="Path to protocol summary CSV (or run dir with summary-compatible CSV).",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="results/paper_repro/comparison/paper_vs_protocol_merged.csv",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="results/paper_repro/comparison/paper_vs_protocol_comparison.md",
    )
    args = parser.parse_args()

    paper_paths = _resolve_paths(args.paper_summary, default_filename_if_dir="summary.csv")
    protocol_paths = _resolve_paths(args.protocol_summary, default_filename_if_dir="summary.csv")

    merged_rows: List[Dict[str, Any]] = []
    for path in paper_paths:
        merged_rows.extend(_load_rows_from_summary_csv(path=path, source="paper_repro"))
    for path in protocol_paths:
        merged_rows.extend(_load_rows_from_summary_csv(path=path, source="protocol"))

    if not merged_rows:
        raise RuntimeError("no rows loaded from provided summaries")

    _apply_baseline_deltas(merged_rows)
    merged_rows = sorted(
        merged_rows,
        key=lambda row: (str(row["source"]), str(row["model_variant"]), str(row["run_name"])),
    )

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_md = Path(args.out_md).expanduser().resolve()
    ensure_dir(out_csv.parent)
    ensure_dir(out_md.parent)

    csv_fields = [
        "source",
        "run_name",
        "model_variant",
        "hidden_size",
        "attention_dim",
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "loss_mean",
        "loss_std",
        "seeds",
        "num_seeds",
        "delta_accuracy_vs_baseline",
        "delta_macro_f1_vs_baseline",
        "summary_path",
    ]
    write_csv(out_csv, merged_rows, fieldnames=csv_fields)

    markdown = build_markdown(merged_rows)
    out_md.write_text(markdown, encoding="utf-8")

    print(f"Merged CSV written: {out_csv}")
    print(f"Markdown report written: {out_md}")


if __name__ == "__main__":
    main()
