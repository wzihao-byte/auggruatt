import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PrunedAttentionGRU import TEMPORAL_POOLING_MODES, prunedAttentionGRU
from dual_domain_model import build_model
from protocol_data import load_dataset_bundle
from protocol_utils import write_csv, write_json


def _count_params(module: Optional[torch.nn.Module], trainable_only: bool = False) -> int:
    if module is None:
        return 0
    if trainable_only:
        return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
    return int(sum(p.numel() for p in module.parameters()))


def _resolve_io_dims(
    dataset: str,
    input_dim: Optional[int],
    output_dim: Optional[int],
    metadata_path: Optional[str],
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


def _parse_poolings(text: str) -> List[str]:
    values = [token.strip().lower() for token in str(text).split(",") if token.strip()]
    if not values:
        raise ValueError("temporal pooling list is empty")
    unknown = [name for name in values if name not in TEMPORAL_POOLING_MODES]
    if unknown:
        raise ValueError(
            f"unsupported pooling mode(s): {unknown}; expected subset of {TEMPORAL_POOLING_MODES}"
        )
    return values


def _audit_single_pooling(
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
        raise TypeError(f"unexpected model type for baseline: {type(model)!r}")
    if model.fc is None:
        raise RuntimeError("baseline classifier head is missing (fc is None)")

    representation_dim = int(model.fc.in_features)
    fc_params_trainable = _count_params(model.fc, trainable_only=True)

    pooling_module: Optional[torch.nn.Module]
    pooling_desc: str
    representation_source: str
    if pooling_name == "attn":
        pooling_module = model.attention
        pooling_desc = "MaskedAttention(query/key/value + context_vector)"
        representation_source = (
            "attention-weighted sum over value-projected GRU states "
            "[B,T,hidden_dim]->[B,T,attention_dim]->[B,attention_dim]"
        )
    elif pooling_name == "mean":
        pooling_module = model.temporal_projection
        pooling_desc = "temporal mean + MaskedLinear(hidden_dim->attention_dim)"
        representation_source = (
            "temporal mean of GRU states then projection "
            "[B,T,hidden_dim]->[B,hidden_dim]->[B,attention_dim]"
        )
    elif pooling_name == "last":
        pooling_module = model.temporal_projection
        pooling_desc = "last timestep + MaskedLinear(hidden_dim->attention_dim)"
        representation_source = (
            "last GRU state then projection "
            "[B,T,hidden_dim]->[B,hidden_dim]->[B,attention_dim]"
        )
    else:
        raise ValueError(f"unsupported pooling mode: {pooling_name}")

    pooling_params_trainable = _count_params(pooling_module, trainable_only=True)
    pooling_params_total = _count_params(pooling_module, trainable_only=False)
    classifier_param_count = int(fc_params_trainable + pooling_params_trainable)
    total_params_in_model = _count_params(model, trainable_only=False)

    notes = (
        f"representation source: {representation_source}; projection: {pooling_desc}; "
        f"classifier_param_count=trainable(fc={fc_params_trainable}+pooling_proj={pooling_params_trainable}); "
        f"pooling module carries {pooling_params_total - pooling_params_trainable} non-trainable mask params."
    )
    return {
        "pooling_name": pooling_name,
        "representation_dim_to_classifier": representation_dim,
        "classifier_param_count": classifier_param_count,
        "total_params_in_model": total_params_in_model,
        "notes": notes,
    }


def run_audit(args: argparse.Namespace) -> Dict[str, Any]:
    poolings = _parse_poolings(args.temporal_poolings)
    input_dim, output_dim = _resolve_io_dims(
        dataset=args.dataset,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        metadata_path=args.metadata_path,
    )
    rows = [
        _audit_single_pooling(
            pooling_name=pooling_name,
            input_dim=input_dim,
            hidden_size=args.hidden_size,
            attention_dim=args.attention_dim,
            output_dim=output_dim,
        )
        for pooling_name in poolings
    ]
    return {
        "dataset": args.dataset,
        "input_dim": int(input_dim),
        "hidden_size": int(args.hidden_size),
        "attention_dim": int(args.attention_dim),
        "output_dim": int(output_dim),
        "classifier_param_count_definition": (
            "trainable params in final fc layer plus pooling-specific projection/attention module"
        ),
        "rows": rows,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit fairness of temporal pooling variants by representation dim and classifier-head capacity."
    )
    parser.add_argument("--dataset", type=str, default="aril")
    parser.add_argument("--metadata-path", type=str, default=None)
    parser.add_argument("--input-dim", type=int, default=None)
    parser.add_argument("--output-dim", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=32)
    parser.add_argument("--temporal-poolings", type=str, default="attn,mean,last")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/protocol/aril_pooling_fair_compare",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    payload = run_audit(args)
    rows = payload["rows"]

    out_dir = (REPO_ROOT / args.out_dir).resolve()
    json_path = out_dir / "pooling_fairness_audit.json"
    csv_path = out_dir / "pooling_fairness_audit.csv"

    write_json(json_path, payload)
    fieldnames = [
        "pooling_name",
        "representation_dim_to_classifier",
        "classifier_param_count",
        "total_params_in_model",
        "notes",
    ]
    write_csv(csv_path, rows, fieldnames=fieldnames)

    print("Pooling fairness audit")
    print(f"dataset={payload['dataset']} input_dim={payload['input_dim']} hidden_size={payload['hidden_size']} attention_dim={payload['attention_dim']} output_dim={payload['output_dim']}")
    print("-" * 72)
    for row in rows:
        print(
            f"{row['pooling_name']:>5} | repr_dim={row['representation_dim_to_classifier']:<4} | "
            f"classifier_params={row['classifier_param_count']:<7} | total_params={row['total_params_in_model']}"
        )
    print("-" * 72)
    print(f"saved: {json_path}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
