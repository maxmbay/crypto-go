"""Train every market-only design on one chronological split and score them together.

Train: 2021-01-01 through 2025-06-30.
Validation: 2025-07-01 through 2025-12-31. Early stopping and LightGBM rounds use this only.
Test: 2026-01-01 onward. Reported once, not used to pick architecture.

Run: uv run python -m cryptogo.bakeoff
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from cryptogo.extract import ensure_hourly
from cryptogo.ledger import cross_sectional_ic, sign_hit, simulate
from cryptogo.panel import (
    TEST_START,
    build_dataset,
    discover_hourly_files,
    log_cost,
    select_universe,
    split_masks,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs"
CACHE = OUTPUT / "dataset.npz"
RESULTS = OUTPUT / "bakeoff_results.json"

# Kraken cross-platform spot schedule as published for July 2026.
# Tier 1 is the starter taker. Tier 5 is the $50k-volume / $100k-assets taker.
TIER1_TAKER = 0.0080
TIER5_TAKER = 0.0030
MAX_POSITIONS = 5
SEED = 7
Y_CLIP = 1.0
NEURAL_MODELS = (
    "mlp",
    "nlinear",
    "dlinear",
    "nbeats",
    "tsmixer",
    "tcn",
    "lstm",
    "gru",
    "transformer",
    "transformer_gru",
    "gru_transformer",
    "patchtst",
    "cross_transformer",
    "cross_flow",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Market-only forecast bake-off")
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--attention", action="store_true")
    args = parser.parse_args()
    if args.worker:
        run_worker(args)
    else:
        orchestrate(args)


def orchestrate(args: argparse.Namespace) -> None:
    """One process per model so LightGBM and PyTorch do not share OpenMP."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    load_dataset(rebuild=args.rebuild_data)
    done = load_results()
    finished = {row["model"] for row in done if "error" not in row}
    for name in model_names(args.only):
        if name in finished and not args.only:
            print(f"skip {name}, already in {RESULTS.name}", flush=True)
            continue
        command = [
            sys.executable,
            "-m",
            "cryptogo.bakeoff",
            "--worker",
            "--device",
            args.device,
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--only",
            name,
        ]
        if args.attention:
            command.append("--attention")
        print(f"\n=== {name} ===", flush=True)
        completed = subprocess.run(command)
        if completed.returncode != 0:
            print(f"{name} failed with exit {completed.returncode}", flush=True)
    print("\n" + format_table(load_results()), flush=True)


def run_worker(args: argparse.Namespace) -> None:
    global torch, SEQUENCE_MODELS, CrossAssetTransformer, CrossAssetFlow
    np.random.seed(SEED)
    names = model_names(args.only)
    needs_torch = any(name in NEURAL_MODELS for name in names)
    if needs_torch:
        import torch as torch_mod
        from cryptogo.models import SEQUENCE_MODELS as sequence_models
        from cryptogo.models import CrossAssetFlow as flow_cls
        from cryptogo.models import CrossAssetTransformer as cross_cls

        torch = torch_mod
        SEQUENCE_MODELS = sequence_models
        CrossAssetTransformer = cross_cls
        CrossAssetFlow = flow_cls
        torch.manual_seed(SEED)
        torch.set_num_threads(8)
        device = pick_device(args.device)
    else:
        device = None
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_dataset(rebuild=args.rebuild_data)
    if args.attention:
        from cryptogo.attention import attach_attention, build_feature_matrix

        attention_root = ROOT / "data" / "attention"
        features = build_feature_matrix(data, attention_root)
        data = attach_attention(data, features)
        print(f"attention columns={features.shape[1]} observed_wiki={float(features[:, 2].mean()):.3f}", flush=True)
    bundle = prepare_bundle(data)
    print(f"device={device} models={names}", flush=True)
    print(
        f"samples={data['y'].shape[0]} train={int(bundle['train'].sum())} "
        f"val={int(bundle['val'].sum())} test={int(bundle['test'].sum())} "
        f"assets={', '.join(data['asset_names'].tolist())}",
        flush=True,
    )

    done = load_results()
    finished = {row["model"] for row in done}
    for name in names:
        if name in finished and not args.only:
            print(f"skip {name}, already in {RESULTS.name}", flush=True)
            continue
        print(f"\n=== {name} ===", flush=True)
        try:
            result = run_model(
                name,
                data,
                bundle,
                device,
                args.epochs,
                args.patience,
                label_prefix="attn_" if args.attention else "",
                checkpoint=OUTPUT / ("cross_flow_attn.pt" if args.attention else "cross_flow.pt"),
            )
        except Exception:
            traceback.print_exc()
            if device is not None and device.type != "cpu":
                print(f"{name} failed on {device.type}, retrying on cpu", flush=True)
                try:
                    result = run_model(
                        name,
                        data,
                        bundle,
                        torch.device("cpu"),
                        args.epochs,
                        args.patience,
                        label_prefix="attn_" if args.attention else "",
                        checkpoint=OUTPUT / ("cross_flow_attn.pt" if args.attention else "cross_flow.pt"),
                    )
                except Exception:
                    traceback.print_exc()
                    result = {"model": name, "error": traceback.format_exc(limit=1)}
            else:
                result = {"model": name, "error": traceback.format_exc(limit=1)}
        rows_out = result if isinstance(result, list) else [result]
        for row in rows_out:
            done = [item for item in done if item.get("model") != row.get("model")]
            done.append(row)
            if "error" not in row and "test_ic" in row:
                print(format_row(row), flush=True)
        RESULTS.write_text(json.dumps(done, indent=2))
    print("\n" + format_table(done), flush=True)


def model_names(only: list[str] | None) -> list[str]:
    names = [
        "momentum",
        "reversion",
        "ridge",
        "lightgbm",
        *NEURAL_MODELS,
    ]
    if only:
        unknown = sorted(set(only) - set(names))
        if unknown:
            raise SystemExit(f"unknown model(s): {unknown}")
        return only
    return names


def pick_device(choice: str):
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and choice != "cpu":
        return torch.device("mps")
    return torch.device("cpu")


def load_results() -> list[dict]:
    if not RESULTS.exists():
        return []
    return json.loads(RESULTS.read_text())


def load_dataset(rebuild: bool) -> dict[str, np.ndarray]:
    if CACHE.exists() and not rebuild:
        print(f"loading {CACHE}", flush=True)
        loaded = np.load(CACHE, allow_pickle=False)
        return {key: loaded[key] for key in loaded.files}
    hourly = ensure_hourly()
    grouped = discover_hourly_files(hourly)
    if not grouped:
        raise SystemExit(f"no hourly CSVs under {hourly}")
    print(f"selecting universe from {len(grouped)} pair files", flush=True)
    selection = select_universe(grouped)
    print("universe:", ", ".join(stem for stem, _paths in selection), flush=True)
    data = build_dataset(selection)
    np.savez(CACHE, **data)
    print(f"cached {data['y'].shape[0]} samples -> {CACHE}", flush=True)
    return data


def prepare_bundle(data: dict[str, np.ndarray]) -> dict:
    train, val, test = split_masks(data["ts"])
    if train.sum() == 0 or val.sum() == 0 or test.sum() == 0:
        raise RuntimeError(
            f"empty split train={int(train.sum())} val={int(val.sum())} test={int(test.sum())} "
            f"ts {int(data['ts'].min())}..{int(data['ts'].max())}"
        )
    x_train, x_val, x_test, x_mean, x_std = standardize(data["x"][train], data["x"][val], data["x"][test])
    s_train, s_val, s_test, s_mean, s_std = standardize_static(
        data["static"][train], data["static"][val], data["static"][test]
    )
    return {
        "train": train,
        "val": val,
        "test": test,
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "s_train": s_train,
        "s_val": s_val,
        "s_test": s_test,
        "x_mean": x_mean,
        "x_std": x_std,
        "s_mean": s_mean,
        "s_std": s_std,
    }


def run_model(
    name: str,
    data: dict[str, np.ndarray],
    bundle: dict,
    device: torch.device,
    epochs: int,
    patience: int,
    label_prefix: str = "",
    checkpoint: Path | None = None,
) -> dict:
    train = bundle["train"]
    val = bundle["val"]
    test = bundle["test"]
    y = data["y"]
    y_train = y[train]
    y_val = y[val]
    y_test = y[test]
    context = {
        "x_train": bundle["x_train"],
        "x_val": bundle["x_val"],
        "x_test": bundle["x_test"],
        "s_train": bundle["s_train"],
        "s_val": bundle["s_val"],
        "s_test": bundle["s_test"],
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "raw_test": data["x"][test],
    }
    extras: dict = {}
    med_val = None
    med_test = None
    quantile_preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if name in {"momentum", "reversion"}:
        score = context["raw_test"][:, -24:, 0].sum(axis=1)
        pred_test = score if name == "momentum" else -score
        pred_val = None
        val_loss = float("nan")
    elif name == "ridge":
        pred_val, pred_test, val_loss = fit_ridge(context)
    elif name == "lightgbm":
        pred_val, pred_test, val_loss = fit_lightgbm(context)
    elif name == "cross_transformer":
        pred_val, pred_test, val_loss, extras, _med_val, _med_test, _quantiles, _variance = fit_cross(
            data, bundle, device, epochs, patience, checkpoint=checkpoint
        )
    elif name == "cross_flow":
        pred_val, pred_test, val_loss, extras, med_val, med_test, quantile_preds, _variance = fit_cross(
            data, bundle, device, epochs, patience, objective="nll", checkpoint=checkpoint
        )
    else:
        pred_val, pred_test, val_loss = fit_sequence(name, context, device, epochs, patience)

    row = score_predictions(
        name,
        data["ts"][test],
        data["asset"][test],
        pred_test,
        y_test,
        val_loss,
    )
    if pred_val is not None:
        row["val_ic"] = cross_sectional_ic(data["ts"][val], pred_val, y_val)
    else:
        val_score = data["x"][val][:, -24:, 0].sum(axis=1)
        rule = val_score if name == "momentum" else -val_score
        row["val_ic"] = cross_sectional_ic(data["ts"][val], rule, y_val) if name in {"momentum", "reversion"} else None
    row["train_rows"] = int(train.sum())
    row["val_rows"] = int(val.sum())
    row["test_rows"] = int(test.sum())
    row["test_start"] = int(TEST_START)
    row["assets"] = data["asset_names"].tolist()
    row["benchmarks"] = benchmarks(data)
    row.update(extras)
    row.update(candidate_bias(pred_test, y_test, log_cost(TIER5_TAKER)))
    scored_now = [row]
    if name == "cross_flow" and pred_val is not None:
        finite = np.isfinite(pred_val) & np.isfinite(y_val)
        slope, intercept = np.polyfit(pred_val[finite], y_val[finite], 1)
        cal_val = slope * pred_val + intercept
        cal_test = slope * pred_test + intercept
        cal_row = score_predictions(
            "cross_flow_calibrated",
            data["ts"][test],
            data["asset"][test],
            cal_test,
            y_test,
            val_loss,
        )
        cal_row["val_ic"] = cross_sectional_ic(data["ts"][val], cal_val, y_val)
        cal_row["train_rows"] = row["train_rows"]
        cal_row["val_rows"] = row["val_rows"]
        cal_row["test_rows"] = row["test_rows"]
        cal_row["test_start"] = row["test_start"]
        cal_row["assets"] = row["assets"]
        cal_row["benchmarks"] = row["benchmarks"]
        cal_row["decision"] = "val_affine_mean"
        cal_row["calibration_slope"] = float(slope)
        cal_row["calibration_intercept"] = float(intercept)
        cal_row.update(candidate_bias(cal_test, y_test, log_cost(TIER5_TAKER)))
        scored_now.append(cal_row)
    if med_test is None:
        if label_prefix:
            for item in scored_now:
                item["model"] = label_prefix + item["model"]
        return scored_now[0] if len(scored_now) == 1 else scored_now
    median_row = score_predictions(
        "cross_flow_median",
        data["ts"][test],
        data["asset"][test],
        med_test,
        y_test,
        val_loss,
    )
    if med_val is not None:
        median_row["val_ic"] = cross_sectional_ic(data["ts"][val], med_val, y_val)
    median_row["train_rows"] = row["train_rows"]
    median_row["val_rows"] = row["val_rows"]
    median_row["test_rows"] = row["test_rows"]
    median_row["test_start"] = row["test_start"]
    median_row["assets"] = row["assets"]
    median_row["benchmarks"] = row["benchmarks"]
    median_row["val_nll"] = extras.get("val_nll")
    median_row["test_nll"] = extras.get("test_nll")
    median_row["decision"] = "median"
    median_row.update(candidate_bias(med_test, y_test, log_cost(TIER5_TAKER)))
    scored = [*scored_now, median_row]
    for label, (q_val, q_test) in quantile_preds.items():
        q_row = score_predictions(
            f"cross_flow_{label}",
            data["ts"][test],
            data["asset"][test],
            q_test,
            y_test,
            val_loss,
        )
        q_row["val_ic"] = cross_sectional_ic(data["ts"][val], q_val, y_val)
        q_row["train_rows"] = row["train_rows"]
        q_row["val_rows"] = row["val_rows"]
        q_row["test_rows"] = row["test_rows"]
        q_row["test_start"] = row["test_start"]
        q_row["assets"] = row["assets"]
        q_row["benchmarks"] = row["benchmarks"]
        q_row["val_nll"] = extras.get("val_nll")
        q_row["test_nll"] = extras.get("test_nll")
        q_row["decision"] = label
        q_row.update(candidate_bias(q_test, y_test, log_cost(TIER5_TAKER)))
        scored.append(q_row)
    if label_prefix:
        for row in scored:
            row["model"] = label_prefix + row["model"]
    return scored


def candidate_bias(pred: np.ndarray, y: np.ndarray, hurdle: float) -> dict[str, float]:
    """Mean forecast versus mean outcome wherever the forecast clears the tier-5 hurdle.

    This ignores the five-slot cap, so it is a bias check rather than a ledger.
    """
    mask = np.isfinite(pred) & np.isfinite(y) & (pred > hurdle)
    if not np.any(mask):
        return {"hurdle_candidates": 0.0, "hurdle_mean_pred": float("nan"), "hurdle_mean_y": float("nan")}
    return {
        "hurdle_candidates": float(mask.sum()),
        "hurdle_mean_pred": float(pred[mask].mean()),
        "hurdle_mean_y": float(y[mask].mean()),
    }


def score_predictions(
    name: str,
    ts: np.ndarray,
    asset: np.ndarray,
    pred: np.ndarray,
    y: np.ndarray,
    val_loss: float,
) -> dict:
    finite = np.isfinite(pred) & np.isfinite(y)
    ledgers = {
        "gross": simulate(ts[finite], asset[finite], pred[finite], y[finite], 0.0, MAX_POSITIONS),
        "tier5": simulate(ts[finite], asset[finite], pred[finite], y[finite], TIER5_TAKER, MAX_POSITIONS),
        "tier1": simulate(ts[finite], asset[finite], pred[finite], y[finite], TIER1_TAKER, MAX_POSITIONS),
    }
    return {
        "model": name,
        "val_loss": val_loss,
        "test_ic": cross_sectional_ic(ts[finite], pred[finite], y[finite]),
        "sign_hit": sign_hit(pred[finite], y[finite]),
        "test_mae": float(np.mean(np.abs(pred[finite] - y[finite]))),
        "gross_return": ledgers["gross"]["return"],
        "tier5_return": ledgers["tier5"]["return"],
        "tier1_return": ledgers["tier1"]["return"],
        "gross_trades": ledgers["gross"]["trades"],
        "tier5_trades": ledgers["tier5"]["trades"],
        "tier1_trades": ledgers["tier1"]["trades"],
        "tier5_drawdown": ledgers["tier5"]["max_drawdown"],
        "tier5_mean_net_log": ledgers["tier5"]["mean_net_log"],
        "hurdle_tier5": log_cost(TIER5_TAKER),
        "hurdle_tier1": log_cost(TIER1_TAKER),
    }


def benchmarks(data: dict[str, np.ndarray]) -> dict[str, float]:
    names = data["asset_names"].tolist()
    first = data["hold_first"]
    last = data["hold_last"]
    assets = data["hold_asset"]
    out = {}
    wealths = {0.0: [], TIER5_TAKER: [], TIER1_TAKER: []}
    btc_id = names.index("BTC") if "BTC" in names else None
    for fee, bucket in wealths.items():
        for asset_id, entry, exit_ in zip(assets, first, last, strict=True):
            gross = math.log(float(exit_) / float(entry))
            net = gross - log_cost(fee)
            wealth = math.exp(net)
            bucket.append(wealth)
            if btc_id is not None and int(asset_id) == btc_id:
                label = "btc_gross" if fee == 0 else ("btc_tier5" if fee == TIER5_TAKER else "btc_tier1")
                out[label] = wealth - 1.0
        label = "ew_gross" if fee == 0 else ("ew_tier5" if fee == TIER5_TAKER else "ew_tier1")
        out[label] = float(np.mean(bucket) - 1.0) if bucket else float("nan")
    return out


def fit_ridge(context: dict) -> tuple[np.ndarray, np.ndarray, float]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(market_table(context["x_train"], context["s_train"]))
    x_val = scaler.transform(market_table(context["x_val"], context["s_val"]))
    x_test = scaler.transform(market_table(context["x_test"], context["s_test"]))
    model = Ridge(alpha=10.0)
    model.fit(x_train, np.clip(context["y_train"], -Y_CLIP, Y_CLIP))
    pred_val = model.predict(x_val).astype(np.float32)
    pred_test = model.predict(x_test).astype(np.float32)
    loss = float(np.mean((pred_val - context["y_val"]) ** 2))
    return pred_val, pred_test, loss


def market_table(x: np.ndarray, static: np.ndarray) -> np.ndarray:
    """Lag summaries a tree or linear model can actually split on."""
    ret = x[:, :, 0]
    span = x[:, :, 1]
    log_vol = x[:, :, 2]
    log_trades = x[:, :, 3]
    btc = x[:, :, 4]
    eth = x[:, :, 5]
    columns = [
        ret[:, -1],
        ret[:, -4:].sum(axis=1),
        ret[:, -24:].sum(axis=1),
        ret.sum(axis=1),
        span[:, -1],
        span.mean(axis=1),
        log_vol[:, -1],
        log_vol.mean(axis=1),
        log_trades[:, -1],
        btc[:, -1],
        btc[:, -24:].sum(axis=1),
        eth[:, -1],
        eth[:, -24:].sum(axis=1),
    ]
    return np.column_stack([*columns, static]).astype(np.float32)


def fit_lightgbm(context: dict) -> tuple[np.ndarray, np.ndarray, float]:
    import lightgbm as lgb

    x_train = market_table(context["x_train"], context["s_train"])
    x_val = market_table(context["x_val"], context["s_val"])
    x_test = market_table(context["x_test"], context["s_test"])
    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=SEED,
        verbosity=-1,
    )
    model.fit(
        x_train,
        np.clip(context["y_train"], -Y_CLIP, Y_CLIP),
        eval_X=x_val,
        eval_y=np.clip(context["y_val"], -Y_CLIP, Y_CLIP),
        eval_metric="l2",
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    pred_val = model.predict(x_val).astype(np.float32)
    pred_test = model.predict(x_test).astype(np.float32)
    loss = float(np.mean((pred_val - context["y_val"]) ** 2))
    print(f"  lightgbm trees={model.best_iteration_}", flush=True)
    return pred_val, pred_test, loss


def fit_sequence(
    name: str,
    context: dict,
    device: torch.device,
    epochs: int,
    patience: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    model = SEQUENCE_MODELS[name]().to(device)
    pred_val, pred_test, loss = train_sequence(
        model,
        context["x_train"],
        context["s_train"],
        context["y_train"],
        context["x_val"],
        context["s_val"],
        context["y_val"],
        context["x_test"],
        context["s_test"],
        device,
        epochs,
        patience,
    )
    return pred_val, pred_test, loss


def train_sequence(
    model: torch.nn.Module,
    x_train: np.ndarray,
    s_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    s_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    s_test: np.ndarray,
    device: torch.device,
    epochs: int,
    patience: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    y_fit = np.clip(y_train, -Y_CLIP, Y_CLIP).astype(np.float32)
    best_state = None
    best_loss = math.inf
    stale = 0
    batch = 2048
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(x_train.shape[0])
        total = 0.0
        seen = 0
        for start in range(0, order.shape[0], batch):
            idx = order[start : start + batch]
            xb = torch.from_numpy(x_train[idx]).to(device)
            sb = torch.from_numpy(s_train[idx]).to(device)
            yb = torch.from_numpy(y_fit[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.huber_loss(model(xb, sb), yb, delta=0.2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * idx.shape[0]
            seen += idx.shape[0]
        val_loss = eval_loss(model, x_val, s_val, y_val, device)
        print(
            f"  epoch {epoch} train={total / max(seen, 1):.6f} val={val_loss:.6f}",
            flush=True,
        )
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    pred_val = predict_sequence(model, x_val, s_val, device)
    pred_test = predict_sequence(model, x_test, s_test, device)
    return pred_val, pred_test, best_loss


def eval_loss(
    model: torch.nn.Module,
    x: np.ndarray,
    static: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    seen = 0
    target = np.clip(y, -Y_CLIP, Y_CLIP).astype(np.float32)
    with torch.inference_mode():
        for start in range(0, x.shape[0], 4096):
            xb = torch.from_numpy(x[start : start + 4096]).to(device)
            sb = torch.from_numpy(static[start : start + 4096]).to(device)
            yb = torch.from_numpy(target[start : start + 4096]).to(device)
            loss = torch.nn.functional.huber_loss(model(xb, sb), yb, delta=0.2, reduction="sum")
            total += float(loss)
            seen += xb.shape[0]
    return total / max(seen, 1)


def predict_sequence(
    model: torch.nn.Module,
    x: np.ndarray,
    static: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, x.shape[0], 4096):
            xb = torch.from_numpy(x[start : start + 4096]).to(device)
            sb = torch.from_numpy(static[start : start + 4096]).to(device)
            parts.append(model(xb, sb).detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def fit_cross(
    data: dict[str, np.ndarray],
    bundle: dict,
    device: torch.device,
    epochs: int,
    patience: int,
    objective: str = "huber",
    checkpoint: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """One token per asset. Missing names are padding, not zeros that get a label.

    objective "huber" is the point-forecast head. "nll" trains the conditional spline.
    """
    train = bundle["train"]
    val = bundle["val"]
    test = bundle["test"]
    ts = data["ts"]
    asset = data["asset"].astype(np.int32)
    n_assets = int(data["asset_names"].shape[0])
    times = np.unique(ts)
    time_index = {int(stamp): i for i, stamp in enumerate(times)}
    grid = -np.ones((n_assets, times.shape[0]), dtype=np.int32)
    for row, (stamp, asset_id) in enumerate(zip(ts, asset, strict=True)):
        grid[int(asset_id), time_index[int(stamp)]] = row

    def positions(mask: np.ndarray) -> np.ndarray:
        keep = np.array([time_index[int(stamp)] for stamp in np.unique(ts[mask])], dtype=np.int32)
        return keep

    train_pos = positions(train)
    val_pos = positions(val)
    test_pos = positions(test)
    x_all = ((data["x"] - bundle["x_mean"]) / bundle["x_std"]).astype(np.float32)
    s_all = ((data["static"] - bundle["s_mean"]) / bundle["s_std"]).astype(np.float32)
    y_all = data["y"]

    width = int(data["x"].shape[-1])
    static_dim = int(data["static"].shape[1])
    model = (
        CrossAssetFlow(n_features=width, static_dim=static_dim)
        if objective == "nll"
        else CrossAssetTransformer(n_features=width, static_dim=static_dim)
    ).to(device)
    checkpoint = checkpoint or (OUTPUT / "cross_flow.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_state = None
    best_loss = math.inf
    stale = 0
    batch = 128

    def move(xb, sb, yb, pad):
        return xb.to(device), sb.to(device), yb.to(device), pad.to(device)

    def row_loss(xb, sb, yb, pad_b, reduction: str):
        valid = ~pad_b
        count = int(valid.sum())
        if count == 0:
            return None, 0
        if objective == "nll":
            values = -model.log_prob(xb, sb, pad_b, yb)[valid]
        else:
            pred = model(xb, sb, pad_b)
            target = torch.clamp(yb, -Y_CLIP, Y_CLIP)
            values = torch.nn.functional.huber_loss(pred[valid], target[valid], delta=0.2, reduction="none")
        loss = values.sum() if reduction == "sum" else values.mean()
        return loss, count

    def point_forecast(xb, sb, pad_b):
        if objective == "nll":
            return model.mean(xb, sb, pad_b)
        return model(xb, sb, pad_b)

    def batched_loss(pos: np.ndarray) -> float:
        model.eval()
        total = 0.0
        seen = 0
        with torch.inference_mode():
            for start in range(0, pos.shape[0], batch):
                xb, sb, yb, pad = move(*gather(pos[start : start + batch], grid, x_all, s_all, y_all, n_assets))
                loss, count = row_loss(xb, sb, yb, pad, "sum")
                if loss is None:
                    continue
                total += float(loss)
                seen += count
        return total / max(seen, 1)

    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(train_pos.shape[0])
        for start in range(0, order.shape[0], batch):
            xb, sb, yb, pad = move(
                *gather(train_pos[order[start : start + batch]], grid, x_all, s_all, y_all, n_assets)
            )
            optimizer.zero_grad(set_to_none=True)
            loss, _count = row_loss(xb, sb, yb, pad, "mean")
            if loss is None:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_loss = batched_loss(val_pos)
        print(f"  epoch {epoch} val={val_loss:.6f}", flush=True)
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    elif objective == "nll" and epochs == 0 and checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    quantile_levels = (0.25, 0.40, 0.75) if objective == "nll" else ()

    def predict_split(
        mask: np.ndarray, pos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray], float]:
        pred = np.full(int(mask.sum()), np.nan, dtype=np.float32)
        variance = np.full(int(mask.sum()), np.nan, dtype=np.float32) if objective == "nll" else None
        median = np.full(int(mask.sum()), np.nan, dtype=np.float32) if objective == "nll" else None
        quantile_out = {
            f"q{int(level * 100)}": np.full(int(mask.sum()), np.nan, dtype=np.float32) for level in quantile_levels
        }
        rows = np.flatnonzero(mask)
        locator = {int(row): i for i, row in enumerate(rows)}
        nll_total = 0.0
        nll_seen = 0
        model.eval()
        with torch.inference_mode():
            for start in range(0, pos.shape[0], batch):
                chunk = pos[start : start + batch]
                xb, sb, yb, pad = move(*gather(chunk, grid, x_all, s_all, y_all, n_assets))
                if objective == "nll":
                    mean_t, var_t = model.moments(xb, sb, pad)
                    out = mean_t.cpu().numpy()
                    var_out = var_t.cpu().numpy()
                else:
                    out = point_forecast(xb, sb, pad).cpu().numpy()
                    var_out = None
                med_out = model.median(xb, sb, pad).cpu().numpy() if objective == "nll" else None
                q_outs = {
                    label: model.quantile(xb, sb, pad, level).cpu().numpy()
                    for label, level in zip(quantile_out, quantile_levels, strict=True)
                }
                if objective == "nll":
                    nll, count = row_loss(xb, sb, yb, pad, "sum")
                    if nll is not None:
                        nll_total += float(nll)
                        nll_seen += count
                source = grid[:, chunk]
                for local_asset in range(n_assets):
                    for local_time, row in enumerate(source[local_asset]):
                        if row < 0 or int(row) not in locator:
                            continue
                        slot = locator[int(row)]
                        pred[slot] = out[local_time, local_asset]
                        if variance is not None and var_out is not None:
                            variance[slot] = var_out[local_time, local_asset]
                        if median is not None and med_out is not None:
                            median[slot] = med_out[local_time, local_asset]
                        for label, values in q_outs.items():
                            quantile_out[label][slot] = values[local_time, local_asset]
        nll_mean = nll_total / nll_seen if nll_seen else float("nan")
        return pred, median, quantile_out, variance, nll_mean

    val_pred, val_med, val_q, val_var, val_nll = predict_split(val, val_pos)
    test_pred, test_med, test_q, test_var, test_nll = predict_split(test, test_pos)
    paired_q = {label: (val_q[label], test_q[label]) for label in test_q}
    flow_stats = {"val_nll": val_nll, "test_nll": test_nll, "decision": "mean"} if objective == "nll" else {}
    if objective == "nll" and test_var is not None:
        std = np.sqrt(np.clip(test_var, 0, None))
        finite_std = std[np.isfinite(std)]
        flow_stats["test_std_p50"] = float(np.median(finite_std)) if finite_std.size else float("nan")
        if "q75" in test_q and "q25" in test_q:
            iqr = test_q["q75"] - test_q["q25"]
            finite_iqr = iqr[np.isfinite(iqr)]
            flow_stats["test_iqr_p50"] = float(np.median(finite_iqr)) if finite_iqr.size else float("nan")
    if objective == "nll" and best_state is not None:
        torch.save(best_state, checkpoint)
    return val_pred, test_pred, best_loss, flow_stats, val_med, test_med, paired_q, (val_var, test_var)


def gather(
    pos: np.ndarray,
    grid: np.ndarray,
    x_all: np.ndarray,
    s_all: np.ndarray,
    y_all: np.ndarray,
    n_assets: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = grid[:, pos]  # (assets, batch)
    missing = rows < 0
    safe = rows.copy()
    safe[missing] = 0
    xb = x_all[safe]  # (assets, batch, L, F)
    sb = s_all[safe]
    yb = y_all[safe].astype(np.float32)
    xb = np.transpose(xb, (1, 0, 2, 3))
    sb = np.transpose(sb, (1, 0, 2))
    yb = np.transpose(yb, (1, 0))
    missing = np.transpose(missing, (1, 0))
    xb[missing] = 0
    sb[missing] = 0
    yb[missing] = 0
    return (
        torch.from_numpy(np.ascontiguousarray(xb)),
        torch.from_numpy(np.ascontiguousarray(sb)),
        torch.from_numpy(np.ascontiguousarray(yb)),
        torch.from_numpy(np.ascontiguousarray(missing)),
    )


def standardize(
    train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = train.std(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    scaled = tuple(((arr - mean) / std).astype(np.float32) for arr in (train, *others))
    return (*scaled, mean, std)


def standardize_static(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0, keepdims=True).astype(np.float32)
    std = train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    scaled = tuple(((arr - mean) / std).astype(np.float32) for arr in (train, *others))
    return (*scaled, mean, std)


def flatten(x: np.ndarray, static: np.ndarray) -> np.ndarray:
    return np.concatenate([x.reshape(x.shape[0], -1), static], axis=1).astype(np.float32)


def format_row(row: dict) -> str:
    nll = f" nll={row['test_nll']:.3f}" if "test_nll" in row else ""
    return (
        f"{row['model']:<20} ic={row['test_ic']:+.4f} hit={row['sign_hit']:.3f} "
        f"gross={row['gross_return']:+.1%} tier5={row['tier5_return']:+.1%} "
        f"tier1={row['tier1_return']:+.1%} trades={int(row['tier5_trades'])}{nll}"
    )


def format_table(rows: list[dict]) -> str:
    usable = [row for row in rows if "error" not in row and "test_ic" in row]
    usable.sort(key=lambda row: row["test_ic"] if row["test_ic"] == row["test_ic"] else -999, reverse=True)
    header = f"{'model':<20} {'val_ic':>8} {'test_ic':>8} {'hit':>6} {'gross':>8} {'tier5':>8} {'tier1':>8} {'trades':>7}"
    lines = [header, "-" * len(header)]
    for row in usable:
        val_ic = row.get("val_ic")
        val_text = f"{val_ic:+.4f}" if isinstance(val_ic, float) else "  n/a"
        lines.append(
            f"{row['model']:<20} {val_text:>8} {row['test_ic']:+8.4f} {row['sign_hit']:6.3f} "
            f"{row['gross_return']:+8.1%} {row['tier5_return']:+8.1%} {row['tier1_return']:+8.1%} "
            f"{int(row['tier5_trades']):7d}"
        )
    if usable:
        bench = usable[-1].get("benchmarks") or {}
        lines.append("")
        lines.append(
            "buy-hold over the test window: "
            f"BTC gross {bench.get('btc_gross', float('nan')):+.1%}, "
            f"BTC tier5 {bench.get('btc_tier5', float('nan')):+.1%}, "
            f"equal-weight gross {bench.get('ew_gross', float('nan')):+.1%}, "
            f"equal-weight tier5 {bench.get('ew_tier5', float('nan')):+.1%}"
        )
    failed = [row["model"] for row in rows if "error" in row]
    if failed:
        lines.append("failed: " + ", ".join(failed))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
