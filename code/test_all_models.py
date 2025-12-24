"""
Run a compact evaluation across all supported 1D models and save plots.

It compares predictions against the ground-truth field at a chosen time index
and saves line plots plus a per-PDE nRMSE bar chart.
"""

# 运行方式示例：python3 code/test_all_models.py --device cpu --out-dir code/model_eval_outputs


from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_expert_model import PDE_CONFIG_MAP, load_config, predict_next_frame, predict_pinn

# run_expert_model already extends sys.path so pdebench is importable
from pdebench.models.fno.fno import FNO1d  # type: ignore
from pdebench.models.unet.unet import UNet1d  # type: ignore


@dataclass(frozen=True)
class ModelTask:
    pde_name: str
    pde_case: str
    data_path: Path
    sample_id: str | int
    time_index: int
    rr: int
    rr_t: int


def _normalize_sample_id(sample_id: str | int, f: h5py.File) -> str | int:
    if "tensor" in f:
        sample_int = int(sample_id)
        if sample_int >= f["tensor"].shape[0]:
            raise ValueError(f"Sample index out of range: {sample_int}")
        return sample_int

    if isinstance(sample_id, int):
        key = f"{sample_id:04d}"
    else:
        key = str(sample_id)
        if key.isdigit() and len(key) < 4:
            key = key.zfill(4)
    if key not in f:
        raise ValueError(f"Seed not found in file: {key}")
    return key


def _load_scalar_series(
    data_path: Path, sample_id: str | int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | int]:
    with h5py.File(data_path, "r") as f:
        sample_id = _normalize_sample_id(sample_id, f)
        if "tensor" in f:
            data = np.array(f["tensor"][sample_id], dtype=np.float32)  # [t, x]
            x = np.array(f["x-coordinate"], dtype=np.float32)
            t = np.array(f["t-coordinate"], dtype=np.float32)[: data.shape[0]]
        else:
            group = f[sample_id]
            data = np.array(group["data"], dtype=np.float32)
            if data.ndim == 3:
                data = data[..., 0]
            x = np.array(group["grid"]["x"], dtype=np.float32)
            t = np.array(group["grid"]["t"], dtype=np.float32)
    return data, x, t, sample_id


def _apply_downsample(
    data: np.ndarray, x: np.ndarray, t: np.ndarray, rr: int, rr_t: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_ds = data[::rr_t, ::rr]
    return data_ds, x[::rr], t[::rr_t]


def _select_time_index(time_index: int, t_len: int, rr_t: int, initial_step: int) -> int:
    if t_len <= initial_step:
        return max(0, t_len - 1)
    idx = time_index // max(rr_t, 1)
    idx = min(idx, t_len - 1)
    if idx < initial_step:
        idx = initial_step
    return idx


def _infer_weight_path(cfg: dict[str, object], model_family: str, weights_dir: Path) -> Path:
    filename = Path(str(cfg["filename"])).stem
    exact = weights_dir / f"{filename}_{model_family}.pt"
    if exact.exists():
        return exact
    pattern = f"{filename}_{model_family}*.pt"
    candidates = sorted(weights_dir.glob(pattern))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"Missing weight file: {exact.name} or pattern {pattern} in {weights_dir}"
    )


def _frames_to_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    arrs = []
    for frame in frames:
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        arrs.append(arr)
    stacked = np.stack(arrs, axis=1)  # [x, t_init, 1]
    return torch.from_numpy(stacked).unsqueeze(0)


def _load_fno_unet_model(
    pde_case: str, device: str
) -> tuple[torch.nn.Module, str, int]:
    spec = PDE_CONFIG_MAP[pde_case]
    cfg = load_config(spec.config_file)
    initial_step = int(cfg.get("initial_step", 2))
    if spec.model_family == "FNO":
        model = FNO1d(
            num_channels=int(cfg.get("num_channels", 1)),
            modes=int(cfg.get("modes", 12)),
            width=int(cfg.get("width", 20)),
            initial_step=initial_step,
        )
    elif spec.model_family == "Unet":
        in_channels = int(cfg.get("in_channels", 1)) * initial_step
        out_channels = int(cfg.get("out_channels", 1))
        model = UNet1d(in_channels, out_channels)
    else:
        raise ValueError(f"Unsupported model family for full dataset error: {spec.model_family}")

    weights_dir = Path(__file__).resolve().parent.parent / "assignment" / "pde_model"
    weight_path = _infer_weight_path(cfg, spec.model_family, weights_dir)
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, spec.model_family, initial_step


def _compute_full_dataset_error_curve(
    pde_case: str,
    data_path: Path,
    time_index: int,
    rr: int,
    rr_t: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    model, model_family, initial_step = _load_fno_unet_model(pde_case, device)
    device_t = torch.device(device)

    with h5py.File(data_path, "r") as f:
        if "tensor" in f:
            x = np.array(f["x-coordinate"], dtype=np.float32)
            t = np.array(f["t-coordinate"], dtype=np.float32)
            x_ds = x[::rr]
            t_ds = t[::rr_t]
            t_idx = _select_time_index(time_index, len(t_ds), rr_t, initial_step)
            t_value = float(t_ds[t_idx])
            grid = torch.from_numpy(x_ds).float().view(1, -1, 1).to(device_t)

            total_err = np.zeros_like(x_ds, dtype=np.float64)
            n_samples = f["tensor"].shape[0]
            for i in range(n_samples):
                data = np.array(f["tensor"][i], dtype=np.float32)  # [t, x]
                data_ds = data[::rr_t, ::rr]
                t_idx_i = min(t_idx, data_ds.shape[0] - 1)
                start = max(0, t_idx_i - initial_step)
                t_idx_i = start + initial_step
                if t_idx_i >= data_ds.shape[0]:
                    t_idx_i = data_ds.shape[0] - 1
                    start = max(0, t_idx_i - initial_step)
                frames = [data_ds[start + j] for j in range(initial_step)]
                xx = _frames_to_tensor(frames).to(device_t)
                with torch.no_grad():
                    if model_family == "FNO":
                        inp = xx.reshape(xx.shape[0], xx.shape[1], -1)
                        out = model(inp, grid)
                        pred = out.squeeze(0).squeeze(-2).cpu().numpy().reshape(-1)
                    else:
                        inp = xx.reshape(xx.shape[0], xx.shape[1], -1).permute(0, 2, 1)
                        out = model(inp)
                        pred = out.permute(0, 2, 1).squeeze(0).cpu().numpy().reshape(-1)

                target = data_ds[t_idx_i]
                total_err += np.abs(pred - target)

            mean_err = total_err / max(n_samples, 1)
            return x_ds, mean_err, t_value

        keys = sorted(f.keys())
        group0 = f[keys[0]]
        x = np.array(group0["grid"]["x"], dtype=np.float32)
        t = np.array(group0["grid"]["t"], dtype=np.float32)
        x_ds = x[::rr]
        t_ds = t[::rr_t]
        t_idx = _select_time_index(time_index, len(t_ds), rr_t, initial_step)
        t_value = float(t_ds[t_idx])
        grid = torch.from_numpy(x_ds).float().view(1, -1, 1).to(device_t)

        total_err = np.zeros_like(x_ds, dtype=np.float64)
        for seed in keys:
            group = f[seed]
            data = np.array(group["data"], dtype=np.float32)
            if data.ndim == 3:
                data = data[:, :, 0]
            data_ds = data[::rr_t, ::rr]
            t_idx_i = min(t_idx, data_ds.shape[0] - 1)
            start = max(0, t_idx_i - initial_step)
            t_idx_i = start + initial_step
            if t_idx_i >= data_ds.shape[0]:
                t_idx_i = data_ds.shape[0] - 1
                start = max(0, t_idx_i - initial_step)
            frames = [data_ds[start + j] for j in range(initial_step)]
            xx = _frames_to_tensor(frames).to(device_t)
            with torch.no_grad():
                if model_family == "FNO":
                    inp = xx.reshape(xx.shape[0], xx.shape[1], -1)
                    out = model(inp, grid)
                    pred = out.squeeze(0).squeeze(-2).cpu().numpy().reshape(-1)
                else:
                    inp = xx.reshape(xx.shape[0], xx.shape[1], -1).permute(0, 2, 1)
                    out = model(inp)
                    pred = out.permute(0, 2, 1).squeeze(0).cpu().numpy().reshape(-1)

            target = data_ds[t_idx_i]
            total_err += np.abs(pred - target)

        mean_err = total_err / max(len(keys), 1)
        return x_ds, mean_err, t_value


def _compute_metrics(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    mse = float(np.mean((pred - target) ** 2))
    rmse = float(np.sqrt(mse))
    denom = float(np.sqrt(np.mean(target**2)) + 1e-12)
    nrmse = rmse / denom
    max_err = float(np.max(np.abs(pred - target)))
    return mse, nrmse, max_err


def _predict_fno_unet(
    pde_case: str,
    data: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    time_index: int,
    rr: int,
    rr_t: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    spec = PDE_CONFIG_MAP[pde_case]
    cfg = load_config(spec.config_file)
    initial_step = int(cfg.get("initial_step", 2))

    data_ds, x_ds, t_ds = _apply_downsample(data, x, t, rr, rr_t)

    t_idx = _select_time_index(time_index, data_ds.shape[0], rr_t, initial_step)
    t0 = max(0, t_idx - initial_step)
    t_idx = t0 + initial_step
    if t_idx >= data_ds.shape[0]:
        t_idx = data_ds.shape[0] - 1
        t0 = max(0, t_idx - initial_step)

    frames = [data_ds[t0 + i] for i in range(initial_step)]
    pred = predict_next_frame(pde_case, *frames, grid_coords=x_ds, device=device)
    target = data_ds[t_idx]
    t_value = float(t_ds[t_idx])
    return pred.reshape(-1), target.reshape(-1), t_value


def _predict_pinn(
    pde_case: str,
    data: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    time_index: int,
    rr: int,
    rr_t: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    data_ds, x_ds, t_ds = _apply_downsample(data, x, t, rr, rr_t)
    t_idx = _select_time_index(time_index, data_ds.shape[0], rr_t, 1)
    t_value = float(t_ds[t_idx])
    coords = np.stack([x_ds, np.full_like(x_ds, t_value, dtype=np.float32)], axis=1)
    pred = predict_pinn(pde_case, coords, device=device).reshape(-1)
    target = data_ds[t_idx].reshape(-1)
    return pred, target, t_value


def _plot_field(ax: plt.Axes, x: np.ndarray, target: np.ndarray, pred: np.ndarray, title: str) -> None:
    ax.plot(x, target, label="target", linewidth=1.5)
    ax.plot(x, pred, label="pred", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(True, alpha=0.3)


def _plot_error(ax: plt.Axes, x: np.ndarray, error: np.ndarray, title: str) -> None:
    ax.plot(x, error, color="tab:red", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("abs error")
    ax.grid(True, alpha=0.3)


def _expand_ylim(ax: plt.Axes, values: np.ndarray, factor: float = 1.5) -> None:
    max_val = float(np.max(values)) if values.size else 0.0
    if max_val <= 0:
        max_val = 1.0
    ax.set_ylim(0.0, max_val * factor)


def _select_resolution(pde_name: str, pde_case: str) -> tuple[int, int]:
    spec = PDE_CONFIG_MAP[pde_case]
    if pde_name == "DiffSorp" or spec.model_family == "PINN":
        return 1, 1
    if pde_name in ("Burgers", "Advection") and spec.model_family in ("FNO", "Unet"):
        return 4, 5
    return 1, 1


def _build_tasks(base_dir: Path) -> list[ModelTask]:
    tasks: list[ModelTask] = []
    burgers_time = 50
    advection_time = 50
    diffsorp_time = 50

    tasks.extend(
        [
            ModelTask(
                "Burgers",
                "Burgers_FNO",
                base_dir / "data" / "1D_Burgers_Sols_Nu1.0.hdf5",
                PDE_CONFIG_MAP["Burgers_PINN"].pinn_sample or 0,
                burgers_time,
                *_select_resolution("Burgers", "Burgers_FNO"),
            ),
            ModelTask(
                "Burgers",
                "Burgers_Unet",
                base_dir / "data" / "1D_Burgers_Sols_Nu1.0.hdf5",
                PDE_CONFIG_MAP["Burgers_PINN"].pinn_sample or 0,
                burgers_time,
                *_select_resolution("Burgers", "Burgers_Unet"),
            ),
            ModelTask(
                "Burgers",
                "Burgers_PINN",
                base_dir / "data" / "1D_Burgers_Sols_Nu1.0.hdf5",
                PDE_CONFIG_MAP["Burgers_PINN"].pinn_sample or 0,
                burgers_time,
                *_select_resolution("Burgers", "Burgers_PINN"),
            ),
            ModelTask(
                "Advection",
                "Advection_FNO",
                base_dir / "data" / "1D_Advection_Sols_beta4.0.hdf5",
                PDE_CONFIG_MAP["Advection_PINN"].pinn_sample or 0,
                advection_time,
                *_select_resolution("Advection", "Advection_FNO"),
            ),
            ModelTask(
                "Advection",
                "Advection_Unet",
                base_dir / "data" / "1D_Advection_Sols_beta4.0.hdf5",
                PDE_CONFIG_MAP["Advection_PINN"].pinn_sample or 0,
                advection_time,
                *_select_resolution("Advection", "Advection_Unet"),
            ),
            ModelTask(
                "Advection",
                "Advection_PINN",
                base_dir / "data" / "1D_Advection_Sols_beta4.0.hdf5",
                PDE_CONFIG_MAP["Advection_PINN"].pinn_sample or 0,
                advection_time,
                *_select_resolution("Advection", "Advection_PINN"),
            ),
            ModelTask(
                "DiffSorp",
                "DiffSorp_FNO",
                base_dir / "data" / "1D_diff-sorp_NA_NA.hdf5",
                PDE_CONFIG_MAP["DiffSorp_PINN"].pinn_sample or "0000",
                diffsorp_time,
                *_select_resolution("DiffSorp", "DiffSorp_FNO"),
            ),
            ModelTask(
                "DiffSorp",
                "DiffSorp_Unet",
                base_dir / "data" / "1D_diff-sorp_NA_NA.hdf5",
                PDE_CONFIG_MAP["DiffSorp_PINN"].pinn_sample or "0000",
                diffsorp_time,
                *_select_resolution("DiffSorp", "DiffSorp_Unet"),
            ),
            ModelTask(
                "DiffSorp",
                "DiffSorp_PINN",
                base_dir / "data" / "1D_diff-sorp_NA_NA.hdf5",
                PDE_CONFIG_MAP["DiffSorp_PINN"].pinn_sample or "0000",
                diffsorp_time,
                *_select_resolution("DiffSorp", "DiffSorp_PINN"),
            ),
        ]
    )
    return tasks


def _save_metrics_csv(out_dir: Path, rows: Iterable[dict[str, str | float]]) -> None:
    out_path = out_dir / "metrics.csv"
    headers = ["pde", "model", "sample", "time", "mse", "nrmse", "max_err"]
    with out_path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            values = [str(row[h]) for h in headers]
            f.write(",".join(values) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all 1D PDE models")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument(
        "--out-dir",
        default="code/model_eval_outputs",
        help="Output directory for plots and metrics",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    out_dir = (base_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = _build_tasks(base_dir)
    metrics_rows: list[dict[str, str | float]] = []
    grouped: dict[str, list[dict[str, object]]] = {}

    for task in tasks:
        data, x, t, sample_id = _load_scalar_series(task.data_path, task.sample_id)
        spec = PDE_CONFIG_MAP[task.pde_case]

        if spec.model_family == "PINN":
            pred, target, t_value = _predict_pinn(
                task.pde_case,
                data,
                x,
                t,
                task.time_index,
                task.rr,
                task.rr_t,
                args.device,
            )
        else:
            pred, target, t_value = _predict_fno_unet(
                task.pde_case,
                data,
                x,
                t,
                task.time_index,
                task.rr,
                task.rr_t,
                args.device,
            )

        data_ds, x_ds, _ = _apply_downsample(data, x, t, task.rr, task.rr_t)
        x_plot = x_ds[: pred.shape[0]]

        mse, nrmse, max_err = _compute_metrics(pred, target)
        metrics_rows.append(
            {
                "pde": task.pde_name,
                "model": task.pde_case,
                "sample": sample_id,
                "time": f"{t_value:.4f}",
                "mse": f"{mse:.6e}",
                "nrmse": f"{nrmse:.6e}",
                "max_err": f"{max_err:.6e}",
            }
        )

        if spec.model_family in ("FNO", "Unet"):
            err_x, err_curve, err_time = _compute_full_dataset_error_curve(
                task.pde_case,
                task.data_path,
                task.time_index,
                task.rr,
                task.rr_t,
                args.device,
            )
        else:
            err_x = x_plot
            err_curve = np.abs(pred - target)
            err_time = t_value

        grouped.setdefault(task.pde_name, []).append(
            {
                "task": task,
                "x": x_plot,
                "pred": pred,
                "target": target,
                "error_x": err_x,
                "error_curve": err_curve,
                "error_time": err_time,
                "nrmse": nrmse,
                "mse": mse,
                "max_err": max_err,
                "time": t_value,
                "sample": sample_id,
            }
        )

    _save_metrics_csv(out_dir, metrics_rows)

    for pde_name, results in grouped.items():
        results = sorted(results, key=lambda r: r["task"].pde_case)
        nrows = len(results)
        fig, axes = plt.subplots(nrows, 1, figsize=(8, 2.6 * nrows), sharex=True)
        if nrows == 1:
            axes = [axes]
        for ax, item in zip(axes, results):
            task = item["task"]
            title = (
                f"{task.pde_case} sample={item['sample']} "
                f"t={item['time']:.3f} nRMSE={item['nrmse']:.3e}"
            )
            _plot_field(ax, item["x"], item["target"], item["pred"], title)
            if pde_name == "Burgers":
                ax.set_ylim(-0.4, -0.2)
            ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{pde_name}_fields.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(nrows, 1, figsize=(8, 2.6 * nrows), sharex=True)
        if nrows == 1:
            axes = [axes]
        for ax, item in zip(axes, results):
            task = item["task"]
            title = f"{task.pde_case} abs error (t={item['error_time']:.3f})"
            _plot_error(ax, item["error_x"], item["error_curve"], title)
            _expand_ylim(ax, np.asarray(item["error_curve"]))
        fig.tight_layout()
        fig.savefig(out_dir / f"{pde_name}_errors.png", dpi=150)
        plt.close(fig)

        labels = [item["task"].pde_case for item in results]
        nrmse_vals = [item["nrmse"] for item in results]
        fig, ax = plt.subplots(figsize=(8, 3.6))
        x_pos = np.arange(len(labels))
        ax.bar(x_pos, nrmse_vals, color="tab:blue", alpha=0.8)
        ax.set_title(f"{pde_name} nRMSE")
        ax.set_ylabel("nRMSE")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        _expand_ylim(ax, np.asarray(nrmse_vals))
        fig.tight_layout()
        fig.savefig(out_dir / f"{pde_name}_nrmse.png", dpi=150)
        plt.close(fig)

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
