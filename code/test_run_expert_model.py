"""
命令行测试脚本：随机抽取 HDF5 数据中的样本，调用 run_expert_model 预测下一帧并与真值对比。

示例：
  /bin/python3 code/test_run_expert_model.py --pde-case Burgers_FNO --data data/1D_Burgers_Sols_Nu1.0.hdf5 --num-samples 3
  python code/test_run_expert_model.py --pde-case DiffSorp_FNO --data data/1D_diff-sorp_NA_NA.hdf5 --num-samples 1
  python code/test_run_expert_model.py --pde-case DiffSorp_Unet --data data/1D_diff-sorp_NA_NA.hdf5 --num-samples 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch

from run_expert_model import (
    PDE_CONFIG_MAP,
    load_config,
    predict_next_frame,
    predict_pinn,
)


def _find_dataset(f: h5py.File) -> np.ndarray:
    """尝试找到主数据集并返回为 ndarray，支持常见的 PDEBench 布局。"""
    keys = list(f.keys())
    if not keys:
        raise ValueError("HDF5 文件中没有任何数据集")

    candidate = None
    if "tensor" in keys:
        candidate = f["tensor"]
    elif "data" in keys:
        candidate = f["data"]
    else:
        candidate = f[keys[0]]
        # 如果是 group，则尝试取其中的第一个数据集
        if isinstance(candidate, h5py.Group):
            sub_keys = list(candidate.keys())
            if not sub_keys:
                raise ValueError(f"Group {keys[0]} 中没有数据")
            candidate = candidate[sub_keys[0]]

    arr = np.array(candidate, dtype=np.float32)
    return arr


def _canonicalize_data(arr: np.ndarray) -> np.ndarray:
    """
    将原始数组整理成 [n, t, x, c] 形状。
    处理常见几种情况：
    - [n, t, x, c] 直接返回
    - [n, t, x] -> 增加通道维
    - [t, x, c] -> 增加 batch 维
    - [t, x] -> 增加 batch 和通道维
    """
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        a, b, c = arr.shape
        if c <= 4:
            # 可能是 [t, x, c] 或 [n, t, c]，根据时间长度判断
            if a < b:
                # 视为 [t, x, c]
                return arr[None, ...]  # [1, t, x, c]
            else:
                # 视为 [n, t, c]，缺少 x 轴，退化成 x=1
                return arr[..., None]  # [n, t, c, 1]
        else:
            # 视为 [n, t, x]，补一个通道维
            return arr[..., None]
    if arr.ndim == 2:
        # [t, x]
        return arr[None, ..., None]  # [1, t, x, 1]
    raise ValueError(f"暂不支持的数据维度: {arr.shape}")


def _find_coords(f: h5py.File, target_len: int | None = None) -> np.ndarray | None:
    """尝试读取 1D x 坐标；若不存在则返回 None。"""
    candidate_keys = ["x-coordinate", "x", "grid/x"]
    for key in candidate_keys:
        if key in f:
            arr = np.array(f[key], dtype=np.float32)
            if arr.ndim == 1 and (target_len is None or arr.shape[0] == target_len):
                return arr

    # 兜底：在根目录找任意 1D 数据集，长度匹配则使用
    for key in f.keys():
        obj = f[key]
        if isinstance(obj, h5py.Dataset):
            arr = np.array(obj, dtype=np.float32)
            if arr.ndim == 1 and (target_len is None or arr.shape[0] == target_len):
                return arr
    return None


def _normalize_pinn_sample_id(
    sample_id: str | int | None, f: h5py.File
) -> str | int:
    if "tensor" in f:
        if sample_id is None:
            raise ValueError("PINN 需要指定样本索引")
        try:
            sample_int = int(sample_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"样本索引无效: {sample_id}") from exc
        if sample_int < 0 or sample_int >= f["tensor"].shape[0]:
            raise ValueError(f"样本索引超出范围: {sample_int}")
        return sample_int

    if sample_id is None:
        raise ValueError("PINN 需要指定样本 seed")
    if isinstance(sample_id, int):
        key = f"{sample_id:04d}"
    else:
        key = str(sample_id)
        if key.isdigit() and len(key) < 4:
            key = key.zfill(4)
    if key not in f:
        raise ValueError(f"样本 seed 不存在: {key}")
    return key


def _load_pinn_data(
    data_path: Path, sample_id: str | int | None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str | int]:
    with h5py.File(data_path, "r") as f:
        sample_id = _normalize_pinn_sample_id(sample_id, f)
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


def _select_time_indices(t_len: int, count: int) -> np.ndarray:
    count = min(count, t_len)
    if count <= 1:
        return np.array([0], dtype=int)
    return np.linspace(0, t_len - 1, num=count, dtype=int)


def _predict_pinn_frame(
    pde_case: str, x_coords: np.ndarray, t_value: float, device: str
) -> np.ndarray:
    coords = np.stack(
        [x_coords, np.full_like(x_coords, t_value, dtype=np.float32)], axis=1
    )
    pred = predict_pinn(pde_case, coords, device=device)
    return pred.reshape(-1)


def _search_pinn_sample(
    data_path: Path,
    pde_case: str,
    device: str,
    topk: int = 5,
    max_samples: int | None = None,
) -> list[tuple[str | int, float]]:
    with h5py.File(data_path, "r") as f:
        if "tensor" in f:
            tensor = f["tensor"]
            x = np.array(f["x-coordinate"], dtype=np.float32)
            t = np.array(f["t-coordinate"], dtype=np.float32)[: tensor.shape[1]]
            x_slice = slice(0, x.shape[0], 1)
            t_idx = _select_time_indices(tensor.shape[1], 100)
            x_sel = x[x_slice]
            t_sel = t[t_idx]
            xx, tt = np.meshgrid(x_sel, t_sel, indexing="ij")
            coords = np.stack([xx.ravel(), tt.ravel()], axis=1)
            pred = predict_pinn(pde_case, coords, device=device).reshape(xx.shape)

            sample_slice = slice(0, max_samples) if max_samples else slice(None)
            data_sub = tensor[sample_slice, t_idx, x_slice]
            data_sub = np.transpose(data_sub, (0, 2, 1))  # [n, x, t]
            diff = data_sub - pred[None, ...]
            print(data_sub[9991])
            print(pred)
            mse = np.mean(diff**2, axis=(1, 2))
            rmse = np.sqrt(mse)
            denom = np.sqrt(np.mean(data_sub**2, axis=(1, 2))) + 1e-12
            nrmse = rmse / denom

            idxs = np.argsort(nrmse)[:topk]
            return [(int(i), float(nrmse[i])) for i in idxs]

        keys = sorted(f.keys())
        if max_samples:
            keys = keys[:max_samples]
        group0 = f[keys[0]]
        x = np.array(group0["grid"]["x"], dtype=np.float32)
        t = np.array(group0["grid"]["t"], dtype=np.float32)
        x_slice = slice(0, x.shape[0], 16)
        t_idx = np.arange(t.shape[0])
        x_sel = x[x_slice]
        t_sel = t[t_idx]
        xx, tt = np.meshgrid(x_sel, t_sel, indexing="ij")
        coords = np.stack([xx.ravel(), tt.ravel()], axis=1)
        pred = predict_pinn(pde_case, coords, device=device).reshape(xx.shape)

        best: list[tuple[float, str]] = []
        for seed in keys:
            group = f[seed]
            data = np.array(group["data"], dtype=np.float32)
            data = data[t_idx, x_slice, 0]
            data = np.transpose(data, (1, 0))  # [x, t]
            diff = data - pred
            mse = np.mean(diff**2)
            rmse = np.sqrt(mse)
            denom = np.sqrt(np.mean(data**2)) + 1e-12
            nrmse = rmse / denom
            if len(best) < topk:
                best.append((nrmse, seed))
                best.sort()
            elif nrmse < best[-1][0]:
                best[-1] = (nrmse, seed)
                best.sort()

        return [(seed, float(err)) for err, seed in best]


def _prepare_sample(
    arr: np.ndarray,
    initial_step: int,
    rr: int,
    rr_t: int,
    rng: np.random.Generator,
):
    """
    从数据中随机抽取一个样本和起始时间。
    会按 reduced_resolution (rr) / reduced_resolution_t (rr_t) 做下采样。
    返回 frames (len=initial_step) 和真值 next_frame。
    """
    if arr.ndim != 4:
        raise ValueError(f"预处理后应为 4 维 [n,t,x,c]，当前 {arr.shape}")

    # [sample, t, x, c]
    n, t_len_full, x_full, c = arr.shape

    # 下采样
    t_indices = np.arange(0, t_len_full, rr_t, dtype=int)
    x_indices = np.arange(0, x_full, rr, dtype=int)
    t_len = len(t_indices)

    if t_len <= initial_step:
        raise ValueError(
            f"时间长度不足，t_len={t_len}，initial_step={initial_step}"
        )

    sample_idx = rng.integers(0, n)
    start_t = rng.integers(0, t_len - initial_step)
    end_t = start_t + initial_step

    # 实际的时间索引映射回原时间步，便于打印
    start_t_raw = t_indices[start_t]
    end_t_raw = t_indices[end_t]

    frames = [
        arr[sample_idx, t_indices[t], x_indices, 0] for t in range(start_t, end_t)
    ]
    target = arr[sample_idx, t_indices[end_t], x_indices, 0]

    return frames, target, sample_idx, start_t_raw, end_t_raw


def _metrics(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float, float]:
    """返回 MSE, nRMSE, max_abs_err。"""
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    mse = float(np.mean((pred - target) ** 2))
    rmse = float(np.sqrt(mse))
    denom = float(np.sqrt(np.mean(target**2)) + 1e-12)
    nrmse = rmse / denom
    max_err = float(np.max(np.abs(pred - target)))
    return mse, nrmse, max_err


def main():
    parser = argparse.ArgumentParser(description="测试 run_expert_model 的预测效果")
    parser.add_argument(
        "--pde-case",
        required=True,
        choices=list(PDE_CONFIG_MAP.keys()),
        help="选择要测试的 PDE/模型组合，如 Burgers_FNO、Burgers_Unet、DiffSorp_FNO 等",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="HDF5 数据路径，默认取 config 中的 filename 并在 data/ 下查找",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="随机抽取的样本数量",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子",
    )
    parser.add_argument(
        "--rr",
        type=int,
        default=None,
        help="覆盖配置中的 reduced_resolution（空间下采样倍率）。不填则使用配置值。",
    )
    parser.add_argument(
        "--rr-t",
        type=int,
        default=None,
        dest="rr_t",
        help="覆盖配置中的 reduced_resolution_t（时间下采样步长）。不填则使用配置值。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="推理设备，例如 cpu 或 cuda:0",
    )
    parser.add_argument(
        "--pinn-sample",
        type=str,
        default=None,
        help="PINN 模型对应的样本索引或 seed（可选，默认使用内置配置）。",
    )
    parser.add_argument(
        "--pinn-search",
        action="store_true",
        help="搜索最匹配的 PINN 样本/seed（会忽略 --pinn-sample）。",
    )
    parser.add_argument(
        "--pinn-search-topk",
        type=int,
        default=5,
        help="PINN 搜索时输出前 K 个结果。",
    )
    parser.add_argument(
        "--pinn-search-max",
        type=int,
        default=None,
        help="PINN 搜索时仅扫描前 N 个样本/seed。",
    )
    args = parser.parse_args()

    spec = PDE_CONFIG_MAP[args.pde_case]
    cfg = load_config(spec.config_file)
    initial_step = cfg.get("initial_step", 2)
    rr = args.rr if args.rr is not None else cfg.get("reduced_resolution", 1)
    rr_t = args.rr_t if args.rr_t is not None else cfg.get("reduced_resolution_t", 1)

    if args.data is not None:
        data_path = Path(args.data)
    else:
        fname = cfg["filename"]
        # 若未包含扩展名，默认补上 .hdf5
        if not Path(fname).suffix:
            fname = fname + ".hdf5"
        data_path = Path("data") / fname

    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {data_path}")

    rng = np.random.default_rng(args.seed)

    if spec.model_family == "PINN":
        if args.pinn_search:
            results = _search_pinn_sample(
                data_path,
                args.pde_case,
                device=args.device,
                topk=args.pinn_search_topk,
                max_samples=args.pinn_search_max,
            )
            print("PINN 匹配结果（topK）：")
            for rank, (sample_id, score) in enumerate(results, start=0):
                print(f"  {rank}. sample={sample_id}, nRMSE={score:.6f}")
            return

        sample_id = args.pinn_sample if args.pinn_sample is not None else spec.pinn_sample
        data, x_coords, t_coords, sample_id = _load_pinn_data(
            data_path, sample_id
        )
        x_indices = np.arange(0, x_coords.shape[0], rr, dtype=int)
        t_indices = np.arange(0, t_coords.shape[0], rr_t, dtype=int)
        if len(x_indices) == 0 or len(t_indices) == 0:
            raise ValueError("下采样导致空数据，请检查 rr/rr_t 设置")

        print(
            f"载入 PINN 数据: sample={sample_id}, data_shape={data.shape}, "
            f"rr={rr}, rr_t={rr_t}"
        )

        mse_list, nrmse_list, max_list = [], [], []
        for i in range(args.num_samples):
            t_idx = int(rng.choice(t_indices))
            # t_idx = 30
            t_val = float(t_coords[t_idx])
            pred = _predict_pinn_frame(
                args.pde_case, x_coords[x_indices], t_val, args.device
            )
            target = data[t_idx, x_indices]

            mse, nrmse, max_err = _metrics(pred, target)
            mse_list.append(mse)
            nrmse_list.append(nrmse)
            max_list.append(max_err)

            print(
                f"[{i+1}/{args.num_samples}] sample={sample_id}, "
                f"t_idx={t_idx} (t={t_val:.4f})"
            )
            print("  pred[:5]  :", np.asarray(pred).reshape(-1)[:5])
            print("  target[:5]:", np.asarray(target).reshape(-1)[:5])
            print(f"  MSE={mse:.4e}, nRMSE={nrmse:.4e}, max|err|={max_err:.4e}")

        print("\n整体平均：")
        print(
            f"  MSE={np.mean(mse_list):.4e}, nRMSE={np.mean(nrmse_list):.4e}, max|err|={np.mean(max_list):.4e}"
        )
        return

    with h5py.File(data_path, "r") as f:
        arr = _find_dataset(f)
        arr = _canonicalize_data(arr)
        coords = _find_coords(f, target_len=arr.shape[2])

    print(
        f"载入数据形状: {arr.shape}, initial_step={initial_step}, "
        f"rr={rr}, rr_t={rr_t}"
    )

    mse_list, nrmse_list, max_list = [], [], []
    for i in range(args.num_samples):
        frames, target, sample_idx, start_raw, end_raw = _prepare_sample(
            arr, initial_step, rr, rr_t, rng
        )
        pred = predict_next_frame(
            args.pde_case, *frames, grid_coords=coords, device=args.device
        )

        mse, nrmse, max_err = _metrics(pred, target)
        mse_list.append(mse)
        nrmse_list.append(nrmse)
        max_list.append(max_err)

        # 打印一小段对比（前 5 个位置）
        print(
            f"[{i+1}/{args.num_samples}] sample={sample_idx}, "
            f"t_raw={start_raw}->{end_raw} (stride={rr_t})"
        )
        print("  pred[:5]  :", np.asarray(pred).reshape(-1)[:5])
        print("  target[:5]:", np.asarray(target).reshape(-1)[:5])
        print(f"  MSE={mse:.4e}, nRMSE={nrmse:.4e}, max|err|={max_err:.4e}")

    print("\n整体平均：")
    print(
        f"  MSE={np.mean(mse_list):.4e}, nRMSE={np.mean(nrmse_list):.4e}, max|err|={np.mean(max_list):.4e}"
    )


if __name__ == "__main__":
    # 避免不必要的 GPU 初始化，如果指定了 cuda 则由 PyTorch 自行处理
    torch.set_grad_enabled(False)
    main()
