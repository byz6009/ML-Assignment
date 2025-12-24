"""
运行专家模型：给定前两帧，预测下一帧。

依赖的权重与配置：
- 权重放在 assignment/pde_model 目录下（如 1D_Burgers_Sols_Nu1.0_FNO.pt）。
- 配置复制自原始项目，位于 assignment/pde_model/config/args。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import yaml

# 将 pdebench 加入路径，便于直接引用模型定义
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDEBENCH_ROOT = PROJECT_ROOT / "PDEBench-main"
import sys

sys.path.append(str(PDEBENCH_ROOT))

from pdebench.models.fno.fno import FNO1d  # noqa: E402
from pdebench.models.unet.unet import UNet1d  # noqa: E402


# 配置与权重所在位置（使用 assignment 目录下的副本）
CONFIG_DIR = PROJECT_ROOT / "assignment" / "pde_model" / "config" / "args"
WEIGHTS_DIR = PROJECT_ROOT / "assignment" / "pde_model"
CONFIG_DIFF_SORP = CONFIG_DIR / "config_diff-sorp.yaml"
PINN_BURGERS = WEIGHTS_DIR / "1D_Burgers_Sols_Nu1.0_PINN.pt-5000.pt"
PINN_DIFF_SORP = WEIGHTS_DIR / "1D_diff-sorp_NA_NA_0001.h5_PINN.pt-15000.pt"
PINN_ADVECTION = WEIGHTS_DIR / "1D_Advection_Sols_beta4.0_PINN.pt-5000.pt"


@dataclass
class ModelSpec:
    config_file: Path
    filename_key: str  # 配置里的文件名键，如 filename: 1D_Burgers_Sols_Nu1.0.hdf5
    model_family: str  # "FNO" 或 "Unet"
    pinn_weight: Path | None = None
    pinn_output_relu: bool = False
    pinn_sample: int | str | None = None


# 支持的 PDE 案例与默认模型类型
PDE_CONFIG_MAP: Dict[str, ModelSpec] = {
    # Burgers
    "Burgers_FNO": ModelSpec(CONFIG_DIR / "config_Bgs.yaml", "filename", "FNO"),
    "Burgers_Unet": ModelSpec(CONFIG_DIR / "config_Bgs.yaml", "filename", "Unet"),
    "Burgers_PINN": ModelSpec(
        CONFIG_DIR / "config_Bgs.yaml",
        "filename",
        "PINN",
        pinn_weight=PINN_BURGERS,
        pinn_sample=5187,
    ),
    # Advection
    "Advection_FNO": ModelSpec(CONFIG_DIR / "config_Adv.yaml", "filename", "FNO"),
    "Advection_Unet": ModelSpec(CONFIG_DIR / "config_Adv.yaml", "filename", "Unet"),
    "Advection_PINN": ModelSpec(
        CONFIG_DIR / "config_Adv.yaml",
        "filename",
        "PINN",
        pinn_weight=PINN_ADVECTION,
        pinn_sample=8648,
    ),
    # Diffusion-Sorption 1D
    "DiffSorp_FNO": ModelSpec(CONFIG_DIFF_SORP, "filename", "FNO"),
    "DiffSorp_Unet": ModelSpec(CONFIG_DIFF_SORP, "filename", "Unet"),
    "DiffSorp_PINN": ModelSpec(
        CONFIG_DIFF_SORP,
        "filename",
        "PINN",
        pinn_weight=PINN_DIFF_SORP,
        pinn_output_relu=True,
        pinn_sample="1",
    ),
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"未找到配置文件: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache()
def load_config(config_path: Path) -> Dict[str, Any]:
    return _load_yaml(config_path)


def _infer_weight_path(cfg: Dict[str, Any], model_family: str) -> Path:
    """
    根据配置里的 filename 推断权重名称，支持精确匹配和模糊匹配：
    - 优先 filename_模型族.pt（如 1D_Burgers_Sols_Nu1.0_FNO.pt 或 ..._Unet.pt）
    - 若未找到，尝试 filename_模型族*.pt（如 1D_Burgers_Sols_Nu1.0_Unet-PF-20.pt）
    """
    filename = Path(cfg["filename"]).stem
    exact = WEIGHTS_DIR / f"{filename}_{model_family}.pt"
    if exact.exists():
        return exact

    # 模糊匹配
    pattern = f"{filename}_{model_family}*.pt"
    candidates = sorted(WEIGHTS_DIR.glob(pattern))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"未找到权重文件，尝试过: {exact.name} 或模式 {pattern}，目录 {WEIGHTS_DIR}"
    )


class PINNNet(torch.nn.Module):
    def __init__(self, layer_sizes: Tuple[int, ...], output_relu: bool = False):
        super().__init__()
        self.linears = torch.nn.ModuleList()
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.linears.append(torch.nn.Linear(in_dim, out_dim))
        self.act = torch.nn.Tanh()
        self.output_relu = output_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.linears):
            x = layer(x)
            if idx < len(self.linears) - 1:
                x = self.act(x)
        if self.output_relu:
            x = torch.relu(x)
        return x


def _infer_pinn_layers(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, ...]:
    weight_keys = [
        key for key in state_dict if key.startswith("linears.") and key.endswith(".weight")
    ]
    weight_keys.sort(key=lambda k: int(k.split(".")[1]))
    if not weight_keys:
        raise ValueError("PINN 权重不包含 linears.*.weight")
    layer_sizes = [state_dict[weight_keys[0]].shape[1]]
    for key in weight_keys:
        layer_sizes.append(state_dict[key].shape[0])
    return tuple(layer_sizes)


@lru_cache()
def _load_pinn_model(
    weight_path: str, output_relu: bool, device_str: str
) -> PINNNet:
    checkpoint = torch.load(weight_path, map_location=device_str)
    state_dict = (
        checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    )
    layer_sizes = _infer_pinn_layers(state_dict)
    model = PINNNet(layer_sizes, output_relu=output_relu)
    model.load_state_dict(state_dict)
    model.to(device_str)
    model.eval()
    return model


def _build_grid(
    x_len: int,
    x_min: float = -1.0,
    x_max: float = 1.0,
    coords: np.ndarray | None = None,
) -> torch.Tensor:
    """
    构造 1D 网格坐标，形状 [1, x_len, 1]。
    如果提供 coords，则直接使用 coords（会按需要截取/重排）。
    """
    if coords is not None:
        coords = np.asarray(coords, dtype=np.float32)
        if coords.shape[0] != x_len:
            coords = coords[:x_len]
        grid = torch.from_numpy(coords)
    else:
        grid = torch.linspace(x_min, x_max, steps=x_len, dtype=torch.float32)
    return grid.view(1, x_len, 1)


def _prepare_input(frames: Tuple[np.ndarray, ...]) -> torch.Tensor:
    """
    将多帧堆叠成形状 [1, x, t_init, v]，其中 v=1。
    允许输入形状为 [x] 或 [x, 1]。
    """
    arrs = []
    for idx, fr in enumerate(frames):
        fr = np.asarray(fr, dtype=np.float32)
        if fr.ndim == 1:
            fr = fr[:, None]
        arrs.append(fr)
        if idx > 0 and arrs[-1].shape != arrs[0].shape:
            raise ValueError(f"帧形状不一致: {arrs[0].shape} vs {arrs[-1].shape}")
    stacked = np.stack(arrs, axis=1)  # [x, t_init, 1]
    return torch.from_numpy(stacked).unsqueeze(0)  # [1, x, t_init, 1]


def predict_next_frame(
    pde_case: str,
    *frames: np.ndarray,
    grid_coords: np.ndarray | None = None,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """
    给定前若干帧，使用指定 PDE 案例的权重预测下一帧。

    参数
    ----
    pde_case: str
        例如 "Burgers_FNO"、"Burgers_Unet"、"Advection_FNO" 等（见 PDE_CONFIG_MAP）。
    frames: np.ndarray
        连续的若干帧数据（数量须与配置中的 initial_step 一致），形状 [x] 或 [x, 1]，分辨率应与训练时一致。
    device: "cpu" | torch.device
        推理设备。

    返回
    ----
    np.ndarray
        预测的下一帧，形状 [x, 1]。
    """
    if pde_case not in PDE_CONFIG_MAP:
        raise ValueError(f"不支持的 pde_case: {pde_case}，可选: {list(PDE_CONFIG_MAP)}")

    spec = PDE_CONFIG_MAP[pde_case]
    cfg = load_config(spec.config_file)
    initial_step = cfg.get("initial_step", 2)
    if spec.model_family == "PINN":
        raise ValueError("PINN 需要坐标与时间输入，请使用 predict_pinn")
    if len(frames) != initial_step:
        raise ValueError(
            f"该模型需要 {initial_step} 帧作为输入，当前提供 {len(frames)} 帧"
        )

    # 准备输入与网格
    xx = _prepare_input(tuple(frames))  # [1, x, t_init, 1]
    x_len = xx.shape[1]
    grid = _build_grid(
        x_len,
        x_min=cfg.get("x_min", -1.0),
        x_max=cfg.get("x_max", 1.0),
        coords=grid_coords,
    )

    device = torch.device(device)
    xx = xx.to(device)
    grid = grid.to(device)

    # 构造模型
    if spec.model_family == "FNO":
        model = FNO1d(
            num_channels=cfg.get("num_channels", 1),
            modes=cfg.get("modes", 12),
            width=cfg.get("width", 20),
            initial_step=initial_step,
        )
    elif spec.model_family == "Unet":
        in_channels = cfg.get("in_channels", 1) * initial_step
        out_channels = cfg.get("out_channels", 1)
        model = UNet1d(in_channels, out_channels)
    else:
        raise ValueError(f"未知模型类型: {spec.model_family}")

    # 加载权重
    weight_path = _infer_weight_path(cfg, spec.model_family)
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    with torch.no_grad():
        if spec.model_family == "FNO":
            # FNO 期望输入 [b, x, t_init*v]；这里 t_init=2, v=1 -> [1, x, 2]
            inp = xx.reshape(xx.shape[0], xx.shape[1], -1)
            out = model(inp, grid)  # [1, x, 1, v]
        else:  # Unet
            inp = xx.reshape(xx.shape[0], xx.shape[1], -1).permute(0, 2, 1)  # [1, x, 2*v]
            out = model(inp)  # [1, x, v]
            out = out.permute(0, 2, 1).unsqueeze(-2)  # 对齐成 [1, x, 1, v]

    return out.squeeze(0).squeeze(-2).cpu().numpy()  # [x, v]


def predict_pinn(
    pde_case: str,
    coords: np.ndarray,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """
    PINN 预测：输入坐标与时间，输出对应的解。

    参数
    ----
    coords: np.ndarray
        形状 [n, 2]，每行是 [x, t]。
    """
    if pde_case not in PDE_CONFIG_MAP:
        raise ValueError(f"不支持的 pde_case: {pde_case}，可选: {list(PDE_CONFIG_MAP)}")

    spec = PDE_CONFIG_MAP[pde_case]
    if spec.model_family != "PINN":
        raise ValueError(f"{pde_case} 不是 PINN 模型")
    if spec.pinn_weight is None:
        raise ValueError(f"{pde_case} 未配置 PINN 权重路径")

    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords 应为 [n,2]，当前 {coords.shape}")

    device = torch.device(device)
    model = _load_pinn_model(
        str(spec.pinn_weight), spec.pinn_output_relu, str(device)
    )
    with torch.no_grad():
        out = model(torch.from_numpy(coords).to(device))
    return out.cpu().numpy()


__all__ = ["predict_next_frame", "predict_pinn", "PDE_CONFIG_MAP", "load_config"]
if "__main__" == __name__:
    # 简单测试
    x = np.linspace(-1, 1, 1024, dtype=np.float32)
    f0 = np.sin(np.pi * x)  # t=0
    f1 = np.sin(np.pi * (x - 0.1))  # t=1

    pred = predict_next_frame("Advection_FNO", f0, f1, device="cpu")
    print("预测的下一帧形状:", pred.shape)
    print(pred)
