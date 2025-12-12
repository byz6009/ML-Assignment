from __future__ import annotations

import pickle
from pathlib import Path
import time  # 添加时间模块

import numpy as np
import torch
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent / "model" / "PDEBench-main"))
from pdebench.models.fno.fno import FNO1d, FNO2d, FNO3d
from pdebench.models.fno.utils import FNODatasetMult, FNODatasetSingle
from pdebench.models.metrics import metrics
from torch import nn

# ======== 添加：内存监控和时间戳函数 ========
def print_gpu_memory(label=""):
    """打印GPU内存使用情况"""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[{timestamp}] {label:30} | 当前: {allocated:6.2f}GB | 保留: {reserved:6.2f}GB | 峰值: {max_allocated:6.2f}GB")
    else:
        print(f"[{timestamp}] {label}: CUDA不可用")

def get_timestamp():
    """获取当前时间戳（毫秒精度）"""
    return time.time()

def format_time_delta(start_time, end_time=None):
    """格式化时间差"""
    if end_time is None:
        end_time = time.time()
    delta_ms = (end_time - start_time) * 1000
    if delta_ms < 1000:
        return f"{delta_ms:.1f}ms"
    else:
        return f"{delta_ms/1000:.2f}s"
# ===================================

# torch.manual_seed(0)
# np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== 添加：初始内存状态 ========
print("=" * 80)
print(f"设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
print_gpu_memory("脚本启动时")
print("=" * 80)
# ===================================


def run_training(
    if_training,
    continue_training,
    num_workers,
    modes,
    width,
    initial_step,
    t_train,
    num_channels,
    batch_size,
    epochs,
    learning_rate,
    scheduler_step,
    scheduler_gamma,
    model_update,
    flnm,
    single_file,
    reduced_resolution,
    reduced_resolution_t,
    reduced_batch,
    plot,
    channel_plot,
    x_min,
    x_max,
    y_min,
    y_max,
    t_min,
    t_max,
    base_path="../data/",
    training_type="autoregressive",
):
    # print(
    #    f"Epochs = {epochs}, learning rate = {learning_rate}, scheduler step = {scheduler_step}, scheduler gamma = {scheduler_gamma}"
    # )

    ################################################################
    # load data
    ################################################################

    if single_file:
        # filename
        model_name = flnm[:-5] + "_FNO"
        # print("FNODatasetSingle")

        # Initialize the dataset and dataloader
        train_data = FNODatasetSingle(
            flnm,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
            initial_step=initial_step,
            saved_folder=base_path,
        )
        val_data = FNODatasetSingle(
            flnm,
            reduced_resolution=reduced_resolution,
            reduced_resolution_t=reduced_resolution_t,
            reduced_batch=reduced_batch,
            initial_step=initial_step,
            if_test=True,
            saved_folder=base_path,
        )

    else:
        # filename
        model_name = flnm + "_FNO"

        # print("FNODatasetMult")
        train_data = FNODatasetMult(
            flnm,
            saved_folder=base_path,
        )
        val_data = FNODatasetMult(
            flnm,
            if_test=True,
            saved_folder=base_path,
        )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    # ======== 添加：数据加载后监控 ========
    print_gpu_memory("数据加载器创建后")
    # ===================================

    ################################################################
    # training and evaluation
    ################################################################

    _, _data, _ = next(iter(val_loader))
    dimensions = len(_data.shape)
    # print("Spatial Dimension", dimensions - 3)
    if dimensions == 4:
        model = FNO1d(
            num_channels=num_channels,
            width=width,
            modes=modes,
            initial_step=initial_step,
        ).to(device)
    elif dimensions == 5:
        model = FNO2d(
            num_channels=num_channels,
            width=width,
            modes1=modes,
            modes2=modes,
            initial_step=initial_step,
        ).to(device)
    elif dimensions == 6:
        model = FNO3d(
            num_channels=num_channels,
            width=width,
            modes1=modes,
            modes2=modes,
            modes3=modes,
            initial_step=initial_step,
        ).to(device)

    # ======== 添加：模型创建后详细监控 ========
    print_gpu_memory("模型创建后")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")
    print(f"模型配置: modes={modes}, width={width}, initial_step={initial_step}")
    # =======================================

    # Set maximum time step of the data to train
    t_train = min(t_train, _data.shape[-2])

    model_path = model_name + ".pt"

    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total parameters = {total_params}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    )

    # ======== 添加：优化器创建后监控 ========
    print_gpu_memory("优化器创建后")
    # =====================================

    loss_fn = nn.MSELoss(reduction="mean")
    loss_val_min = np.inf

    start_epoch = 0

    if not if_training:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        Lx, Ly, Lz = 1.0, 1.0, 1.0
        errs = metrics(
            val_loader,
            model,
            Lx,
            Ly,
            Lz,
            plot,
            channel_plot,
            model_name,
            x_min,
            x_max,
            y_min,
            y_max,
            t_min,
            t_max,
            initial_step=initial_step,
        )
        with Path(model_name + ".pickle").open("wb") as pb:
            pickle.dump(errs, pb)

        return

    # If desired, restore the network by loading the weights saved in the .pt
    # file
    if continue_training:
        # print("Restoring model (that is the network's weights) from file...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.train()

        # Load optimizer state dict
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        start_epoch = checkpoint["epoch"]
        loss_val_min = checkpoint["loss"]

    # ======== 添加：训练开始前监控 ========
    print("\n" + "="*80)
    print("开始训练...")
    print("="*80)
    # ===========================================

    for ep in range(start_epoch, epochs):
        model.train()
        
        # ======== 添加：每个epoch开始监控 ========
        epoch_start_time = get_timestamp()
        print(f"\n[Epoch {ep}/{epochs}]")
        print_gpu_memory(f"Epoch {ep} 开始")
        # =======================================
        
        # t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0
        
        for batch_idx, (xx, yy, grid) in enumerate(train_loader):
            # ======== 添加：每个batch开始监控 ========
            batch_start_time = get_timestamp()
            if batch_idx % 10 == 0:
                print(f"\n  Batch {batch_idx}:")
                print_gpu_memory("    batch开始前")
            # =======================================
            
            loss = 0

            # xx: input tensor (first few time steps) [b, x1, ..., xd, t_init, v]
            # yy: target tensor [b, x1, ..., xd, t, v]
            # grid: meshgrid [b, x1, ..., xd, dims]
            data_transfer_start = get_timestamp()
            xx = xx.to(device)  # noqa: PLW2901
            yy = yy.to(device)  # noqa: PLW2901
            grid = grid.to(device)  # noqa: PLW2901
            data_transfer_time = get_timestamp()

            # ======== 添加：数据转移后监控 ========
            if batch_idx % 10 == 0:
                print_gpu_memory("    数据转移到GPU后")
                print(f"      数据转移耗时: {format_time_delta(data_transfer_start, data_transfer_time)}")
            # ===================================

            # Initialize the prediction tensor
            pred = yy[..., :initial_step, :]
            # Extract shape of the input tensor for reshaping (i.e. stacking the
            # time and channels dimension together)
            inp_shape = list(xx.shape)
            inp_shape = inp_shape[:-2]
            inp_shape.append(-1)

            if training_type in ["autoregressive"]:
                # ======== 添加：autoregressive循环开始监控 ========
                ar_start_time = get_timestamp()
                # ================================================
                
                # Autoregressive loop
                for t in range(initial_step, t_train):
                    # ======== 添加：每个时间步开始监控 ========
                    step_start_time = get_timestamp()
                    if batch_idx % 10 == 0 and t == initial_step:
                        print(f"      开始时间步 t={t}")
                    # ========================================
                    
                    # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                    inp = xx.reshape(inp_shape)

                    # Extract target at current time step
                    y = yy[..., t : t + 1, :]

                    # Model run
                    im = model(inp, grid)

                    # ======== 添加：前向传播后监控 ========
                    forward_time = get_timestamp()
                    if batch_idx % 10 == 0 and t == initial_step:
                        print_gpu_memory(f"      t={t} 前向传播后")
                        print(f"      前向传播耗时: {format_time_delta(step_start_time, forward_time)}")
                    # =====================================

                    # Loss calculation
                    _batch = im.size(0)
                    loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))

                    # Concatenate the prediction at current time step into the
                    # prediction tensor
                    pred = torch.cat((pred, im), -2)

                    # Concatenate the prediction at the current time step to be used
                    # as input for the next time step
                    xx = torch.cat((xx[..., 1:, :], im), dim=-2)  # noqa: PLW2901
                    
                    # ======== 添加：拼接后监控 ========
                    if batch_idx % 10 == 0 and t == initial_step:
                        cat_time = get_timestamp()
                        print_gpu_memory(f"      t={t} 拼接后")
                        print(f"      张量拼接耗时: {format_time_delta(forward_time, cat_time)}")
                    # =================================

                # ======== 添加：autoregressive循环结束监控 ========
                ar_total_time = get_timestamp()
                if batch_idx % 10 == 0:
                    print_gpu_memory("    autoregressive循环后")
                    print(f"      autoregressive总耗时: {format_time_delta(ar_start_time, ar_total_time)}")
                    print(f"      总时间步数: {t_train - initial_step}")
                # ===============================================

                train_l2_step += loss.item()
                _batch = yy.size(0)
                _yy = yy[..., :t_train, :]  # if t_train is not -1
                l2_full = loss_fn(pred.reshape(_batch, -1), _yy.reshape(_batch, -1))
                train_l2_full += l2_full.item()

                backward_start = get_timestamp()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                backward_time = get_timestamp()
                
                # ======== 添加：反向传播后监控 ========
                if batch_idx % 10 == 0:
                    print_gpu_memory("    反向传播后")
                    print(f"      反向传播耗时: {format_time_delta(backward_start, backward_time)}")
                # =====================================

            if training_type in ["single"]:
                x = xx[..., 0, :]
                y = yy[..., t_train - 1 : t_train, :]
                pred = model(x, grid)
                _batch = yy.size(0)
                loss += loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1))

                train_l2_step += loss.item()
                train_l2_full += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # ======== 添加：每个batch结束监控 ========
            batch_end_time = get_timestamp()
            if batch_idx % 10 == 0:
                print_gpu_memory("    batch结束后")
                print(f"      batch总耗时: {format_time_delta(batch_start_time, batch_end_time)}")
            # =====================================

        # ======== 添加：每个epoch训练后监控 ========
        epoch_train_time = get_timestamp()
        print_gpu_memory(f"Epoch {ep} 训练结束后")
        print(f"  Epoch {ep} 训练总耗时: {format_time_delta(epoch_start_time, epoch_train_time)}")
        # =========================================

        if ep % model_update == 0:
            val_l2_step = 0
            val_l2_full = 0
            val_start_time = get_timestamp()
            with torch.no_grad():
                for val_batch_idx, (xx, yy, grid) in enumerate(val_loader):
                    # ======== 添加：验证batch监控 ========
                    if val_batch_idx % 20 == 0:
                        val_batch_start = get_timestamp()
                        print(f"    验证 Batch {val_batch_idx}")
                        print_gpu_memory("      验证batch开始前")
                    # ===================================
                    
                    loss = 0
                    xx = xx.to(device)  # noqa: PLW2901
                    yy = yy.to(device)  # noqa: PLW2901
                    grid = grid.to(device)  # noqa: PLW2901

                    if training_type in ["autoregressive"]:
                        pred = yy[..., :initial_step, :]
                        inp_shape = list(xx.shape)
                        inp_shape = inp_shape[:-2]
                        inp_shape.append(-1)

                        for t in range(initial_step, yy.shape[-2]):
                            inp = xx.reshape(inp_shape)
                            y = yy[..., t : t + 1, :]
                            im = model(inp, grid)
                            _batch = im.size(0)
                            loss += loss_fn(
                                im.reshape(_batch, -1), y.reshape(_batch, -1)
                            )

                            pred = torch.cat((pred, im), -2)

                            xx = torch.cat((xx[..., 1:, :], im), dim=-2)  # noqa: PLW2901

                        val_l2_step += loss.item()
                        _batch = yy.size(0)
                        _pred = pred[..., initial_step:t_train, :]
                        _yy = yy[..., initial_step:t_train, :]
                        val_l2_full += loss_fn(
                            _pred.reshape(_batch, -1), _yy.reshape(_batch, -1)
                        ).item()

                    if training_type in ["single"]:
                        x = xx[..., 0, :]
                        y = yy[..., t_train - 1 : t_train, :]
                        pred = model(x, grid)
                        _batch = yy.size(0)
                        loss += loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1))

                        val_l2_step += loss.item()
                        val_l2_full += loss.item()
                    
                    # ======== 添加：验证batch结束监控 ========
                    if val_batch_idx % 20 == 0:
                        val_batch_end = get_timestamp()
                        print_gpu_memory("      验证batch结束后")
                        print(f"      验证batch耗时: {format_time_delta(val_batch_start, val_batch_end)}")
                    # ===================================

                val_total_time = get_timestamp()
                print(f"  验证总耗时: {format_time_delta(val_start_time, val_total_time)}")

                if val_l2_full < loss_val_min:
                    loss_val_min = val_l2_full
                    torch.save(
                        {
                            "epoch": ep,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss_val_min,
                        },
                        model_path,
                    )
                    
                    # ======== 添加：模型保存后监控 ========
                    print(f"    保存最佳模型，验证loss: {val_l2_full:.6f}")
                    # ===================================

        # t2 = default_timer()
        scheduler.step()
        
        # ======== 添加：每个epoch结束汇总 ========
        epoch_end_time = get_timestamp()
        print(f"\n[Epoch {ep} 完成]")
        print(f"  训练loss: {train_l2_full:.6f}")
        if ep % model_update == 0:
            print(f"  验证loss: {val_l2_full:.6f}")
        print(f"  Epoch总耗时: {format_time_delta(epoch_start_time, epoch_end_time)}")
        print_gpu_memory(f"  Epoch {ep} 最终状态")
        print("-" * 80)
        # =======================================
        
        # print(
        #    "epoch: {0}, loss: {1:.5f}, t2-t1: {2:.5f}, trainL2: {3:.5f}, testL2: {4:.5f}".format(
        #        ep, loss.item(), t2 - t1, train_l2_full, val_l2_full
        #    )
        # )
    
    # ======== 添加：训练完成总结 ========
    print("\n" + "="*80)
    print("训练完成!")
    print_gpu_memory("最终状态")
    if torch.cuda.is_available():
        max_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"峰值显存使用: {max_used:.2f}GB / 80GB ({max_used/80*100:.1f}%)")
    print("="*80)
    # ===================================


if __name__ == "__main__":
    run_training()
    # print("Done.")