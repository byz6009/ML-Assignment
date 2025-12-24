# train_burgers_lora_mlp_run4_e50.py
import os
import json
import time
import math
import functools
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# 配置区（你最常改的都在这里）
# -----------------------------
BASE_MODEL_PATH = "/lustre/home/2300011093/PiERN/model/Qwen2.5-7B-Instruct"
DATA_PATH = "../data/burgers_text2computation_full_final.jsonl"

MAX_LENGTH = 5312
LABEL_DIM = 512

# 你要 4 卡，先给一个“有点激进但通常可跑”的起点：
BATCH_SIZE = 10           # per GPU batch
GRAD_ACCUM_STEPS = 2      # 梯度累积（有效全局 batch = BATCH_SIZE * world_size * GRAD_ACCUM）

LR = 1e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

EPOCHS = 50
LOG_EVERY_UPDATES = 20    # 每多少个 optimizer update 打一次日志（更稳定）
SAVE_EVERY_UPDATES = 250  # 每多少个 optimizer update 存一次 last（≈每 epoch 一次，取决于 steps_per_epoch）
EVAL_EVERY_EPOCH = 5      # 每多少个 epoch 做一次 eval（只 rank0）
EVAL_N = 1000

# ====== 新 run（不会覆盖旧文件）======
RUN_NAME = "lora_mlp_run4_e50_b10_ga1_se250"
SAVE_DIR = f"../model_runs/{RUN_NAME}"
os.makedirs(SAVE_DIR, exist_ok=True)

HEAD_BEST = os.path.join(SAVE_DIR, f"head_best_{RUN_NAME}.pt")
HEAD_LAST = os.path.join(SAVE_DIR, f"head_last_{RUN_NAME}.pt")
LORA_BEST = os.path.join(SAVE_DIR, f"lora_best_{RUN_NAME}.pt")
LORA_LAST = os.path.join(SAVE_DIR, f"lora_last_{RUN_NAME}.pt")

# ====== 关键：初始化 / 续训加载顺序 ======
# 1) 如果你“同一个 RUN_NAME”里已经有 last/best，就优先从本 run 续训
# 2) 如果本 run 还没有 checkpoint，则用你上一轮训练得到的 last 作为初始化（不会覆盖旧 run）
INIT_HEAD_PATH = "../model_runs/lora_mlp_run_b8_ga2_e5_se500/head_last_lora_mlp_run_b8_ga2_e5_se500.pt"
INIT_LORA_PATH = "../model_runs/lora_mlp_run_b8_ga2_e5_se500/lora_last_lora_mlp_run_b8_ga2_e5_se500.pt"

# LoRA 超参（带 MLP）
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# 目标模块：注意力 + MLP
TARGET_MODULE_KEYS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# 更快一点（可选）
torch.backends.cuda.matmul.allow_tf32 = True


# -----------------------------
# Burgers prompt wrap（你之前那段 system）
# -----------------------------
BURGERS_SYSTEM_TEXT = """你是一名结合语言理解能力与物的理建模能力的智能助手，能够根据用户输入灵活切换任务模式。  
当用户提出的是Burgers'方程的计算预测相关的问题，例如：“请根据Burgers'方程求解整个空间域上的速度场数值解”，你必须严格遵守以下规则：
1.回答开始时只能输出<自然语言>+<数值计算结果>两个部分
2.<数值计算结果>部分不需要你进行计算，使用<数值计算结果>代替这个部分。
3.回答示例1：“根据Burgers'方程的数值求解结果，在整个空间域内的速度场数值解为<数值计算结果>”
回答示例2：“好的，我已经明白了。经过计算，当前时间帧上全域速度场数值解是<数值计算结果>”
类似的答案，可以不与回答示例完全一致但意思要相同。
自然语言部分请你像真实系统里那样自然回答用户（以下内容不必都使用）：
-可以选择简单回答用户的请求，比如“好的，我已经明白了。”但不要只限于这句话，如果选择这一点放在文中，必须放在最开头。
-可以选择说明一下答案的来源，比如“根据提供的数据，预测结果如下：”
-可以应用用户输入的问题中的内容来充实你的回答，但不要超过1句。
4.如果用户提出的是普通语言对话（如“你是谁”或“你好”），则按普通对话正常回答"""


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def setup_distributed() -> Tuple[bool, int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed(distributed: bool) -> None:
    if distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# -----------------------------
# Dataset
# -----------------------------
def wrap_prompt(system_text: str, user_text: str) -> str:
    # Qwen2.5 instruct 的 chat 格式（你之前验证过的）
    return (
        "<|im_start|>system\n\n" + system_text.strip() + "<|im_end|>\n"
        "<|im_start|>user\n\n" + user_text.strip() + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


class BurgersDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int, label_dim: int):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_dim = label_dim

        self.offsets = []
        with open(path, "rb") as f:
            off = 0
            for line in f:
                self.offsets.append(off)
                off += len(line)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        with open(self.path, "r", encoding="utf-8") as f:
            f.seek(self.offsets[idx])
            row = json.loads(f.readline())

        # 你数据里：prompt 是文本；label 是两个 256 拼成 512（你已经构造好）
        user_text = row["prompt"]
        label = row["label"]

        if len(label) != self.label_dim:
            raise ValueError(f"label_dim mismatch: expect {self.label_dim}, got {len(label)} at idx={idx}")

        input_str = wrap_prompt(BURGERS_SYSTEM_TEXT, user_text)

        enc = self.tokenizer(
            input_str,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        labels = torch.tensor(label, dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# -----------------------------
# LoRA 实现（只存/载 LoRA 权重）
# -----------------------------
class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        self.dropout = nn.Dropout(dropout)

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # LoRA A/B
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))

        # init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # base 不训练（外面也会整体 freeze）
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base
        out = self.base(x)
        # lora
        lora = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return out + lora * self.scaling


def inject_lora(model: nn.Module, target_keys: List[str], r: int, alpha: int, dropout: float, rank: int) -> int:
    """
    把命中的 nn.Linear 替换成 LoRALinear
    返回注入的模块数
    """
    injected = 0
    for name, module in list(model.named_modules()):
        # 只替换目标 keys 的线性层
        if not any(name.endswith(k) for k in target_keys):
            continue
        if isinstance(module, nn.Linear):
            # 找到父模块并替换属性
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            last = parts[-1]
            setattr(parent, last, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
            injected += 1
    if rank == 0:
        print(f"Injected LoRA into {injected} Linear modules. (r={r}, alpha={alpha}, dropout={dropout})")
    return injected


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    只导出 LoRA 参数：*.lora_A / *.lora_B
    """
    sd = {}
    for k, v in model.state_dict().items():
        if k.endswith("lora_A") or k.endswith("lora_B"):
            sd[k] = v.detach().cpu()
    return sd


def load_lora_state_dict(model: nn.Module, path: str, rank: int, tag: str) -> bool:
    if not path or (not os.path.exists(path)):
        return False
    st = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(st, strict=False)
    if rank == 0:
        print(f"Loaded LoRA ({tag}) from {path}")
        print(f"LoRA load strict=False done. missing={len(missing)} unexpected={len(unexpected)}")
    return True


# -----------------------------
# Head（带 MLP）
# -----------------------------
class RegressionHead(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_head(head: nn.Module, path: str, rank: int, tag: str) -> bool:
    if not path or (not os.path.exists(path)):
        return False
    st = torch.load(path, map_location="cpu")

    # 兼容老 head-only 的 key（"0.weight" 这种）-> 新版 "net.0.weight"
    if any(k.startswith("0.") or k.startswith("2.") for k in st.keys()):
        st = {f"net.{k}": v for k, v in st.items()}

    head.load_state_dict(st, strict=True)
    if rank == 0:
        print(f"Loaded HEAD ({tag}) from {path}")
    return True


# -----------------------------
# checkpointing：避免 DDP “mark ready twice”
# -----------------------------
def enable_non_reentrant_checkpointing(base_model: nn.Module, rank: int) -> None:
    """
    解决 DDP + reentrant checkpoint 的 "mark ready twice" 问题：
    优先用 transformers 支持的 use_reentrant=False；不支持则 monkeypatch torch.utils.checkpoint.checkpoint。
    """
    try:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if rank == 0:
            print("Enabled gradient checkpointing (use_reentrant=False via transformers kw).")
        return
    except TypeError:
        import torch.utils.checkpoint as ckpt
        ckpt.checkpoint = functools.partial(ckpt.checkpoint, use_reentrant=False)
        base_model.gradient_checkpointing_enable()
        if rank == 0:
            print("Enabled gradient checkpointing (use_reentrant=False via monkeypatch).")
        return


# -----------------------------
# Eval（只 rank0）
# -----------------------------
@torch.no_grad()
def eval_n(model: nn.Module, loader: DataLoader, device: torch.device, n: int, rank: int) -> Tuple[float, float]:
    model.eval()
    cnt = 0
    mse_sum = 0.0
    mae_sum = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        preds = model(input_ids, attention_mask)
        diff = preds.float() - labels.float()
        mse = (diff * diff).mean(dim=1)  # [B]
        mae = diff.abs().mean(dim=1)     # [B]

        bsz = input_ids.size(0)
        for i in range(bsz):
            mse_sum += float(mse[i].item())
            mae_sum += float(mae[i].item())
            cnt += 1
            if cnt >= n:
                break
        if cnt >= n:
            break

    if cnt == 0:
        return 0.0, 0.0
    return mse_sum / cnt, mae_sum / cnt


# -----------------------------
# 主训练
# -----------------------------
def main():
    distributed, rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    eff_global_batch = BATCH_SIZE * world_size * GRAD_ACCUM_STEPS

    if rank == 0:
        print("distributed =", distributed, "world_size =", world_size, "local_rank =", local_rank)
        print("RUN_NAME =", RUN_NAME)
        print("SAVE_DIR =", SAVE_DIR)
        print("BASE_MODEL_PATH =", BASE_MODEL_PATH)
        print("DATA_PATH =", DATA_PATH)
        print(f"MAX_LENGTH={MAX_LENGTH} LABEL_DIM={LABEL_DIM} BATCH_SIZE(per_gpu)={BATCH_SIZE} "
              f"GRAD_ACCUM={GRAD_ACCUM_STEPS} effective_global_batch={eff_global_batch}")
        print(f"EPOCHS={EPOCHS} LR={LR} WD={WEIGHT_DECAY} LOG_EVERY_UPDATES={LOG_EVERY_UPDATES} SAVE_EVERY_UPDATES={SAVE_EVERY_UPDATES}")
        print(f"LoRA targets={TARGET_MODULE_KEYS}")
        print(f"INIT_HEAD_PATH = {INIT_HEAD_PATH} exists={os.path.exists(INIT_HEAD_PATH)}")
        print(f"INIT_LORA_PATH = {INIT_LORA_PATH} exists={os.path.exists(INIT_LORA_PATH)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, use_fast=False)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    base.config.use_cache = False

    enable_non_reentrant_checkpointing(base, rank)

    # 让 checkpoint 有梯度入口
    try:
        base.enable_input_require_grads()
        if rank == 0:
            print("Enabled input require grads for checkpointing.")
    except Exception:
        emb = base.get_input_embeddings()
        emb.register_forward_hook(lambda m, inp, out: out.requires_grad_(True))
        if rank == 0:
            print("Enabled input require grads via embedding forward hook.")

    # 冻结 base 原始参数
    for p in base.parameters():
        p.requires_grad = False

    # 注入 LoRA（产生可训练参数）
    inject_lora(base, TARGET_MODULE_KEYS, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT, rank=rank)

    head = RegressionHead(hidden_size=base.config.hidden_size, out_dim=LABEL_DIM).to(device)

    dataset = BurgersDataset(DATA_PATH, tokenizer, max_length=MAX_LENGTH, label_dim=LABEL_DIM)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    steps_per_epoch = len(loader)
    if rank == 0:
        print("steps_per_epoch =", steps_per_epoch)

    # ========== resume 逻辑 ==========
    # LoRA：优先本 run 的 last/best；否则用 INIT_LORA_PATH
    resumed_lora = (
        load_lora_state_dict(base, LORA_LAST, rank, "RUN_LAST")
        or load_lora_state_dict(base, LORA_BEST, rank, "RUN_BEST")
        or load_lora_state_dict(base, INIT_LORA_PATH, rank, "INIT_LORA")
    )
    if rank == 0 and not resumed_lora:
        print("WARNING: No LoRA checkpoint loaded. Start LoRA from init params.")

    # Head：优先本 run 的 last/best；否则用 INIT_HEAD_PATH
    resumed_head = (
        load_head(head, HEAD_LAST, rank, "RUN_LAST")
        or load_head(head, HEAD_BEST, rank, "RUN_BEST")
        or load_head(head, INIT_HEAD_PATH, rank, "INIT_HEAD")
    )
    if rank == 0 and not resumed_head:
        print("WARNING: No HEAD checkpoint loaded. Start head from init params.")

    # 只优化 LoRA + head
    trainable_params = [p for p in list(base.parameters()) + list(head.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

    loss_fn = nn.MSELoss()

    class FullModel(nn.Module):
        def __init__(self, base_model, head_model):
            super().__init__()
            self.base = base_model
            self.head = head_model

        def forward(self, input_ids, attention_mask):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                out = self.base.model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden = out.last_hidden_state  # [B,T,H] bf16

            mask = attention_mask.unsqueeze(-1)  # [B,T,1]
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = pooled.float()
            return self.head(pooled)

    model = FullModel(base, head).to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        try:
            model._set_static_graph()
            if rank == 0:
                print("DDP set static graph.")
        except Exception:
            if rank == 0:
                print("DDP static graph not available (ignored).")

    if rank == 0:
        params_iter = model.module.parameters() if distributed else model.parameters()
        trainable = sum(p.numel() for p in params_iter if p.requires_grad)
        params_iter2 = model.module.parameters() if distributed else model.parameters()
        total = sum(p.numel() for p in params_iter2)
        print("trainable/total =", trainable, "/", total)

    def save_all(tag: str, is_best: bool, extra: str = ""):
        if rank != 0:
            return
        head_path = HEAD_BEST if is_best else HEAD_LAST
        lora_path = LORA_BEST if is_best else LORA_LAST

        base_to_save = model.module.base if distributed else model.base
        head_to_save = model.module.head if distributed else model.head

        torch.save(head_to_save.state_dict(), head_path)
        torch.save(lora_state_dict(base_to_save), lora_path)
        print(f"[{now_str()}] Saved {tag}: head={head_path}, lora={lora_path} {extra}".rstrip())

    best_loss = float("inf")
    micro_step = 0
    update_step = 0
    t_log = time.time()
    optimizer.zero_grad(set_to_none=True)

    # 为了 eval 方便，复用 loader（只 rank0 用）
    eval_loader = loader

    try:
        for epoch in range(EPOCHS):
            if distributed:
                sampler.set_epoch(epoch)

            model.train()
            for step_in_epoch, batch in enumerate(loader, start=1):
                micro_step += 1

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                preds = model(input_ids, attention_mask)
                loss = loss_fn(preds, labels) / GRAD_ACCUM_STEPS
                loss.backward()

                if micro_step % GRAD_ACCUM_STEPS == 0:
                    if MAX_GRAD_NORM is not None and MAX_GRAD_NORM > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    update_step += 1

                    raw_loss = loss.item() * GRAD_ACCUM_STEPS

                    # 日志：按 update_step 打
                    if rank == 0 and (update_step % LOG_EVERY_UPDATES == 0):
                        dt = time.time() - t_log
                        speed = LOG_EVERY_UPDATES / max(dt, 1e-6)
                        t_log = time.time()
                        print(
                            f"[{now_str()}] [epoch {epoch+1}/{EPOCHS} "
                            f"step {step_in_epoch}/{steps_per_epoch} "
                            f"micro_step {micro_step} update_step {update_step}] "
                            f"loss={raw_loss:.6f} speed={speed:.3f} update/s"
                        )
                        print("pred[:5] :", preds[0, :5].detach().float().cpu().tolist())
                        print("true[:5] :", labels[0, :5].detach().float().cpu().tolist())

                    # best：用 raw_loss 判别
                    if rank == 0 and raw_loss < best_loss:
                        best_loss = raw_loss
                        save_all("BEST", True, extra=f"(best_loss={best_loss:.6f}, update_step={update_step})")

                    # last：按 update_step 存
                    if rank == 0 and (update_step % SAVE_EVERY_UPDATES == 0):
                        save_all("LAST", False, extra=f"(update_step={update_step})")

            # 每隔若干 epoch 做一次 eval（rank0）
            if rank == 0 and (EVAL_EVERY_EPOCH > 0) and ((epoch + 1) % EVAL_EVERY_EPOCH == 0):
                mse, mae = eval_n(model.module if distributed else model, eval_loader, device, n=EVAL_N, rank=rank)
                print(f"[{now_str()}] ===== EVAL @ epoch {epoch+1} ===== n={EVAL_N} avg_mse={mse:.6f} avg_mae={mae:.6f}")

        if rank == 0:
            print(f"[{now_str()}] Training finished. best_loss={best_loss:.6f} update_step={update_step} micro_step={micro_step}")
            save_all("LAST", False, extra="(end)")

    except KeyboardInterrupt:
        if rank == 0:
            print(f"[{now_str()}] KeyboardInterrupt: saving LAST before exit ...")
        save_all("LAST", False, extra="(on KeyboardInterrupt)")

    finally:
        if distributed:
            torch.distributed.barrier()
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
