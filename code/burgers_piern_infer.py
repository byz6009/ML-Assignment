"""
Burgers PiERN-style inference: router + text2computation + expert model.

- Only Burgers
- reduced resolution: rr=4, rr_t=5
- initial_step=2
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import List

import h5py
import jsonlines
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from run_expert_model import predict_next_frame

# -----------------------------
# Configuration
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT2COMP_DEVICE = torch.device("cpu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_PATH = PROJECT_ROOT / "model" / "Qwen" / "Qwen2.5-7B-Instruct"
ROUTER_WEIGHTS = PROJECT_ROOT / "model" / "token_router_best_model.pt"
HEAD_WEIGHTS = PROJECT_ROOT / "model" / "head_best_lora_mlp_run4_e50_b10_ga1_se250.pt"
LORA_WEIGHTS = PROJECT_ROOT / "model" / "lora_best_lora_mlp_run4_e50_b10_ga1_se250.pt"

TEST_DATA = PROJECT_ROOT / "data" / "burgers_text2computation_test.jsonl"
BURGERS_DATA = PROJECT_ROOT / "data" / "1D_Burgers_Sols_Nu1.0.hdf5"

RESULT_PLACEHOLDER = "<数值计算结果>"

RR = 4
RR_T = 5
INITIAL_STEP = 2
MAX_LENGTH = 5312
LLM_MAX_LENGTH = 8192
LABEL_DIM = 512
MIN_ASSISTANT_TOKENS = 8

TARGET_MODULE_KEYS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

BURGER_SYSTEM_TEXT = """你是一名结合语言理解能力与物的理建模能力的智能助手，能够根据用户输入灵活切换任务模式。  
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


# -----------------------------
# Model definitions
# -----------------------------
class LMClassifier1D(nn.Module):
    def __init__(self, vocab_size, embed_dim=1536, hidden_dim=128, output_dim=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        masked = embedded * attention_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        logits = self.fc(pooled)
        return logits


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

        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))

        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return out + lora * self.scaling


def inject_lora(model: nn.Module, target_keys: List[str], r: int, alpha: int, dropout: float) -> None:
    for name, module in list(model.named_modules()):
        if not any(name.endswith(k) for k in target_keys):
            continue
        if isinstance(module, nn.Linear):
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], LoRALinear(module, r=r, alpha=alpha, dropout=dropout))


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


class Text2Computation(nn.Module):
    def __init__(self, base_model: AutoModelForCausalLM, head: nn.Module):
        super().__init__()
        self.base = base_model
        self.head = head

    def forward(self, input_ids, attention_mask):
        out = self.base.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = pooled.float()
        return self.head(pooled)


# -----------------------------
# Utilities
# -----------------------------
def wrap_prompt(system_text: str, user_text: str) -> str:
    return (
        "<|im_start|>system\n\n" + system_text.strip() + "<|im_end|>\n"
        "<|im_start|>user\n\n" + user_text.strip() + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def inference_router_from_ids(input_ids, attention_mask, router_model):
    with torch.no_grad():
        logits = router_model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).squeeze(-1)
        preds = (probs > 0.5).long()
    return preds.item(), probs.item()


def format_numeric_result(values: np.ndarray) -> str:
    formatted = ", ".join(f"{v:.6f}" for v in values.tolist())
    return f"[{formatted}]"


def decode_assistant_text(tokenizer, generated_ids: torch.Tensor, prompt_len: int) -> str:
    content_ids = generated_ids[prompt_len:]
    if content_ids.numel() == 0:
        return ""
    return tokenizer.decode(content_ids, skip_special_tokens=True).strip()


def has_textual_content(text: str) -> bool:
    return bool(re.search(r"[A-Za-z\u4e00-\u9fff]", text))


def build_final_response(assistant_text: str, numeric_str: str) -> str:
    assistant_text = assistant_text.strip()
    if RESULT_PLACEHOLDER in assistant_text:
        return assistant_text.replace(RESULT_PLACEHOLDER, numeric_str, 1).strip()
    if not has_textual_content(assistant_text):
        assistant_text = ""
    if assistant_text:
        if numeric_str in assistant_text:
            return assistant_text.strip()
        return f"{assistant_text.strip()} {numeric_str}".strip()
    return f"根据Burgers'方程的数值求解结果，在整个空间域内的速度场数值解为{numeric_str}"

def compute_expert_output(
    text2comp_model: Text2Computation,
    tokenizer,
    prompt: str,
    x_coords: np.ndarray,
    device: torch.device,
    expert_device: torch.device | None = None,
) -> np.ndarray:
    model_device = next(text2comp_model.parameters()).device
    enc = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(model_device)
    attention_mask = enc["attention_mask"].to(model_device)

    with torch.no_grad():
        preds = text2comp_model(input_ids, attention_mask).squeeze(0).cpu().numpy()

    if preds.shape[0] != LABEL_DIM:
        raise ValueError(f"Unexpected label dim: {preds.shape[0]}")

    frame0 = preds[: LABEL_DIM // 2]
    frame1 = preds[LABEL_DIM // 2 :]

    x_ds = x_coords[::RR]
    if x_ds.shape[0] != frame0.shape[0]:
        raise ValueError(
            f"x-coordinate length {x_ds.shape[0]} does not match frame length {frame0.shape[0]}"
        )

    if expert_device is None:
        expert_device = device
    pred = predict_next_frame(
        "Burgers_FNO", frame0, frame1, grid_coords=x_ds, device=expert_device
    )
    return pred.reshape(-1)


def generate_response_with_router(
    messages,
    tokenizer,
    llm_model,
    router_model,
    text2comp_model,
    x_coords,
    device,
    max_new_tokens=80,
    debug: bool = False,
):
    text = wrap_prompt(messages[0]["content"], messages[1]["content"])
    inputs = tokenizer(
        [text],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=LLM_MAX_LENGTH,
    ).to(device)

    generated_ids = inputs["input_ids"]
    prompt_len = generated_ids.size(1)

    expert_result = None

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = llm_model(input_ids=generated_ids, use_cache=False)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        generated_count = generated_ids.size(1) - prompt_len
        if next_token.item() == tokenizer.eos_token_id and generated_count < MIN_ASSISTANT_TOKENS:
            masked_logits = next_token_logits.clone()
            masked_logits[:, tokenizer.eos_token_id] = -1e9
            next_token = torch.argmax(masked_logits, dim=-1).unsqueeze(-1)
        elif next_token.item() == tokenizer.eos_token_id:
            break

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        new_tokens = generated_ids[:, prompt_len:]
        if new_tokens.numel() == 0:
            continue

        if new_tokens.size(1) < MIN_ASSISTANT_TOKENS:
            continue

        assistant_text = decode_assistant_text(tokenizer, generated_ids[0], prompt_len)
        has_placeholder = RESULT_PLACEHOLDER in assistant_text
        has_text = has_textual_content(assistant_text)
        if debug and (step < 5 or step % 10 == 0 or has_placeholder):
            snippet = assistant_text[:120].replace("\n", " ")
            print(
                f"[debug] step={step} tokens={new_tokens.size(1)} "
                f"placeholder={has_placeholder} text_ok={has_text} text='{snippet}'"
            )

        if not has_placeholder:
            continue

        attention_mask = torch.ones_like(new_tokens).to(device)
        pred, _ = inference_router_from_ids(new_tokens, attention_mask, router_model)

        if pred == 1:
            if expert_result is None:
                expert_result = compute_expert_output(
                    text2comp_model, tokenizer, text, x_coords, device
                )
            numeric_str = format_numeric_result(expert_result)
            assistant_text = decode_assistant_text(tokenizer, generated_ids[0], prompt_len)
            return build_final_response(assistant_text, numeric_str)
        if debug:
            print("[debug] router did not trigger, continue generating")

    assistant_text = decode_assistant_text(tokenizer, generated_ids[0], prompt_len)
    if debug:
        snippet = assistant_text[:160].replace("\n", " ")
        print(f"[debug] final assistant text='{snippet}'")
    if expert_result is None:
        expert_result = compute_expert_output(
            text2comp_model, tokenizer, text, x_coords, device
        )
    numeric_str = format_numeric_result(expert_result)
    return build_final_response(assistant_text, numeric_str)


def load_x_coords(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.array(f["x-coordinate"], dtype=np.float32)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Burgers PiERN inference")
    parser.add_argument("--max-samples", type=int, default=3, help="limit samples")
    parser.add_argument("--output", type=str, default="burgers_piern_results.txt")
    parser.add_argument("--debug", action="store_true", help="print debug info")
    args = parser.parse_args()

    print("Loading tokenizer and models...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH, use_fast=False, trust_remote_code=True
    )

    vocab_size = len(tokenizer)
    router_model = LMClassifier1D(vocab_size).to(DEVICE)
    router_model.load_state_dict(torch.load(ROUTER_WEIGHTS, map_location=DEVICE))
    router_model.eval()

    base_for_reg = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True
    ).to(TEXT2COMP_DEVICE)
    inject_lora(base_for_reg, TARGET_MODULE_KEYS, r=16, alpha=32, dropout=0.05)
    lora_state = torch.load(LORA_WEIGHTS, map_location="cpu")
    base_for_reg.load_state_dict(lora_state, strict=False)
    base_for_reg.eval()

    head = RegressionHead(hidden_size=base_for_reg.config.hidden_size, out_dim=LABEL_DIM).to(TEXT2COMP_DEVICE)
    head_state = torch.load(HEAD_WEIGHTS, map_location="cpu")
    if any(k.startswith("0.") or k.startswith("2.") for k in head_state.keys()):
        head_state = {f"net.{k}": v for k, v in head_state.items()}
    head.load_state_dict(head_state, strict=True)

    text2comp_model = Text2Computation(base_for_reg, head).to(TEXT2COMP_DEVICE)
    text2comp_model.eval()

    llm_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True
    ).to(DEVICE)
    llm_model.config.use_cache = False
    llm_model.eval()

    x_coords = load_x_coords(Path(BURGERS_DATA))

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as fout, jsonlines.open(
        TEST_DATA, "r"
    ) as reader:
        for idx, record in enumerate(reader):
            if idx >= args.max_samples:
                break
            if not isinstance(record, dict) or "prompt" not in record:
                continue
            message = str(record["prompt"]).strip()
            if not message:
                continue

            messages = [
                {"role": "system", "content": BURGER_SYSTEM_TEXT},
                {"role": "user", "content": message},
            ]

            response = generate_response_with_router(
                messages,
                tokenizer,
                llm_model,
                router_model,
                text2comp_model,
                x_coords,
                DEVICE,
                max_new_tokens=120,
                debug=args.debug,
            )

            fout.write(f"Index {idx}\n")
            fout.write(f"Response: {response}\n")
            fout.write("-" * 40 + "\n")

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
