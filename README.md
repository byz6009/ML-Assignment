Workspace Layout

- `assignment/` – 存放专家模型相关的文件和代码 
  - `readme.txt` – 配置说明  
  - `others/` – 训练脚本（`train_Unet.py`, `train_fno.py` 等，只更改了个别必要参数如intial_step，并添加了测试输出，使用时需进行替换，详见assignment/others/readme.txt）  
  - `pde_model/` – 预训练权重（FNO/Unet/PINN）及说明  
  - `run_result/` – 运行结果的 csv/pdf/pickle 及 `readme.txt`
- `code/` – 
  - `model_eval_outputs`: 重新训练后专家模型的测试结果统计及可视化
  - `burgers_piern_infer.py`: PiERN 融合代码
  - `burgers_piern_results.txt`: 融合代码初步测试结果
  - `run_expert_model.py`: 运行专家模型代码
  - `test_run_expert_model.py`: 测试指定的专家模型，可以调整测试方式和样本数
  - `test_all_models.py`：测试所有的专家模型，得到统计结果的可视化输出
  - `train_burgers_lora_mlp_run4_e50.py`: 文生计算模块训练代码
  - `train_token_router_Burgers.py`: 路由器模块训练代码
- `data/` – 训练与采样所使用的jsonl数据以及pdebench官方数据
- `model/` – 模型文件与依赖，运行时应包含Qwen2.5-7B-Instruct模型、文生和路由器模型文件。
- `PDEBench-main/` – 上游 PDEBench 项目，运行过程中只从中引入模型定义，不使用
- `.vscode/` – VS Code 项目设置（为防止python解解器报错，可无视）
- `PiERN.pdf` – 论文 PDF
- `requirements.txt` – 依赖列表

  注：data和model文件夹内的内容过大。因此并没有放入repo。

first commit: 
1. 选择的问题为diff-sorp,Advection-beta4.0,Burgers-Nu1.0，每个问题都包含了FNO，Unet，PINN三个模型。后续打算对三个FNO模型训练文生和token路由器模块并进行组装
2. 跑通专家模型的代码为于code/run_expert_model.py，运行code/test_run_expert_model.py可以进行测试，运行code/test_all_models.py可以对所有模型做测试并可视化统计结果，可视化结果图片输出位置为code/model_eval_outputs/。模型文件位于assignment/pde_model。组装代码为code/burgers_piern_inter.py。
3. 文生计算模块的训练和测试数据应位于data文件夹中，模型文件位于model文件夹下，包括head_best_lora_mlp_run4_e50_b10_ga1_se250.pt和lora_best_lora_mlp_run4_e50_b10_ga1_se250.pt，在网盘中进行提交
4. token路由器和文生类似，训练数据jsonl应位于data文件夹中,模型文件位于model中，为capacity_token_router_best_model.pt，同样以网盘方式提交。
5. hdf5数据文件采用对应的官方数据集，同样应位于data文件夹下，文件总量过大，故不放入repo与网盘中




以下是PiERN的介绍：

# PiERN

**PiERN** is the official repository of the DREAMLAB-PKU team.  
This project provides code, data, and model implementations for our research work.

---

## 📌 Introduction

Tasks on complex systems often require **high-precision numerical computation** to support decision-making.  
However, current large language models (LLMs) struggle to natively integrate such computations as an intrinsic and interpretable capability. Multi-agent approaches can leverage external experts, but they suffer from **communication overhead** and **limited scalability**.

To address this, we propose **Physically-isolated Experts Routing Network (PiERN)**, an architecture for integrating **computation and reasoning**. Unlike tool-use workflows or function-calling, PiERN **endogenously integrates computational modules into neural networks**. After separately training experts, a text-to-computation module, and a router, PiERN performs reasoning and computation at the **token level**, enabling iterative alternation within a single chain of thought.

We evaluate PiERN on both **linear and nonlinear numerical reasoning tasks**, against LLM finetuning and multi-agent systems. Results show that PiERN achieves not only higher accuracy but also **significant improvements** in:
- Response latency  
- Token usage  
- GPU energy consumption  

PiERN offers an **efficient, interpretable, and scalable** paradigm for interfacing language models with scientific systems.

📄 For more details, please see and cite our [PiERN Paper](./PiERN.pdf).  
👉 Project page: https://github.com/DREAMLAB-PKU/PiERN

## 🚀 Quick Start

Clone the repository and install dependencies, then enter the code directory and make sure all the required models are downloaded (see the corresponding *.py files for download instructions):

```bash
git clone https://github.com/DREAMLAB-PKU/PiERN.git
cd PiERN
pip install -r requirements.txt
cd code
python3 capacity_sample_PiERN.py
