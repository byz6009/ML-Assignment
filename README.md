Workspace Layout

- `assignment/` – 存放专家模型相关的文件和代码 
  - `readme.txt` – 配置说明  
  - `others/` – 训练脚本（`train_Unet.py`, `train_fno.py` 等，只更改了个别必要参数如intial_step，并添加了测试输出，完全没有改变代码逻辑和超参数,使用时需进行替换，详见assignment/others/readme.txt）  
  - `pde_model/` – 预训练权重（FNO/Unet）及说明  
  - `run_result/` – 运行结果的 csv/pdf/pickle 及 `readme.txt`
- `code/` – PiERN 运行专家模型代码以及训练与推理脚本（`run_expert_model.py` 等）
- `data/` – 训练与采样所使用的数据集jsonl生成代码以及 jsonl 数据
- `model/` – 模型文件与依赖 
  - `capacity_expert_model.pt`  
  - `PDEBench-main/` – 上游 PDEBench 项目，运行过程中只从中引入模型定义，不使用
- `.vscode/` – VS Code 项目设置（为防止python解释器报错，可无视）
- `PiERN.pdf` – 论文 PDF
- `requirements.txt` – 依赖列表

first commit: 
  1.选择的专家模型为diff-sorp,Advection-beta4.0-FNO,Advection-beta4.0-Unet,Burgers-Nu1.0-FNO,Burgers-Nu1.0-Unet

  2.跑通专家模型的代码为于code/run_expert_model.py，运行code/test_run_expert_model.py可以进行测试。模型文件位于assignment/pde_model

  3.文生计算模块和token路由器的训练数据以及生成代码应位于data文件夹中，jsonl文件过大，因此以另外方式提交
  
  4.hdf5数据文件同样应位于data文件夹下，文件过大，与jsonl类似不放入主文件夹



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
