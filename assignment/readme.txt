pde_model为重新训练的步长为2的FNO和Unet模型，以及PINN模型
run_result为各个模型的初步跑通结果（利用官方跑通代码测试，我们自己的跑通和测试代码和输出在/code/中）。
others为训练及跑通模型时使用的与官方不一致的脚本。

在训练diff_sorp模型时，发现官方的"PDEBench/pdebench/models/fno/train.py"存在bug，FNODatasetMult对象initial_step参数未正确传入，导致出现错误。训练时已将"PDEBench/pdebench/models/fno/utils.py"中FNODatasetMult类initial_step默认值临时改为2（Unet同理）。
