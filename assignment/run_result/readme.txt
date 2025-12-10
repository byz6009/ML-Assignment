# 跑通了各个重新训练的FNO模型以及官方Unet模型
# 跑通使用的是官方代码，命令示例：
CUDA_VISIBLE_DEVICES='0' python3 train_models_forward.py +args=config_Bgs.yaml ++args.filename='1D_Burgers_Sols_Nu1.0.hdf5' ++args.model_name='FNO' ++args.if_training=False
python3 analyse_result_forward.py
