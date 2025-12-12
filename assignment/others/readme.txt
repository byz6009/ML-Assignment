train_Unet.py为训练unet模型使用的脚本，官方脚本存在bug，在if_training=False时报错。修改后可以正常输出.pickle文件。原脚本路径：PDEBench/pdebench/models/unet/train.py，使用时进行替换
train_fno.py为训练fno模型使用的脚本，在官方代码基础上增加了测试输出，可用于监测存储状态以及训练用时等。未改变原代码任何逻辑。原脚本路径：PDEBench/pdebench/models/fno/train.py，使用时进行替换
