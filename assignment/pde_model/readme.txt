#以下是各个FNO模型训练时与官方配置文件不同的参数

#diff-sorp

initial_step: 2
epochs: 250
num_workers: 16
batch_size: 190

#Adv

initial_step: 2
num_workers: 8
batch_size: 512

#Bgs

initial_step: 2
num_workers: 8
batch_size: 512
