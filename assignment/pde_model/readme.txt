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

#官方Unet模型较大，无法上传。模型分别为"1D_Advection_Sols_beta4.0_Unet-PF-20.pt"和"1D_Burgers_Sols_Nu1.0_Unet-PF-20"
#为了符合官方的文件名格式，使用时须删去下划线及后面的数字（如"_1"）
