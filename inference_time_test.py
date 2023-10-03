import torch
from torchvision.models.resnet import resnet101

from segmentation_models_pytorch import create_model
import numpy as np
iterations = 300   # 重复计算的轮次
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
model = create_model(
    arch="UNet", 
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1,)
device = torch.device("cuda")


model.to(device)

random_input = torch.randn(1, 1, 352, 352).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# GPU预热
for _ in range(50):
    _ = model(random_input)

# 测速
times = torch.zeros(iterations)     # 存储每轮iteration的时间
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = model(random_input)
        ender.record()
        # 同步GPU时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) # 计算时间
        times[iter] = curr_time
        # print(curr_time)

mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))