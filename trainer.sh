#!/bin/bash

# use 'nvidia-smi' API to get used memory
gpu_1='0'
gpu_2='1'
gpu_3='4'
gpu_4='3'
gpu_5='6'

gpus=${gpu_3},${gpu_2}
# gpus=${gpu_5}


# min_size=9500


# memory1=`nvidia-smi --format=csv,noheader --query-gpu=memory.free -i ${gpu_1}`
# memory1=${memory1: 0: -3}
# echo "GPU ${gpu_1} remains ${memory1} MB"

# memory2=`nvidia-smi --format=csv,noheader --query-gpu=memory.free -i ${gpu_2}`
# memory2=${memory2: 0: -3}
# echo "GPU ${gpu_2} remains ${memory2} MB"


# memory3=`nvidia-smi --format=csv,noheader --query-gpu=memory.free -i ${gpu_3}`
# memory3=${memory3: 0: -3}
# echo "GPU ${gpu_3} remains ${memory3} MB"


# memory4=`nvidia-smi --format=csv,noheader --query-gpu=memory.free -i ${gpu_4}`
# memory4=${memory4: 0: -3}
# echo "GPU ${gpu_4} remains ${memory4} MB"
# # set model size

# while ((memory1<min_size || memory2<min_size || memory3<min_size || memory4<min_size))
# do
#     memory1=`nvidia-smi --format=csv,noheader --query-gpu=memory.free -i ${gpu_1}`
#     memory1=${memory1: 0: -3}
#     echo "GPU ${gpu_1} remains ${memory1} MB"

#     memory2=`nvidia-smi --format=csv,noheader --query-gpu=memory.free -i ${gpu_2}`
#     memory2=${memory2: 0: -3}
#     echo "GPU ${gpu_2} remains ${memory2} MB"


#     memory3=`nvidia-smi --format=csv,noheader --query-gpu=memory.free -i ${gpu_3}`
#     memory3=${memory3: 0: -3}
#     echo "GPU ${gpu_3} remains ${memory3} MB"


#     memory4=`nvidia-smi --format=csv,noheader --query-gpu=memory.free -i ${gpu_4}`
#     memory4=${memory4: 0: -3}
#     echo "GPU ${gpu_4} remains ${memory4} MB"

#     echo "Out of Memory, Wait to Execute..."
#     sleep 3
# done

# Execute code
echo "AHa! Start to run our code~~~"
arch=UNet
export CUDA_VISIBLE_DEVICES=${gpus}

python train.py \
--gpu_id ${gpus} \
--batchsize 5 \
--trainsize 352 \
--save_path ./cache/ours1/ \
--lr 5e-5 \
--arch ${arch} \
--epoch 200
