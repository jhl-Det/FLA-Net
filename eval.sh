# model=ours_v11
# split=val

# python test_produce_maps.py \
# --gpu_id 6 \
# --ckpt /home/zl/Workspace/video_object_seg/SPNet/cache/${model}/epoch_best.pth \
# --split_file ${split}_list.csv \
# --save_path ./inf_results/${model}_${split}


for num in {1..14..1}
do
    model=sota_unet
    split=test
    python inference_img.py \
    --gpu_id 4 \
    --ckpt ./cache/${model}/epoch_${num}.pth \
    --split_file ${split}_list.csv \
    --save_path ./inf_results/${model}_${split} \
    --arch UNet
done