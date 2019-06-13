export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=1

export CUDA_VISIBLE_DEVICES=6,7

# training
/home/zhouxiangyang/env/paddle_release_home/python/bin/python -u main.py \
    --do_train=True 

# predicting and evaluating
/home/zhouxiangyang/env/paddle_release_home/python/bin/python -u main.py \
    --do_predict=True \
    --do_eval=True \
    --init_from_params="./data/saved_models/params/step_final/"

# saving the inference model
/home/zhouxiangyang/env/paddle_release_home/python/bin/python -u main.py \
    --do_save_inference=True \
    --init_from_params="./data/saved_models/params/step_final/"

