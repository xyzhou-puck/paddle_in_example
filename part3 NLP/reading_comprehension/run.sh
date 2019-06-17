#引入本地环境变量
source ~/.bash_profile

#打开Fluid的显存优化功能
export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=0.0

#设置需要使用的GPU资源，如果想要单卡运行，则设置：
#	export CUDA_VISIBLE_DEVICES=0
#如果想要使用CPU运行，则设置：
# export CUDA_VISIBLE_DEVICES=""
export CUDA_VISIBLE_DEVICES=1,5,7

# training
fluid -u main.py \
    --do_train=True \
    --learning_rate 3e-5 \
    --use_cuda=True

# predicting
fluid -u main.py \
    --do_predict=True \
    --use_cuda=True

# evaluating
fluid -u main.py \
    --do_eval=True

# saving the inference model
fluid -u main.py \
    --do_save_inference=True \
    --use_cuda=True
