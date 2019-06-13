Log I got:

<code>
-----------  Configuration Arguments -----------
activity: relu
batch_size: 32
class_num: 10
conv1_filter_num: 20
conv1_filter_size: 5
conv2_filter_num: 50
conv2_filter_size: 5
do_eval: False
do_predict: False
do_save_inference_model: False
do_train: True
epoch_num: 3
inference_model_dir: ./data/inference_model/
init_from_checkpoint: None
init_from_params: None
learning_rate: 0.001
pool1_size: 2
pool1_stride: 2
pool2_size: 2
pool2_stride: 2
prediciton_dir: ./data/output/
print_step: 100
random_seed: 123
save_checkpoint: checkpoint
save_model_path: ./data/saved_models/
save_param: params
save_step: 1000
skip_steps: 10
use_cuda: True
------------------------------------------------
step: 0, loss: 6.6767
step: 100, loss: 0.4899
step: 200, loss: 0.2087
step: 300, loss: 0.1539
step: 400, loss: 0.3245
step: 500, loss: 0.0815
step: 600, loss: 0.0535
step: 700, loss: 0.0357
step: 800, loss: 0.4966
step: 900, loss: 0.2894
step: 1000, loss: 0.0514
save checkpoint at ./data/saved_models//checkpoint/step_1000
save parameters at ./data/saved_models//params/step_1000
step: 1100, loss: 0.0813
step: 1200, loss: 0.0065
step: 1300, loss: 0.1958
step: 1400, loss: 0.0036
step: 1500, loss: 0.0331
step: 1600, loss: 0.0111
step: 1700, loss: 0.1865
step: 1800, loss: 0.0198
step: 1900, loss: 0.0172
step: 2000, loss: 0.1612
save checkpoint at ./data/saved_models//checkpoint/step_2000
save parameters at ./data/saved_models//params/step_2000
step: 2100, loss: 0.0246
step: 2200, loss: 0.1903
step: 2300, loss: 0.1147
step: 2400, loss: 0.1621
step: 2500, loss: 0.1395
step: 2600, loss: 0.0657
step: 2700, loss: 0.0243
step: 2800, loss: 0.0451
save checkpoint at ./data/saved_models//checkpoint/step_final
save parameters at ./data/saved_models//params/step_final
-----------  Configuration Arguments -----------
activity: relu
batch_size: 32
class_num: 10
conv1_filter_num: 20
conv1_filter_size: 5
conv2_filter_num: 50
conv2_filter_size: 5
do_eval: True
do_predict: True
do_save_inference_model: False
do_train: False
epoch_num: 3
inference_model_dir: ./data/inference_model/
init_from_checkpoint: None
init_from_params: ./data/saved_models/params/step_final/
learning_rate: 0.001
pool1_size: 2
pool1_stride: 2
pool2_size: 2
pool2_stride: 2
prediciton_dir: ./data/output/
print_step: 100
random_seed: 123
save_checkpoint: checkpoint
save_model_path: ./data/saved_models/
save_param: params
save_step: 1000
skip_steps: 10
use_cuda: True
------------------------------------------------
init model from params at ./data/saved_models/params/step_final/
evaluation accuaracy 98.110 percent
-----------  Configuration Arguments -----------
activity: relu
batch_size: 32
class_num: 10
conv1_filter_num: 20
conv1_filter_size: 5
conv2_filter_num: 50
conv2_filter_size: 5
do_eval: False
do_predict: False
do_save_inference_model: True
do_train: False
epoch_num: 3
inference_model_dir: ./data/inference_model/
init_from_checkpoint: None
init_from_params: ./data/saved_models/params/step_final/
learning_rate: 0.001
pool1_size: 2
pool1_stride: 2
pool2_size: 2
pool2_stride: 2
prediciton_dir: ./data/output/
print_step: 100
random_seed: 123
save_checkpoint: checkpoint
save_model_path: ./data/saved_models/
save_param: params
save_step: 1000
skip_steps: 10
use_cuda: True
------------------------------------------------
init model from params at ./data/saved_models/params/step_final/
save inference model at ./data/inference_model/
</code>
