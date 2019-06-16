#encoding=utf8
import os
import sys
import numpy as np
import paddle
import paddle.fluid as fluid

from arg_config import ArgConfig, print_arguments

from train import do_train
from predict import do_predict
from eval import do_eval
from inference_model import do_save_inference_model

if __name__ == "__main__":
   
    args = ArgConfig()
    args = args.build_conf()
    print_arguments(args)

    if args.do_train:
        do_train(args)

    if args.do_predict:
        predictions = do_predict(args)

        if args.do_eval:
            acc = do_eval(args, predictions)
            print("evaluation accuaracy %.3f percent" % (acc * 100))

    if args.do_save_inference_model:
        do_save_inference_model(args)

