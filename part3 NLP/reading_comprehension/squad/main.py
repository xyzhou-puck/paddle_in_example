#encoding=utf8
import os
import sys
sys.path.append("../../")
import numpy as np
import paddle
import paddle.fluid as fluid

from core.toolkit.configure import PDConfig

from train import do_train
from predict import do_predict
from eval import do_eval
from inference_model import do_save_inference_model

if __name__ == "__main__":
   
    args = PDConfig(yaml_file = "./data/config/squad1.yaml")
    args.build()
    args.Print()

    if args.do_train:
        do_train(args)

    if args.do_predict:
        do_predict(args)

    if args.do_eval:
        do_eval(args)

    if args.do_save_inference_model:
        do_save_inference_model(args)

