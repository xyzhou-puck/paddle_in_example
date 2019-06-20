#encoding=utf8
import os
import sys
sys.path.append("../../")
import numpy as np
import argparse
import collections
import paddle
import paddle.fluid as fluid

from core.toolkit.input_field import InputField
from core.toolkit.configure import PDConfig

from bert_mrc_net import create_net

def init_from_pretrain_model(args, exe, program):
    
    assert isinstance(args.init_from_pretrain_model, str)

    if not os.path.exists(args.init_from_pretrain_model):
        raise Warning("The pretrained params do not exist.")
        return False

    def existed_params(var):
        if not isinstance(var, fluid.framework.Parameter):
            return False
        return os.path.exists(os.path.join(args.init_from_pretrain_model, var.name))

    fluid.io.load_vars(
        exe,
        args.init_from_pretrain_model,
        main_program=program,
        predicate=existed_params)

    print("finsih init model from pretrained params from %s" % (args.init_from_pretrain_model))

    return True

def init_from_params(args, exe, program):
    
    assert isinstance(args.init_from_params, str)

    if not os.path.exists(args.init_from_params):
        raise Warning("the params path does not exist.")
        return False

    fluid.io.load_params(
        executor = exe, 
        dirname=args.init_from_params, 
        main_program = program, 
        filename = "params.pdparams")
    
    print("finish init model from params from %s" % (args.init_from_params))

    return True

def do_save_inference_model(args):

    test_prog = fluid.default_main_program()
    startup_prog = fluid.default_startup_program()

    with fluid.program_guard(test_prog, startup_prog):
        test_prog.random_seed = args.random_seed
        startup_prog.random_seed = args.random_seed

        with fluid.unique_name.guard():
            
            # define inputs of the network

            input_slots = [
                {"name": "src_ids", "shape":(-1, args.max_seq_len, 1), "dtype":"int64"},
                {"name": "pos_ids", "shape":(-1, args.max_seq_len, 1), "dtype":"int64"},
                {"name": "sent_ids", "shape":(-1, args.max_seq_len, 1), "dtype":"int64"},
                {"name": "input_mask", "shape":(-1, args.max_seq_len, 1), "dtype":"float32"},
                {"name": "input_span_mask", "shape":(-1, args.max_seq_len), "dtype":"float32"},
                {"name": "unique_id", "shape":(-1, 1), "dtype":"int64"},
            ]

            input_field = InputField(input_slots)
            input_field.build(build_pyreader = True)

            # define the network

            predictions = create_net(is_training = False, 
                model_input = input_field, args = args)

            # declare the outputs to be fetched
            unique_ids, top_k_start_log_probs, top_k_start_indexes, top_k_end_log_probs, top_k_end_indexes = predictions

            # put all fetched outputs into fetch_list
            fetch_list = [unique_ids.name, top_k_start_log_probs.name, top_k_start_indexes.name,
                top_k_end_log_probs.name, top_k_end_indexes.name]

    # prepare predicting

    if args.use_cuda:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    exe = fluid.Executor(place)

    exe.run(startup_prog)

    assert (args.init_from_params) or (args.init_from_pretrain_model)

    if args.init_from_params:
        init_from_params(args, exe, test_prog)

    elif args.init_from_pretrain_model:
        init_from_pretrain_model(args, exe, test_prog)

    # saving inference model

    fluid.io.save_inference_model(
        args.inference_model_dir,
        feeded_var_names = [input_field.src_ids.name, input_field.pos_ids.name, input_field.sent_ids.name, 
                            input_field.input_mask.name, input_field.input_span_mask.name, input_field.unique_id.name],
        target_vars = [unique_ids, top_k_start_log_probs, top_k_start_indexes, top_k_end_log_probs, top_k_end_indexes],
        executor = exe,
        main_program = test_prog,
        model_filename = "model.pdmodel",
        params_filename = "params.pdparams")

    print("save inference model at %s" % (args.inference_model_dir))


if __name__ == "__main__":

    args = PDConfig(yaml_file = "./data/config/squad1.yaml")
    args.build()
    args.Print()

    do_predict(args)

