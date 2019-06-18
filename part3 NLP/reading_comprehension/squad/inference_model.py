#encoding=utf8
import os
import sys
sys.path.append("../")
import numpy as np
import argparse
import collections
import paddle
import paddle.fluid as fluid
from core.algorithm.optimization import optimization

from arg_config import ArgConfig, print_arguments
from bert_mrc_net import create_net
from squad.reader import DataProcessor, write_predictions

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

    print("init model from pretrained params at %s" % (args.init_from_pretrain_model))

    return True

def init_from_params(args, exe, program):
    
    assert isinstance(args.init_from_params, str)

    if not os.path.exists(args.init_from_params):
        raise Warning("the params path does not exist.")
        return False

    fluid.io.load_params(executor = exe, dirname=args.init_from_params, main_program = program, filename = "params.pdparams")
    print("init model from params at %s" % (args.init_from_params))

    return True

def do_save_inference_model(args):

    test_prog = fluid.default_main_program()
    startup_prog = fluid.default_startup_program()

    with fluid.program_guard(test_prog, startup_prog):
        test_prog.random_seed = args.random_seed
        startup_prog.random_seed = args.random_seed

        with fluid.unique_name.guard():
            
            # define reader

            src_ids = fluid.layers.data(
                name = 'src_ids', shape = [-1, args.max_seq_len, 1], dtype = "int64")
            
            pos_ids = fluid.layers.data(
                name = 'pos_ids', shape = [-1, args.max_seq_len, 1], dtype = "int64")

            sent_ids = fluid.layers.data(
                name = 'sent_ids', shape = [-1, args.max_seq_len, 1], dtype = "int64")

            input_mask = fluid.layers.data(
                name = 'input_mask', shape = [-1, args.max_seq_len, 1], dtype = "float32")

            input_span_mask = fluid.layers.data(
                name = 'input_span_mask', shape = [-1, args.max_seq_len], dtype = "float32")

            unique_id = fluid.layers.data(
                name = 'unique_id', shape = [-1, 1], dtype = "int64")

            reader = fluid.io.PyReader(
                feed_list=[src_ids, pos_ids, sent_ids, input_mask, input_span_mask, unique_id],
                capacity=200, iterable=False)

            processor = DataProcessor(
                vocab_path = "./data/pretrain_models/bert_large_cased/vocab.txt",
                do_lower_case = args.do_lower_case,
                max_seq_length = args.max_seq_len,
                in_tokens = False,
                doc_stride = args.doc_stride,
                do_stride = args.do_stride,
                max_query_length = args.max_query_len)

            generator = processor.data_generator(
                data_path = args.predict_file,
                batch_size = args.batch_size,
                phase = "predict",
                shuffle = False,
                dev_count = 1,
                epoch = 1)

            reader.decorate_batch_generator(generator)

            # define the network

            predictions = create_net(is_training = False, 
                model_input = [src_ids, pos_ids, sent_ids, input_mask, input_span_mask, unique_id], args = args)

            unique_ids, top_k_start_log_probs, top_k_start_indexes, top_k_end_log_probs, top_k_end_indexes = predictions
            
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
        feeded_var_names = [src_ids.name, pos_ids.name, sent_ids.name, input_mask.name, input_span_mask.name, unique_id.name],
        target_vars = [unique_ids, top_k_start_log_probs, top_k_start_indexes, top_k_end_log_probs, top_k_end_indexes],
        executor = exe,
        main_program = test_prog,
        model_filename = "model.pdmodel",
        params_filename = "params.pdparams")

    print("save inference model at %s" % (args.inference_model_dir))

if __name__ == "__main__":
    args = ArgConfig()
    args = args.build_conf()
    print_arguments(args)

    do_predict(args)

