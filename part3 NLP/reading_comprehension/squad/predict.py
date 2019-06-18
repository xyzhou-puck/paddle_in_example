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

def do_predict(args):

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
            
            unique_ids.persistable = True
            top_k_start_log_probs.persistable = True
            top_k_start_indexes.persistable = True
            top_k_end_log_probs.persistable = True
            top_k_end_indexes.persistable = True

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

    # start predicting

    compiled_test_prog = fluid.CompiledProgram(test_prog)

    all_results = []
    RawResult = collections.namedtuple("RawResult", [
        "unique_id", "top_k_start_log_probs", "top_k_start_indexes",
        "top_k_end_log_probs", "top_k_end_indexes"])

    reader.start()
    while True:
        try:

            np_unique_ids, np_top_k_start_log_probs, np_top_k_start_indexes, \
                np_top_k_end_log_probs, np_top_k_end_indexes = exe.run(compiled_test_prog, fetch_list = fetch_list)

            for idx in range(np_unique_ids.shape[0]):
                if len(all_results) % 1000 == 0:
                    print("Processing example: %d" % len(all_results))
                unique_id = int(np_unique_ids[idx])

                top_k_start_log_probs = [
                    float(x) for x in np_top_k_start_log_probs[idx].flat]
                top_k_start_indexes = [int(x) for x in np_top_k_start_indexes[idx].flat]
                top_k_end_log_probs = [float(x) for x in np_top_k_end_log_probs[idx].flat]
                top_k_end_indexes = [int(x) for x in np_top_k_end_indexes[idx].flat]

                all_results.append(
                    RawResult(unique_id=unique_id,
                        top_k_start_log_probs=top_k_start_log_probs,
                        top_k_start_indexes=top_k_start_indexes,
                        top_k_end_log_probs=top_k_end_log_probs,
                        top_k_end_indexes=top_k_end_indexes))

        except fluid.core.EOFException:
            break

    features = processor.get_features(
        processor.predict_examples, is_training=False)

    write_predictions(processor.predict_examples, features, all_results,
        args.n_best_size, args.max_answer_length,
        args.do_lower_case, args.output_prediction_file,
        args.output_nbest_file, None,
        args.start_top_k, args.end_top_k, args.verbose)


if __name__ == "__main__":
    args = ArgConfig()
    args = args.build_conf()
    print_arguments(args)

    do_predict(args)

