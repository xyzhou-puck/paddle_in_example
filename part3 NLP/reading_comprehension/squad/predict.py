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

def do_predict(args):

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
            
            # make them persistable, will be removed in PaddlePaddle 1.6
            unique_ids.persistable = True
            top_k_start_log_probs.persistable = True
            top_k_start_indexes.persistable = True
            top_k_end_log_probs.persistable = True
            top_k_end_indexes.persistable = True

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

    compiled_test_prog = fluid.CompiledProgram(test_prog)

    # start predicting

    ## define data-processer and start data-reading
    processor = DataProcessor(
        vocab_path = args.vocab_path,
        do_lower_case = args.do_lower_case,
        max_seq_length = args.max_seq_len,
        in_tokens = args.in_tokens,
        doc_stride = args.doc_stride,
        do_stride = args.do_stride,
        max_query_length = args.max_query_len)

    ## define the data generator
    batch_generator = processor.data_generator(
        data_path = args.predict_file,
        batch_size = args.batch_size,
        phase = "predict",
        shuffle = False,
        dev_count = 1,
        epoch = 1)

    ## decorate the pyreader with batch_generator
    input_field.reader.decorate_batch_generator(batch_generator)

    all_results = []
    RawResult = collections.namedtuple("RawResult", [
        "unique_id", "top_k_start_log_probs", "top_k_start_indexes",
        "top_k_end_log_probs", "top_k_end_indexes"])

    input_field.reader.start()
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

    args = PDConfig(yaml_file = "./data/config/squad1.yaml")
    args.build()
    args.Print()

    do_predict(args)

