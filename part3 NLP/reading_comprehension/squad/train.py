#encoding=utf8
import os
import sys
sys.path.append("../../")
import numpy as np
import argparse
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

def init_from_checkpoint(args, exe, program):
    
    assert isinstance(args.init_from_checkpoint, str)

    if not os.path.exists(args.init_from_checkpoint):
        raise Warning("the checkpoint path does not exist.")
        return False

    fluid.io.load_persistables(executor = exe, dirname=args.init_from_checkpoint, main_program = program, filename = "checkpoint.pdparams")
    print("init model from checkpoint at %s" % (args.init_from_checkpoint))

    return True

def save_checkpoint(args, exe, program, dirname):
    
    assert isinstance(args.save_model_path, str)

    checkpoint_dir = args.save_model_path + "/" + args.save_checkpoint

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    fluid.io.save_persistables(exe, checkpoint_dir + "/" + dirname, main_program = program, filename = "checkpoint.pdparams")
    print("save checkpoint at %s" % (checkpoint_dir + "/" + dirname))

    return True

def save_param(args, exe, program, dirname):

    assert isinstance(args.save_model_path, str)
    
    param_dir = args.save_model_path + "/" + args.save_param

    if not os.path.exists(param_dir):
        os.mkdir(param_dir)

    fluid.io.save_params(exe, param_dir + "/" + dirname, main_program = program, filename = "params.pdparams")
    print("save parameters at %s" % (param_dir + "/" + dirname))

    return True


def do_train(args):

    train_prog = fluid.default_main_program()
    startup_prog = fluid.default_startup_program()

    with fluid.program_guard(train_prog, startup_prog):
        train_prog.random_seed = args.random_seed
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

            start_positions = fluid.layers.data(
                name = 'start_positions', shape = [-1, 1], dtype = "int64")

            end_positions = fluid.layers.data(
                name = 'end_positions', shape = [-1, 1], dtype = "int64")

            is_null_answer = fluid.layers.data(
                name = 'is_null_answer', shape = [-1, 1], dtype = "int64")

            reader = fluid.io.PyReader(
                feed_list=[src_ids, pos_ids, sent_ids, input_mask, input_span_mask, \
                    start_positions, end_positions, is_null_answer],
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
                data_path = args.training_file,
                batch_size = args.batch_size,
                phase = "train",
                shuffle = True,
                dev_count = 4,
                epoch = args.epoch_num)

            reader.decorate_batch_generator(generator)

            # define the network

            loss = create_net(is_training = True, 
                model_input = [src_ids, pos_ids, sent_ids, input_mask, input_span_mask, \
                    start_positions, end_positions, is_null_answer], args = args)
            
            loss.persistable = True

            # define optimizer for learning

            if args.use_cuda:
                dev_count = fluid.core.get_cuda_device_count()
            else:
                dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

            num_train_examples = processor.get_num_examples(phase='train')
            max_train_steps = args.epoch_num * num_train_examples // dev_count // args.batch_size
            warmup_steps = int(max_train_steps * args.warmup_proportion)

            optimizor = optimization(
                loss = loss,
                warmup_steps = warmup_steps,
                num_train_steps = max_train_steps,
                learning_rate=args.learning_rate,
                train_program = train_prog,
                startup_prog = startup_prog,
                weight_decay = args.weight_decay,
                scheduler=args.lr_scheduler,
                use_fp16=args.use_fp16,
                loss_scaling=args.loss_scaling)


    # prepare training

    if args.use_cuda:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    exe = fluid.Executor(place)

    exe.run(startup_prog)

    assert (args.init_from_checkpoint is None) or (args.init_from_pretrain_model is None)

    if args.init_from_checkpoint:
        init_from_checkpoint(args, exe, train_prog)

    if args.init_from_pretrain_model:
        init_from_pretrain_model(args, exe, train_prog)

    build_strategy = fluid.compiler.BuildStrategy()
    build_strategy.enable_inplace = True

    compiled_train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(
        loss_name = loss.name, build_strategy = build_strategy)

    # start training

    step = 0
    for epoch_step in range(args.epoch_num):
        reader.start()
        while True:
            try:

                # this is for minimizing the fetching op, saving the training speed.
                if step % args.print_step == 0:
                    fetch_list = [loss.name]
                else:
                    fetch_list = []

                output = exe.run(compiled_train_prog, fetch_list = fetch_list)

                if step % args.print_step == 0:
                    print("step: %d, loss: %.4f" % (step, np.sum(output[0])))

                if step % args.save_step == 0 and step != 0:

                    if args.save_checkpoint:
                        save_checkpoint(args, exe, train_prog, "step_" + str(step))

                    if args.save_param:
                        save_param(args, exe, train_prog, "step_" + str(step))

                step += 1

            except fluid.core.EOFException:
                reader.reset()
                break

    if args.save_checkpoint:
        save_checkpoint(args, exe, train_prog, "step_final")

    if args.save_param:
        save_param(args, exe, train_prog, "step_final")



if __name__ == "__main__":
    args = ArgConfig()
    args = args.build_conf()
    print_arguments(args)

    do_train(args)

