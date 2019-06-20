#encoding=utf8
import os
import sys
sys.path.append("../../")
import numpy as np
import argparse
import paddle
import paddle.fluid as fluid

#include core lib in paddle-nlp
from core.toolkit.input_field import InputField
from core.toolkit.configure import PDConfig
from core.algorithm.optimization import optimization

# include task-specific libs
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

    print("finish initing model from pretrained params from %s" % (args.init_from_pretrain_model))

    return True

def init_from_checkpoint(args, exe, program):
    
    assert isinstance(args.init_from_checkpoint, str)

    if not os.path.exists(args.init_from_checkpoint):
        raise Warning("the checkpoint path does not exist.")
        return False

    fluid.io.load_persistables(
        executor = exe, 
        dirname=args.init_from_checkpoint, 
        main_program = program, 
        filename = "checkpoint.pdckpt")

    print("finish initing model from checkpoint from %s" % (args.init_from_checkpoint))

    return True

def save_checkpoint(args, exe, program, dirname):
    
    assert isinstance(args.save_model_path, str)

    checkpoint_dir = os.path.join(args.save_model_path, args.save_checkpoint)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    fluid.io.save_persistables(
        exe, 
        os.path.join(checkpoint_dir, dirname),
        main_program = program, 
        filename = "checkpoint.pdparams")
    
    print("save checkpoint at %s" % (os.path.join(checkpoint_dir, dirname)))

    return True

def save_param(args, exe, program, dirname):

    assert isinstance(args.save_model_path, str)
    
    param_dir = os.path.join(args.save_model_path, args.save_param)

    if not os.path.exists(param_dir):
        os.mkdir(param_dir)

    fluid.io.save_params(
        exe, 
        os.path.join(param_dir, dirname),
        main_program = program, 
        filename = "params.pdparams")
    print("save parameters at %s" % (os.path.join(param_dir, dirname)))

    return True


def do_train(args):

    train_prog = fluid.default_main_program()
    startup_prog = fluid.default_startup_program()

    with fluid.program_guard(train_prog, startup_prog):
        train_prog.random_seed = args.random_seed
        startup_prog.random_seed = args.random_seed

        with fluid.unique_name.guard():
            
            # define input and reader

            input_slots = [
                {"name": "src_ids", "shape":(-1, args.max_seq_len, 1), "dtype":"int64"},
                {"name": "pos_ids", "shape":(-1, args.max_seq_len, 1), "dtype":"int64"},
                {"name": "sent_ids", "shape":(-1, args.max_seq_len, 1), "dtype":"int64"},
                {"name": "input_mask", "shape":(-1, args.max_seq_len, 1), "dtype":"float32"},
                {"name": "input_span_mask", "shape":(-1, args.max_seq_len), "dtype":"float32"},
                {"name": "start_positions", "shape":(-1, 1), "dtype":"int64"},
                {"name": "end_positions", "shape":(-1, 1), "dtype":"int64"},
                {"name": "is_null_answer", "shape":(-1, 1), "dtype":"int64"} ]

            input_field = InputField(input_slots)
            input_field.build(build_pyreader = True)

            # define the network

            loss = create_net(is_training = True, 
                model_input = input_field, args = args)
            
            loss.persistable = True

            # define the optimizer

            if args.use_cuda:
                dev_count = fluid.core.get_cuda_device_count()
            else:
                dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))

            # as we need to get the max training steps for warmup training,
            # we define the data processer in advance
            # usually, we can declare data processor later, outsides the program_gurad scope

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
                data_path = args.training_file,
                batch_size = args.batch_size,
                phase = "train",
                shuffle = True,
                dev_count = dev_count,
                epoch = args.epoch)

            num_train_examples = processor.get_num_examples(phase='train')
            max_train_steps = args.epoch * num_train_examples // dev_count // args.batch_size
            warmup_steps = int(max_train_steps * args.warmup_proportion)
            
            print(max_train_steps, warmup_steps, num_train_examples)

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

    ## decorate the pyreader with batch_generator
    input_field.reader.decorate_batch_generator(batch_generator)

    ## define the executor and program for training

    if args.use_cuda:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    exe = fluid.Executor(place)

    exe.run(startup_prog)

    assert (args.init_from_checkpoint == "") or (args.init_from_pretrain_model == "")

    ## init from some checkpoint, to resume the previous training
    if args.init_from_checkpoint:
        init_from_checkpoint(args, exe, train_prog)

    ## init from some pretrain models, to better solve the current task
    if args.init_from_pretrain_model:
        init_from_pretrain_model(args, exe, train_prog)

    build_strategy = fluid.compiler.BuildStrategy()
    build_strategy.enable_inplace = True

    compiled_train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(
        loss_name = loss.name, build_strategy = build_strategy)

    # start training

    step = 0
    for epoch_step in range(args.epoch):
        input_field.reader.start()
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
                input_field.reader.reset()
                break

    if args.save_checkpoint:
        save_checkpoint(args, exe, train_prog, "step_final")

    if args.save_param:
        save_param(args, exe, train_prog, "step_final")



if __name__ == "__main__":
    args = PDConfig(yaml_file = "./data/config/squad1.yaml")
    args.build()
    args.Print()

    do_train(args)

