#encoding=utf8
import six
import argparse

class ArgumentGroup(object):
    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, type, default, help, **kwargs):
        type = str2bool if type == bool else type
        self._group.add_argument(
            "--" + name,
            default=default,
            type=type,
            help=help + ' Default: %(default)s.',
            **kwargs)

class ArgConfig(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        train_g = ArgumentGroup(parser, "training", "training options.")
        train_g.add_arg("epoch_num",             int,    3,      "Number of epoches for fine-tuning.")
        train_g.add_arg("learning_rate",     float, 1e-3,   "Learning rate used to train with warmup.")
        train_g.add_arg("save_step",        int,    1000,   "The steps interval to save checkpoints.")
        train_g.add_arg("print_step",        int,    100,   "The steps interval to print logs.")
        train_g.add_arg("batch_size",        int,    32,     "Batch size for training.")
        train_g.add_arg("class_num",         int,    10,     "The number of output classs.")
        train_g.add_arg("init_from_checkpoint", str,    None,   "The path to init checkpoint.")
        train_g.add_arg("init_from_params",     str,    None,   "The path to init params.")
        train_g.add_arg("save_model_path",  str,    "./data/saved_models/",     "The path to save model.")
        train_g.add_arg("save_checkpoint",  str,    "checkpoint",               "Dirname for checkpoint.")
        train_g.add_arg("save_param",       str,    "params",                   "Dirname for params.")
        train_g.add_arg("random_seed",      int,    123,    "Random seed.")
        train_g.add_arg("prediciton_dir",   str,    "./data/output/",           "Path to save prediction results.")
        train_g.add_arg("inference_model_dir",  str,    "./data/inference_model/",  "Path to inference model.")

        log_g = ArgumentGroup(parser, "logging", "logging related.")
        log_g.add_arg("skip_steps",          int,    10,    "The steps interval to print loss.")

        run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
        run_type_g.add_arg("use_cuda",                     bool,   True,  "If set, use GPU for training.")
        run_type_g.add_arg("do_train",                     bool,   False,  "Whether to perform training.")
        run_type_g.add_arg("do_predict",                   bool,   False,  "Whether to perform prediction.")
        run_type_g.add_arg("do_eval",                      bool,   False,  "Whether to perform evaluation.")
        run_type_g.add_arg("do_save_inference_model",      bool,   False,  "Whether to perform saving cpp inference model.")

        cnn_model_g =ArgumentGroup(parser, "cnn_model", "cnn_model configure.") 
        cnn_model_g.add_arg("conv1_filter_size",    int,    5,      "Filter size for the first cnn.")
        cnn_model_g.add_arg("conv1_filter_num",     int,    20,     "Filter number for the first cnn.")
        cnn_model_g.add_arg("pool1_size",           int,    2,      "Pooling size for the first cnn.")
        cnn_model_g.add_arg("pool1_stride",         int,    2,      "Pooling stride for the first cnn.")
        cnn_model_g.add_arg("activity",             str,    "relu", "Activity fuction")

        cnn_model_g.add_arg("conv2_filter_size",    int,    5,      "Filter size for the second cnn.")
        cnn_model_g.add_arg("conv2_filter_num",     int,    50,     "Filter number for the second cnn.")
        cnn_model_g.add_arg("pool2_size",           int,    2,      "Pooling size for the second cnn.")
        cnn_model_g.add_arg("pool2_stride",         int,    2,      "Pooling stride for the second cnn.")


        custom_g = ArgumentGroup(parser, "customize", "customized options.")
        self.custom_g = custom_g

        self.parser = parser

    def add_arg(self, name, dtype, default, descrip):
        self.custom_g.add_arg(name, dtype, default, descrip)

    def build_conf(self):
        return self.parser.parse_args()

def str2bool(v):
    return v.lower() in ("true", "t", "1")


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')



