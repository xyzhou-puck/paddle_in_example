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
        train_g.add_arg("epoch_num",            int,    2,          "Number of epoches for fine-tuning.")
        train_g.add_arg("learning_rate",        float,  1e-5,       "Learning rate used to train with warmup.")
        train_g.add_arg("save_step",            int,    1000,       "The steps interval to save checkpoints.")
        train_g.add_arg("print_step",           int,    100,        "The steps interval to print logs.")
        train_g.add_arg("batch_size",           int,    2,         "Batch size for training.")
        train_g.add_arg("init_from_checkpoint", str,    None,       "The path to init checkpoint.")
        train_g.add_arg("init_from_params",     str,    None,       "The path to init params.")
        train_g.add_arg("init_from_pretrain_model",     str,    "./data/pretrain_models/bert_large_cased/params/",  "Whether to init from pretrain model.")
        train_g.add_arg("save_model_path",      str,    "./data/saved_models/",     "The path to save model.")
        train_g.add_arg("save_checkpoint",      str,    "checkpoint",               "Dirname for checkpoint.")
        train_g.add_arg("save_param",           str,    "params",                   "Dirname for params.")
        train_g.add_arg("random_seed",          int,    123,        "Random seed.")
        train_g.add_arg("prediciton_dir",       str,    "./data/output/",           "Path to save prediction results.")
        train_g.add_arg("inference_model_dir",  str,    "./data/inference_model/",  "Path to inference model.")
        train_g.add_arg("training_file",        str,    "./data/input/train-v1.1.json", "Path to the training file.")
        train_g.add_arg("predict_file",         str,    "./data/input/dev-v1.1.json",   "Path to the testing file.")
        train_g.add_arg("evaluation_file",      str,    "./data/input/dev-v1.1.json",   "Path to the annotation file.")
        train_g.add_arg("lr_scheduler",         str,    "linear_warmup_decay",      "Scheduler of learning rate.")
        train_g.add_arg("weight_decay",         float,  0.01,   "Weight decay rate for L2 regularizer.")
        train_g.add_arg("warmup_proportion",    float,  0.1,    "Proportion of training steps to perform linear learning rate warmup for.")
        train_g.add_arg("loss_scaling",         float,  1.0,    "Loss scaling factor for mixed precision training, only valid when use_fp16 is enabled.")
        train_g.add_arg("do_lower_case",        bool,   False,  "Using cased/uncased model.")

        pred_g = ArgumentGroup(parser, "predicting",    "predicting options.")
        pred_g.add_arg("n_best_size",           int,    20,     "The total number of n-best predictions to generate in the nbest_predictions.json output file.")
        pred_g.add_arg("max_answer_length",     int,    30,     "Max answer length.")
        pred_g.add_arg("output_prediction_file",        str,    "./data/output/predictions.json",   "Path to output file.")
        pred_g.add_arg("output_nbest_file",     str,    "./data/output/nbest_predictions.json",     "Path to n-best output file.")
        pred_g.add_arg("verbose",               bool,   False,  "")

        log_g = ArgumentGroup(parser, "logging", "logging related.")
        log_g.add_arg("skip_steps",          int,    10,    "The steps interval to print loss.")

        run_type_g = ArgumentGroup(parser, "run_type", "running type options.")
        run_type_g.add_arg("use_cuda",                     bool,   True,  "If set, use GPU for training.")
        run_type_g.add_arg("do_train",                     bool,   False,  "Whether to perform training.")
        run_type_g.add_arg("do_predict",                   bool,   False,  "Whether to perform prediction.")
        run_type_g.add_arg("do_eval",                      bool,   False,  "Whether to perform evaluation.")
        run_type_g.add_arg("do_save_inference_model",      bool,   False,  "Whether to perform saving cpp inference model.")
        run_type_g.add_arg("use_fp16",                     bool,   False,  "Whether to use fp16 in training/predicting.")

        mrc_model_g = ArgumentGroup(parser, "mrc_model",    "mrc model configure.")
        mrc_model_g.add_arg("max_seq_len",  int,    512,    "Max sequence length.")
        mrc_model_g.add_arg("bert_config_path", str,    "./data/pretrain_models/bert_large_cased/bert_config.json", "Path to Bert/Ernie configure file.")
        mrc_model_g.add_arg("start_top_k",  int,    20, "Top k candidate for start position.")
        mrc_model_g.add_arg("end_top_k",    int,    4,  "Top k candidate for end position.")
        mrc_model_g.add_arg("doc_stride",   int,    128,    "Document stride size.")
        mrc_model_g.add_arg("do_stride",    bool,   True,   "Whether to use doc stride.")
        mrc_model_g.add_arg("max_query_len",    int,    64, "Max Query length.")

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



