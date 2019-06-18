#encoding=utf8

import os
import sys
sys.path.append("../../")
import argparse
from copy import deepcopy as copy
import collections
import multiprocessing

import numpy as np
import paddle
import paddle.fluid as fluid

from core.nets.bert import BertModel
from core.toolkit.configure import JsonConfig

def create_initializer(scale=0.02):
    return fluid.initializer.TruncatedNormal(scale=scale)

def compute_passage_regression(input_tensor, name = ""):
    input_shape = list(input_tensor.shape)
    width = input_shape[-1]

    weights = fluid.layers.create_parameter(
        name=name + "passage_regression_weights",
        shape=[width],
        default_initializer=create_initializer(0.02),
        dtype='float32')

    template_var = fluid.layers.fill_constant_batch_size_like(
        input_tensor,
        shape=input_shape,
        dtype='float32',
        value=0)

    weights = fluid.layers.elementwise_add(template_var, weights, axis=-1)

    raw_scores = fluid.layers.reduce_sum(input_tensor * weights, dim=-1)

    return raw_scores

def compute_span_start_logits(input_tensor, span_mask, name = ""):
    input_shape = list(input_tensor.shape)
    seq_length = input_shape[1]
    width = input_shape[2]

    weights = fluid.layers.create_parameter(
        name = name + "span_start_weights",
        shape = [width],
        default_initializer = create_initializer(0.02),
        dtype='float32')

    weights = fluid.layers.reshape(x=weights, shape=[1, width]) 
    weights = fluid.layers.expand(x=weights, expand_times=[seq_length, 1])

    template_var = fluid.layers.fill_constant_batch_size_like(input_tensor, 
        shape=input_shape, dtype='float32', value=0)
    weights = fluid.layers.elementwise_add(template_var, weights, axis=-1)

    mul_tensor = input_tensor * weights
    raw_scores = fluid.layers.reduce_sum(mul_tensor, dim=2)
    raw_scores += (1.0 - fluid.layers.cast(x=span_mask, dtype='float32')) * -10000.0

    return raw_scores

def gather_indexes(sequence_tensor, positions):
    sequence_shape = list(sequence_tensor.shape)
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    template_var = fluid.layers.fill_constant_batch_size_like(
        sequence_tensor, 
        shape=sequence_shape,
        dtype='int32',
        value=1)

    template_var = fluid.layers.slice(
        template_var,
        axes=[1, 2],
        starts=[0, 0],
        ends=[1, 1])

    batch_size = fluid.layers.reduce_sum(template_var)
    flat_offsets = fluid.layers.range(0, batch_size*seq_length, seq_length, 'int32')
    flat_offsets_reshape = fluid.layers.reshape(x=flat_offsets, shape=[-1, 1])
    flat_positions = fluid.layers.reshape(x=positions + flat_offsets_reshape, shape=[-1])
    flat_positions.stop_gradient = True
    flat_positions = fluid.layers.cast(x=flat_positions, dtype='int32')

    flat_sequence_tensor = fluid.layers.reshape(x=sequence_tensor, shape=[-1, width])
    output_tensor = fluid.layers.gather(input=flat_sequence_tensor, index=flat_positions)
    return output_tensor


def compute_span_end_logits(input_tensor, span_mask, flat_start_positions, args, name = ""):
    input_shape = list(input_tensor.shape)
    span_mask_shape = list(span_mask.shape)

    batch_size = args.start_top_k * args.batch_size
    seq_length = span_mask_shape[1]
    width = input_shape[-1]

    start_vectors = gather_indexes(input_tensor, flat_start_positions)
    start_vectors = fluid.layers.reshape(x=start_vectors, shape=[-1, 1, width])
    start_vectors = fluid.layers.expand(x=start_vectors, expand_times=[1, seq_length, 1])
    concat_input = fluid.layers.concat(input=[start_vectors, input_tensor], axis=2)

    weights = fluid.ParamAttr(
        name=name + "conditional_fc_weights",
        initializer=create_initializer(0.02))

    bias = fluid.ParamAttr(name=name + "conditional_fc_bias")

    concat_input_reshape = fluid.layers.reshape(x=concat_input, shape=[-1, 2*width])

    conditional_tensor = fluid.layers.fc(
        input=concat_input_reshape,
        size=width,
        act="gelu",
        name=name + "span_end_conditional",
        param_attr=weights,
        bias_attr=bias)

    conditional_tensor_reshape = fluid.layers.reshape(
        x=conditional_tensor,
        shape=[-1, seq_length, width])

    conditional_tensor = fluid.layers.layer_norm(
        input=conditional_tensor_reshape,
        begin_norm_axis=2,
        param_attr=fluid.ParamAttr(
            name=name + "conditional_layernorm_gamma",
            initializer=create_initializer(0.02)),
        bias_attr=fluid.ParamAttr(name=name + "conditional_layernorm_beta"))

    end_weights = fluid.layers.create_parameter(
        name=name + "span_end_weights",
        shape=[width],
        dtype='float32',
        default_initializer=create_initializer(0.02))

    template_var = fluid.layers.fill_constant_batch_size_like(conditional_tensor,
        shape=list(conditional_tensor.shape), dtype='float32', value=0)

    end_weights = fluid.layers.reshape(x=end_weights, shape=[1, width])
    end_weights = fluid.layers.expand(x=end_weights, expand_times=[seq_length, 1])
    end_weights = fluid.layers.elementwise_add(template_var, end_weights, axis=-1)

    raw_scores = fluid.layers.reduce_sum(conditional_tensor * end_weights, dim=-1)
    raw_scores += (1.0 - fluid.layers.cast(x=span_mask, dtype='float32')) * -10000.0

    logits = fluid.layers.reshape(x=raw_scores, shape=[-1, seq_length])

    return logits

def compute_log_softmax(logits):
    softmax_logits = fluid.layers.softmax(input=logits)
    log_probs = fluid.layers.log(softmax_logits)
    return log_probs

def unsqueeze_for_conditional_span_end(tensor, nbest_size, ndims):
    input_shape = list(tensor.shape)
    tensor = fluid.layers.unsqueeze(input=tensor, axes=[1])
    tensor = fluid.layers.expand(x=tensor, expand_times=[1, nbest_size] + ([1] * (ndims - 1)))
    tensor = fluid.layers.reshape(x=tensor, shape=[-1] + input_shape[1: ])

    return tensor


def create_net(
    is_training,
    model_input,
    args):
    """
    create the network of BERT-based Machine Reading Comprehension Network.
    """

    if is_training:
        src_ids, pos_ids, sent_ids, input_mask, input_span_mask, \
            start_positions, end_positions, is_null_answer = model_input
    else:
        src_ids, pos_ids, sent_ids, input_mask, input_span_mask, \
            unique_id = model_input

    # define the model bert first

    assert isinstance(args.bert_config_path, str)

    bert_conf = JsonConfig(args.bert_config_path)

    base_model = BertModel(
        src_ids = src_ids,
        position_ids = pos_ids,
        sentence_ids = sent_ids,
        input_mask = input_mask,
        config = bert_conf)

    sequence_output = base_model.get_sequence_output()   

    # define the left part in the mrc model

    start_logits = compute_span_start_logits(
        sequence_output, input_span_mask)

    start_log_probs = compute_log_softmax(start_logits)
            
    top_k_start_log_probs, top_k_start_indexes = fluid.layers.topk(
        input=start_log_probs, k=args.start_top_k)

    span_end_input = unsqueeze_for_conditional_span_end(
        sequence_output, args.start_top_k, 3)

    input_span_mask = unsqueeze_for_conditional_span_end(
        input_span_mask, args.start_top_k, 2)
        
    flat_start_positions = fluid.layers.reshape(
        x=top_k_start_indexes, shape=[-1, 1])
        
    end_logits = compute_span_end_logits(
        span_end_input, input_span_mask, flat_start_positions, args)

    end_log_probs = compute_log_softmax(end_logits)
        
    (top_k_end_log_probs, top_k_end_indexes) = fluid.layers.topk(
        input=end_log_probs, k=args.end_top_k)
    
    top_k_end_log_probs = fluid.layers.reshape(
        x=top_k_end_log_probs, shape=[-1, args.start_top_k, args.end_top_k])

    top_k_end_indexes = fluid.layers.reshape(
        x=top_k_end_indexes, shape=[-1, args.start_top_k, args.end_top_k])

    batch_ones = fluid.layers.fill_constant_batch_size_like(
        input=start_logits, dtype='int64', shape=[1], value=1)
    num_seqs = fluid.layers.reduce_sum(batch_ones)

    # if is_training, then return the loss, otherwise return the prediction

    if is_training:
        def compute_loss(logits, positions):
            loss = fluid.layers.softmax_with_cross_entropy(
                logits=logits, label=positions)
            loss = fluid.layers.mean(x=loss)
            return loss

        start_loss = compute_loss(start_logits, start_positions)
        end_logits = fluid.layers.reshape(x=end_logits, shape=[-1, args.start_top_k, list(end_logits.shape)[-1]])
        end_logits = fluid.layers.slice(end_logits, axes=[1], starts=[0], ends=[1])
        end_logits = fluid.layers.reshape(x=end_logits, shape=[-1, list(end_logits.shape)[-1]])
        end_loss = compute_loss(end_logits, end_positions)
                
        total_loss = (start_loss + end_loss) / 2.0

        if args.use_fp16 and args.loss_scaling > 1.0:
            total_loss = total_loss * args.loss_scaling

        return total_loss

    else:
        
        predict = [unique_id, top_k_start_log_probs, top_k_start_indexes, \
            top_k_end_log_probs, top_k_end_indexes]

        return predict




