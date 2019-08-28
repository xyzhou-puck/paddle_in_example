import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.layers.control_flow import StaticRNN as PaddingRNN

def conv2d(x, filters, filter_size, padding=1, bias=False):
    conv_out = layers.conv2d(
        input=x, 
        padding=padding,
        num_filters=filters, 
        filter_size=filter_size, 
        bias_attr=bias)
    return conv_out

def dot(a, b, axis=-1):
    return layers.elementwise_mul(a, b, axis)

def add(a, b, axis=-1):
    return layers.elementwise_add(a, b, axis)

'''

ConvLSTM 

i_t  = sigmoid(conv(W_{ix}, x_{t}) + conv(W_{ih}, h_{t-1}) + b_i)
f_t  = sigmoid(conv(W_{fx}, x_{t}) + conv(W_{fh}, h_{t-1}) + b_f + forget_bias )
o_t  = sigmoid(conv(W_{ox}, x_{t}) + conv(W_{oh}, h_{t-1}) + b_o)
c_t_ = tanh(conv(W_{cx}, x_t) + conv(W_{ch}, h_{t-1}) + b_c)
c_t  = dot(f_t, c_{t-1}) + dot(i_t, c_t_)
h_t  = dot(o_t, tanh(c_t))

'''

class ConvLSTM2D_unit(object):
    def __init__(self, 
                 filters,
                 filter_size, 
                 padding,
                 forget_bias=1.0,
                 name="conv_lstm_2d_uint"):
        self.filters = filters
        self.filter_size = filter_size
        self.padding = padding

        self.forget_bias = layers.fill_constant([1], dtype='float32', value=forget_bias)
        self.forget_bias.stop_gradient = False
        
    def step(self, x_t, h_t_1, c_t_1):
        i_t  = layers.sigmoid(
            conv2d(x_t, self.filters, self.filter_size, bias=True) + 
            conv2d(h_t_1, self.filters, self.filter_size))

        f_t  = layers.sigmoid(add(
            conv2d(x_t, self.filters, self.filter_size, bias=True) + 
            conv2d(h_t_1, self.filters, self.filter_size), self.forget_bias))

        o_t  = layers.sigmoid(
            conv2d(x_t, self.filters, self.filter_size, bias=True) +
            conv2d(h_t_1, self.filters, self.filter_size))

        c_t_ = layers.tanh(
            conv2d(x_t, self.filters, self.filter_size, bias=True) +
            conv2d(h_t_1, self.filters, self.filter_size))

        c_t  = add(dot(f_t, c_t_1), dot(i_t, c_t_))
        h_t  = dot(o_t, layers.tanh(c_t))

        return o_t, h_t, c_t

    def __call__(self, x_t, h_t_1, c_t_1):
        return self.step(x_t, h_t_1, c_t_1)

# rnn_input : (batch x sequence x H x W x C)
def convlstm2d_rnn(rnn_input, 
                   init_hidden,
                   init_cell,
                   padding,
                   hidden_h,
                   hidden_w,
                   filters,
                   filter_size,
                   drop_out=None,
                   sequence_length=None,
                   name='conv_lstm_2d'):

    # transpose : (sequence x batch x C x H x W)
    rnn_input = layers.transpose(rnn_input, [1, 0, 4, 2, 3])
    
    # generate mask
    mask = None
    if sequence_length:
        max_seq_len = layers.shape(rnn_input)[0]
        mask = layers.sequence_mask(sequence_length, maxlen=max_seq_len, dtype='float32')
        mask = layers.transpose(mask, [1, 0])

    # init 
    conv_lstm_2d = ConvLSTM2D_unit(filters, filter_size, padding)

    rnn = PaddingRNN()
    with rnn.step():
        step_in = rnn.step_input(rnn_input)

        if mask:
            step_mask = rnn.step_input(mask)

        if init_hidden and init_cell:
            pre_hidden = rnn.memory(init=init_hidden)
            pre_cell = rnn.memory(init=init_cell)
        else:
            pre_hidden = rnn.memory(batch_ref=rnn_input, shape=[-1, filters, hidden_h, hidden_w])
            pre_cell = rnn.memory(batch_ref=rnn_input, shape=[-1, filters, hidden_h, hidden_w])

        real_out, last_hidden, last_cell = conv_lstm_2d(step_in, pre_hidden, pre_cell)

        if mask:
            last_hidden = dot(last_hidden, step_mask, axis=0) - dot(pre_hidden, (step_mask - 1), axis=0)
            last_cell = dot(last_cell, step_mask, axis=0) - dot(pre_cell, (step_mask - 1), axis=0)

        rnn.update_memory(pre_hidden, last_hidden)
        rnn.update_memory(pre_cell, last_cell)

        rnn.step_output(last_hidden)
        rnn.step_output(last_cell)

        step_input = last_hidden

        if drop_out != None and drop_out > 0.0:
            step_input = layers.dropout(
                step_input,
                dropout_prob=drop_out,
                dropout_implementation='upscale_in_train')

    rnn_res = rnn()
    rnn_out = rnn_res[0]
    last_hidden = layers.slice(rnn_res[1], axes=[0], starts=[-1], ends=[1000000000])

    rnn_out = layers.transpose(rnn_out, [1, 0, 3, 4, 2])
    last_hidden = layers.transpose(last_hidden, [1, 0, 3, 4, 2])

    # print('rnn_out ', rnn_out.shape)
    # print('last_hidden ', last_hidden.shape)

    return rnn_out, last_hidden
























