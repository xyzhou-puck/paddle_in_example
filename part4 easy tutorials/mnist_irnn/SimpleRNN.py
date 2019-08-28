import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.layers.control_flow import StaticRNN as PaddingRNN

def matmul(a, b, a_t=False, b_t=False, alpha=1.0):
    return layers.matmul(a, b, a_t, b_t, alpha)

def add(a, b, axis=-1):
    return layers.elementwise_add(a, b, axis)

def act(a, act='tanh'):
    if act == 'tanh':
        return layers.tanh(a)
    elif act == 'sigmoid':
        return layers.sigmoid(a)
    elif act == 'relu':
        return layers.relu(a)
    else:
        return a

'''

SimpleRNN 

h_t = act((W_1, x_t) + mul(W_2, y_{t-1}) + b)

'''

class SimpleRNN_unit(object):
    def __init__(self, 
                 input,
                 hidden_size,
                 kernel_param_attr=None,
                 recurrent_param_attr=None,
                 bias_attr=None,
                 act='relu',
                 dtype='float32',
                 name='simple_rnn_unit'):

        self.input_size = input.shape[-1]
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.act = act

        self.w_kernel = layers.create_parameter(
            shape=[self.input_size, self.hidden_size],
            dtype=self.dtype,
            default_initializer=kernel_param_attr)

        self.w_recurrent = layers.create_parameter(
            shape=[self.hidden_size, self.hidden_size],
            dtype=self.dtype,
            default_initializer=recurrent_param_attr)

        self.b = layers.create_parameter(
            shape=[self.hidden_size],
            dtype=self.dtype,
            is_bias=True,
            default_initializer=bias_attr)

    def step(self, x_t, h_t_1):
        a_t = add(add(matmul(x_t, self.w_kernel), matmul(h_t_1, self.w_recurrent)), self.b)
        h_t = act(a_t, self.act)
        return h_t

    def __call__(self, x_t, h_t_1):
        return self.step(x_t, h_t_1)

def simple_rnn(rnn_input,
               init_hidden,
               hidden_size,
               kernel_param_attr=None,
               recurrent_param_attr=None,
               bias_attr=None,
               act='relu',
               sequence_length=None,
               name='simple_rnn'):
    
    # Transpose (sequence x batch x hidden)
    rnn_input = layers.transpose(rnn_input, [1, 0, 2])

    # Generate Mask
    mask = None
    if sequence_length:
        max_seq_len = layers.shape(rnn_input)[0]
        mask = layers.sequence_mask(sequence_length, maxlen=max_seq_len, dtype='float32')
        mask = layers.transpose(mask, [1, 0])

    # Init
    simple_rnn = SimpleRNN_unit(rnn_input, 
                                hidden_size, 
                                kernel_param_attr, 
                                recurrent_param_attr,
                                bias_attr,
                                act)

    rnn = PaddingRNN()
    with rnn.step():
        step_in = rnn.step_input(rnn_input)

        if mask:
            step_mask = rnn.step_input(mask)

        if init_hidden:
            pre_hidden = rnn.memory(init=init_hidden)
        else:
            pre_hidden = rnn.memory(batch_ref=rnn_input, shape=[-1, hidden_size])

        last_hidden = simple_rnn(step_in, pre_hidden)

        rnn.update_memory(pre_hidden, last_hidden)

        rnn.step_output(last_hidden)

        step_input = last_hidden

    rnn_out = rnn()

    last_hidden = rnn_out[-1]
    last_hidden = layers.reshape(last_hidden, shape=[1, -1, hidden_size])

    rnn_output = layers.transpose(rnn_out, [1, 0, 2])
    last_hidden = layers.transpose(last_hidden, [1, 0, 2])

    return rnn_out, last_hidden
