# 线性回归
线性回归（Linear Regression \[[1](#参考文献)\]）是机器学习领域中最为经典的算法。在这一章里，我们将学习如何使用PaddlePaddle构建一个基于线性回归算法房价预测模型。

本教程源代码目录在[book/fit_a_line](https://github.com/PaddlePaddle/book/tree/develop/01.fit_a_line)， 初次使用请您参考[Book文档使用说明](https://github.com/PaddlePaddle/book/blob/develop/README.cn.md#运行这本书)。

## 背景介绍
假设我们需要预测一个房屋的价格，一个可行的方法通过考虑这个房屋的一些"特征"来衡量这个房屋的价格，比如，房屋本身的大小，有多少个房间，周边学校，医院，商场的数目，房屋所在地的自然环境（是否有污染）和社会环境（周边的犯罪率）等。例如，在波士顿房价数据集里 \[[2](#参考文献)\]，共统计了以下房屋特征，用于房价预测：

| 特征名  | 解释                                   | 类型                     |
| ------- | -------------------------------------- | ------------------------ |
| CRIM    | 该镇的人均犯罪率                       | 连续值                   |
| ZN      | 占地面积超过25,000平方呎的住宅用地比例 | 连续值                   |
| INDUS   | 非零售商业用地比例                     | 连续值                   |
| CHAS    | 是否邻近 Charles River                 | 离散值，1=邻近；0=不邻近 |
| NOX     | 一氧化氮浓度                           | 连续值                   |
| RM      | 每栋房屋的平均客房数                   | 连续值                   |
| AGE     | 1940年之前建成的自用单位比例           | 连续值                   |
| DIS     | 到波士顿5个就业中心的加权距离          | 连续值                   |
| RAD     | 到径向公路的可达性指数                 | 连续值                   |
| TAX     | 全值财产税率                           | 连续值                   |
| PTRATIO | 学生与教师的比例                       | 连续值                   |
| B       | 1000(BK - 0.63)^2，其中BK为黑人占比    | 连续值                   |
| LSTAT   | 低收入人群占比                         | 连续值                   |
| MEDV    | 同类房屋价格的中位数                   | 连续值                   |

###

我们用变量$x_i$ 代表某一个房屋，用$\{x_{i1}, …, x_{id}\}$ 表示这个房屋不同特征的具体取值，用$y_i$表示这个房屋的真实价格。现在我们使用线性回归模型去计算这个房屋的价格。在使用线性回归模型时，有一个基本假设，那就是真实房价和房屋特征值之间，存在一个线性关系，即：

$$y_i = \omega_1x_{i1} + \omega_2x_{i2} + \ldots + \omega_dx_{id} + b,  i=1,\ldots,n$$

其中，$\{w_1, w_2, …, w_d\}$表示了不同特征的权重，$b$ 是一个偏置（bias）。

在波士顿房价数据集中，和房屋相关的值共有14个：前13个用来描述房屋相关的各种信息，即模型中的 $x_i$；最后一个值为我们要预测的该类房屋价格的中位数，即模型中的 $y_i$。因此，我们的模型就可以表示成：

$$\hat{y_i} = \omega_1x_{i1} + \omega_2x_{i2} + \ldots + \omega_{13}x_{i13} + b$$

$\hat{y_i}$ 表示模型的预测结果，用来和真实值$y_{i}$区分。模型要学习的参数即：$\omega_1, \ldots, \omega_{13}, b$。

建立模型后，我们需要给模型一个优化目标，使得学到的参数能够让预测值$\hat{y_i}$ 尽可能地接近真实值$y_i$。这里我们引入损失函数（[Loss Function](https://en.wikipedia.org/wiki/Loss_function)，或Cost Function）这个概念。 输入任意一个数据样本的目标值$y_{i}$和模型给出的预测值$\hat{y_{i}}$，损失函数输出一个非负的实值。这个实值通常用来反映模型误差的大小。

对于线性回归模型来讲，最常见的损失函数就是均方误差（Mean Squared Error， [MSE](https://en.wikipedia.org/wiki/Mean_squared_error)）了，它的形式是：

$$MSE=\frac{1}{n}\sum_{i=1}^{n}{(\hat{y_i}-y_i)}^2$$

即对于一个大小为$n$的测试集，$MSE$是$n$个数据预测结果误差平方的均值。

对损失函数进行优化所采用的方法一般为梯度下降法。梯度下降法是一种一阶最优化算法。如果$f(x)$在点$x_n$有定义且可微，则认为$f(x)$在点$x_n$沿着梯度的负方向$-▽f(x_n)$下降的是最快的。反复调节$x$，使得$f(x)$接近最小值或者极小值，调节的方式为：

$$x_n+1=x_n-λ▽f(x), n≧0$$

其中λ代表学习率（learning rate）。这种调节的方法称为梯度下降法。

### 训练过程

定义好模型结构之后，我们要通过以下几个步骤进行模型训练
 1. 初始化参数，其中包括权重$\omega_i$和偏置$b$，对其进行初始化（如0均值，1方差）。
 2. 网络正向传播计算网络输出和损失函数。
 3. 根据损失函数进行反向误差传播 （[backpropagation](https://en.wikipedia.org/wiki/Backpropagation)），将网络误差从输出层依次向前传递, 并更新网络中的参数。
 4. 重复2~3步骤，直至网络训练误差达到规定的程度或训练轮次达到设定值。

## 使用PaddlePaddle实现房价预测

以下代码展示了如何使用PaddlePaddle，搭建整个线性回归网络，并使用波士顿房价数据，训练/测试我们构建好的网络，我们将通过代码注释，详细讲解每一步分的作用，源码位于（fit_a_line/trainer.py）：

```python
#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import sys

import math
import numpy

# 引入paddle的python库，包括paddle和paddle.fluid
import paddle
import paddle.fluid as fluid

# save_result函数用于画图，points1 表示预测房价，points2 表示真实房价（GT），我们会把图存在./image/prediction_gt.png中。
def save_result(points1, points2):
    """
    Save the  results into a picture.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x1 = [idx for idx in range(len(points1))]
    y1 = points1
    y2 = points2
    l1 = plt.plot(x1, y1, 'r--', label='predictions')
    l2 = plt.plot(x1, y2, 'g--', label='GT')
    plt.plot(x1, y1, 'ro-', x1, y2, 'g+-')
    plt.title('predictions VS GT')
    plt.legend()
    plt.savefig('./image/prediction_gt.png')

# get_next_batch() 定义了一个数据的生成器（reader），data是一个形状为：batch_size * 14 的python 2D array
# data 的每一行都存储了一个房子的样本，其中，前13列，代表了这个房子的13个特征，最后一列表示了这个房子的真实房价
# 每调用一次 get_next_batch 函数（通过next调用），这个函数就会返回两个 numpy.array
# 第一个 array 是一个形状为 batch_size * 13 的二维数组，代表了这个 batch 的样本
# 第二个 array 是一个长度为 batch_zie 的 1维数组，代表了这个 batch 的真实房价取值
# 除此以外，PaddlePaddle 定义了非常丰富的生成器，以支持大规模复杂场景的神经网络训练，
# 详情参见：http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/data/data_reader_cn.html
def get_next_batch(data, batch_size = 20):
    """
    Genrate next batch for the given reader
    reader: an iterable function, producing examples for training/testing
    batch_size: the batch size for training/testing
    """
    idx = 0
    batch_x = []
    batch_y = []

    while True:
        if idx >= len(data):
            idx = 0

        batch_x.append(data[idx][:-1])
        batch_y.append(data[idx][-1:])

        idx += 1
        if len(batch_x) >= batch_size:
            yield numpy.array(batch_x).astype("float32"), numpy.array(batch_y).astype("float32")
            batch_x = []
            batch_y = []

# 我们程序的入口函数
def main():
    """
    The main function for fitting a line.
    """

    # 我们首先下载训练数据和测试数据，我们可以直接通过 paddle.dataset 来下载波士顿房价数据，
    # 这个数据会被自动存储在 paddle.dataset.uci_housing.UCI_TRAIN_DATA (训练数据) 
    # 和 paddle.dataset.uci_housing.UCI_TEST_DATA (测试数据) 中。
    URL = 'http://paddlemodels.bj.bcebos.com/uci_housing/housing.data'
    MD5 = 'd4accdce7a25600298819f8e28e8d593'
    paddle.dataset.uci_housing.load_data(paddle.dataset.common.download(URL, 'uci_housing', MD5))

    # 然后我们定义一些训练神经网络时需要定义的超参数(hyper-parameters)
    batch_size = 20
    batch_num = len(paddle.dataset.uci_housing.UCI_TRAIN_DATA) // batch_size
    num_epochs = 100
    
    # 这里定义是否使用GPU进行训练和测试
    use_cuda = False

    # 这里定义我们把训练好的模型存储在什么地方
    params_dirname = "fit_a_line.inference.model"
    train_prompt = "Train cost"
    test_prompt = "Test cost"

    # 从这里开始，我们定义计算图
    # 首先，我们先定义2个 place holders，分别是
    # x：代表了一个 batch_size * 13 的 tensor，对应我们数据中的特征部分
    # y：代表了一个 batch_size * 1 的 tensor，对应我们数据中真实房价的部分
    # 注意，由于在神经网络训练的过程中，batch_size可以是动态变化的，因此，我们可以把 batch_size 设置成-1，这时候，PaddlePaddle 会根据输入的数据，实时动态推理真实的 batch_size
    x = fluid.layers.data(name='x', shape=[-1, 13], dtype='float32')
    y = fluid.layers.data(name='y', shape=[-1, 1], dtype='float32')
    # 我们这里定义网络结构，在线性回归模型中，我们可以通过FC (full-conneect)层，来实现线性回归
    # fluid.layers.fc 返回的也是一个 tensor，其中 size 代表了这个 tensor 的输出维度的大小
    # 比如，在这里我们的输入是 batch_size * 13 的 tensor
  	# 那么，我们得到的就是一个 batch_size * 1 的 tensor
    # 详情参考其他op：http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html
    y_predict = fluid.layers.fc(input=x, size=1, act=None)

    # 我们这里定义这个网络的损失函数
 		# 我们使用 square_error_cost 来拟合真实数据
    # 详情参考其他损失函数：http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/layers_cn.html
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(cost)

    # 我们这里定义使用什么优化器来进行参数的更新和优化
    # 详情参考其他优化器：http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/optimizer_cn.html
    sgd_optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001)
    sgd_optimizer.minimize(avg_loss)

    # 在完成网络定义后，我们需要开始训练网络了
    # 在 PaddlePaddle 中，我们通过 program 组织和训练网络，总的来说，PaddlePaddle 有两种网络
    # 1）startup_program：这个 program 负责初始化网络中的所有参数
    # 2）main_program：这个 program 负责执行训练/预测任务
    # 一般来说，PaddlePaddle 会有全局的 startup_program 和 main_program，可以通过 fluid.default_main_program() 和 fluid.default_startup_program() 直接访问这两个program
    # 关于 program 详情参考：http://paddlepaddle.org/documentation/docs/zh/1.4/advanced_usage/design_idea/fluid_design_idea.html
    main_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()

    # 我们这里定义是否使用GPU进行所有计算
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    # Executor(执行器)定义了如何运行一个网络
    # PaddlePaddle对执行器做了比较丰富的优化和封装，让用户可以简单地在单机单卡，单机多卡，多级多卡等不同场景进行深度学习训练和预测
    # 我们这里定义了一个单机单卡的执行器，其他执行器详见：http://paddlepaddle.org/documentation/docs/zh/1.4/api_cn/executor_cn.html
    exe = fluid.Executor(place)
    
    # 我们这里使用startup_program初始化网络中所有定义的参数。
    exe.run(startup_program)
    
    # 现在我们已经完成了准备工作，开始训练吧
    for pass_id in range(num_epochs):
        for step_id in range(batch_num):
						
            # 在训练过程中的每一步，我们首先通过生成器，获取这一步需要的训练数据
            # 在这个例子里，train_x 是一个 batch_size * 13 的 numpy.array，表示了房间的特征：
            # train_x = [[-4.05441001e-02  6.63636327e-02 -3.23562264e-01 -6.91699609e-02
  			#	         -3.43519747e-02  5.56362532e-02 -3.47569622e-02  2.68218592e-02
  			#	         -3.71713340e-01 -2.14193046e-01 -3.35695058e-01  1.01432167e-01
  			#	         -2.11729124e-01]
            #
            #			 ...
            #
           	#			 [-3.24573182e-02 -1.13636367e-01 -1.09852590e-01 -6.91699609e-02
  			#			 -3.43519747e-02 -1.06846981e-01  9.52727906e-03  1.32520276e-04
  			#			 -2.41278574e-01 -1.93200678e-01  2.70687908e-01  8.64288881e-02
  			#            -3.78880575e-02]]
            # train_y 是一个 batch_size * 1 的 numpy.array，表示了真实房价:
            # train_y = [[24. ], [21.6], ..., [18.2]]
            train_x, train_y = next(get_next_batch(paddle.dataset.uci_housing.UCI_TRAIN_DATA, batch_size))
            
            # 这里，我们通过执行器，把从生成器获取到的数据，通过 feed 送给网络，并执行一次梯度更新后，拿到当前网络的loss。
            # 在 feed 中，我们通过一个 dict 定义网络中 placeholder 和 输入数据之间的对应关系
            # 在 PaddlePaddle 中， 我们通过numpy.array或者numpy.ndarray来完成数据的输入和输出
            # fetch_list 定义了我们希望从网络中获取那一步的值，同样也是一个 numpy.array 或者 numpy.ndarray
            # 注意，PaddlePaddle在执行过程中（run函数），会执行整个网络，并不会只执行到 fetch_list 中定义的部分。
            avg_loss_value, = exe.run(
                program = main_program,
                feed = {"x" : train_x, 
                        "y" : train_y},
                fetch_list=[avg_loss])
            
            if step_id % 10 == 0:  # record a train cost every 10 batches
                print("%s, Pass %d, Step %d, Cost %f" %
                      (train_prompt, pass_id, step_id, avg_loss_value[0]))

            if math.isnan(float(avg_loss_value[0])):
                sys.exit("got NaN loss, training failed.")
        
        # 这里我们把训练好的网络，存储在 params_dirname 中。
        # 关于网络的保存和读取，请参见：http://paddlepaddle.org/documentation/docs/zh/1.4/user_guides/howto/training/save_load_variables.html
        if params_dirname is not None:
            # We can save the trained parameters for the inferences later
            fluid.io.save_inference_model(params_dirname, ['x'], [y_predict],
                                          exe)

    
    # 我们这里演示如何读取一个训练好的网络，并完成预测任务。
    # 首先，我们定义预测的执行器
    infer_exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()

    # infer
    with fluid.scope_guard(inference_scope):
      	# 同样的，我们使用 fluid.io.load_inference_model 读取一个已经训练并存储好的网络
        [inference_program, feed_target_names, fetch_targets
         ] = fluid.io.load_inference_model(params_dirname, infer_exe)
        batch_size = 10
				# 我们利用生成器，从测试数据中，获取一个 batch_size 为10的样本，用于测试模型效果
        infer_x, infer_y = next(get_next_batch(paddle.dataset.uci_housing.UCI_TEST_DATA, batch_size))

        assert feed_target_names[0] == 'x'
       
        results = infer_exe.run(
            inference_program,
            feed={feed_target_names[0]: infer_feat},
            fetch_list=fetch_targets)

        # 打印预测结果
        print("infer results: (House Price)")
        for idx, val in enumerate(results[0]):
            print("%d: %.2f" % (idx, val))

        # 打印真实结果
        print("\nground truth:")
        for idx, val in enumerate(infer_label):
            print("%d: %.2f" % (idx, val))
				
        # 将预测结果和真实结果存储到一个图中
        save_result(results[0], infer_label)


if __name__ == '__main__':
    main()

```



## 效果展示

我们使用从[UCI Housing Data Set](http://paddlemodels.bj.bcebos.com/uci_housing/housing.data)获得的波士顿房价数据集进行模型的训练和预测。下面的散点图展示了使用模型对部分房屋价格进行的预测。其中，每个点的横坐标表示同一类房屋真实价格的中位数，纵坐标表示线性回归模型根据特征预测的结果，当二者值完全相等的时候就会落在虚线上。所以模型预测得越准确，则点离虚线越近。

<p align="center">
    <img src = "https://github.com/PaddlePaddle/book/blob/develop/01.fit_a_line/image/predictions.png?raw=true" width=400><br/>
    图1. 预测值 V.S. 真实值
</p>

## 



## 总结
在这章里，我们借助波士顿房价这一数据集，介绍了线性回归模型的基本概念，以及如何使用PaddlePaddle实现训练和测试的过程。

更多关于PaddlePaddle的用法和更多丰富的模型示例，请详见：

## 参考文献

1. https://en.wikipedia.org/wiki/Linear_regression
2. Friedman J, Hastie T, Tibshirani R. The elements of statistical learning[M]. Springer, Berlin: Springer series in statistics, 2001.
3. Murphy K P. Machine learning: a probabilistic perspective[M]. MIT press, 2012.
4. Bishop C M. Pattern recognition[J]. Machine Learning, 2006, 128.

<br/>
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" href="http://purl.org/dc/dcmitype/Text" property="dct:title" rel="dct:type">本教程</span> 由 <a xmlns:cc="http://creativecommons.org/ns#" href="http://book.paddlepaddle.org" property="cc:attributionName" rel="cc:attributionURL">PaddlePaddle</a> 创作，采用 <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">知识共享 署名-相同方式共享 4.0 国际 许可协议</a>进行许可。