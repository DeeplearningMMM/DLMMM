# encoding = 'utf-8'
import os
import time

from DataStruct.globalConfig import GlobalConfig
import random
import numpy as np
import tensorflow as tf
import mindspore
import mindspore.nn
#mindspore设置为动态图模式。
import mindspore.context
mindspore.context.set_context(mode=mindspore.context.PYNATIVE_MODE, device_target="CPU")
import torch



# class edge:
#     fromIndex = 0
#     toIndex = 0
#     # 为了方便格式转化，且涉及的操作仅为基本操作，故只保留操作号，不保留层号
#     operator = 0
#     index = ""
#     def __init__(self, FromIndex, ToIndex, Operator):
#         self.fromIndex = FromIndex
#         self.toIndex = ToIndex
#         self.operator = Operator

def get_input_tensor(x, dtype, environment):
    if environment=="tensorflow":
        tensor_NCHW=tf.convert_to_tensor(x, dtype=dtype)
        tensor_NHWC=tf.transpose(tensor_NCHW,[0,2,3,1])
        return tensor_NHWC
    if environment=="pytorch":
        return torch.Tensor(x).type(dtype=dtype)
    if environment=="mindspore":
        return mindspore.Tensor(x).astype(dtype=dtype)

def exe_module():
    #一定要运行时再import，否则上来就会初始化错误的模型
    from Method.Models.general_testnet_pytorch import GeneralTorchNet as TorchNet
    from Method.Models.general_testnet_tensorflow import GeneralTFNet as TFNet
    from Method.Models.general_testnet_mindspore import GeneralMindsporeNet as MindsporeNet

    # GlobalConfig.final_module =[edge(FromIndex=0,ToIndex=1,Operator=1),
    #                 edge(FromIndex=0,ToIndex=2,Operator=2),
    #                 edge(FromIndex=0,ToIndex=3,Operator=3),
    #                 edge(FromIndex=0,ToIndex=4,Operator=4),
    #                 edge(FromIndex=0,ToIndex=5,Operator=5),
    #                 edge(FromIndex=0,ToIndex=6,Operator=6),
    #                 edge(FromIndex=0,ToIndex=7,Operator=7),
    #                 edge(FromIndex=0,ToIndex=8,Operator=8),
    #                 edge(FromIndex=0,ToIndex=9,Operator=9),
    #                 edge(FromIndex=0,ToIndex=10,Operator=10),
    #                 edge(FromIndex=0,ToIndex=11,Operator=11),
    #                 edge(FromIndex=0,ToIndex=12,Operator=12),
    #                 edge(FromIndex=0,ToIndex=13,Operator=13),
    #                 edge(FromIndex=0,ToIndex=14,Operator=15),
    #                 edge(FromIndex=1,ToIndex=2,Operator=-1),
    #                 edge(FromIndex=2,ToIndex=3,Operator=2),
    #                 edge(FromIndex=3,ToIndex=4,Operator=3),
    #                 edge(FromIndex=4,ToIndex=5,Operator=4),
    #                 edge(FromIndex=5,ToIndex=6,Operator=5),
    #                 edge(FromIndex=6,ToIndex=7,Operator=6),
    #                 edge(FromIndex=7,ToIndex=8,Operator=7),
    #                 edge(FromIndex=8,ToIndex=9,Operator=8),
    #                 edge(FromIndex=9,ToIndex=10,Operator=9),
    #                 edge(FromIndex=10,ToIndex=11,Operator=10),
    #                 edge(FromIndex=11,ToIndex=12,Operator=11),
    #                 edge(FromIndex=12,ToIndex=13,Operator=12),
    #                 edge(FromIndex=13,ToIndex=14,Operator=13),
    #                 edge(FromIndex=14,ToIndex=15,Operator=15),
    #                 edge(FromIndex=15,ToIndex=16,Operator=16)
    #                 ]
    final_module = GlobalConfig.final_module
    activation_types = ["relu","sigmoid","tanh","leakyrelu","prelu","elu"]
    activation = activation_types[random.randint(0,len(activation_types)-1)]
    # GlobalConfig.channels = [1,1,2,3,2,3,4,2,2,3,4,5,6,7,8,8,8]
    channels = GlobalConfig.channels




    #准备输入的numpy
    n,c,h,w = 0,0,0,0
    input_corpus = None
    if GlobalConfig.dataset == 'random':
        n = GlobalConfig.batch
        c = GlobalConfig.c0
        h = GlobalConfig.h
        w = GlobalConfig.w

        input_corpus = np.random.randn(n, c, h, w)
    else:
        # 这两句用于处理main中os调用的错误。
        current_path = os.path.dirname(__file__)
        os.chdir(current_path)
        data = np.load('../Dataset/'+GlobalConfig.dataset+'/inputs.npz')
        input_corpus =data[data.files[0]]
        GlobalConfig.batch = input_corpus.shape[0]
        GlobalConfig.c0 = input_corpus.shape[1]
        GlobalConfig.h = input_corpus.shape[2]
        GlobalConfig.w = input_corpus.shape[3]
        n = GlobalConfig.batch
        c = GlobalConfig.c0
        h = GlobalConfig.h
        w = GlobalConfig.w



    # 根据globalconfig中的final_module,统计模型包含的子算子及算子组合结构个数complexity
    # 结果列表长度为subgraph_Level，表示大小从1到subgraph_Level的各层次子图个数。
    # 将结果列表各层次非同构子结构数求和即可得到复杂度complexity，含义为模型包含的带权重非同构子结构个数。
    from Method.get_all_connected_subgraph import get_complexity
    complexity = get_complexity()

    torch_input = get_input_tensor(input_corpus, dtype = torch.float32, environment = "pytorch")
    # torch_net = TorchNet(channels=channels,final_module=final_module,in_channel=c,activation_type=activation)
    torch_net = TorchNet(channels=channels,final_module=final_module,in_channel=c)
    torch_start_time = time.perf_counter()
    torch_output = torch_net(torch_input)
    torch_elapsed = time.perf_counter() - torch_start_time
    torch_output_numpy = torch_output.detach().numpy()

    tensorflow_input = get_input_tensor(input_corpus, dtype = tf.float32, environment = "tensorflow")
    # tensorflow_net = TFNet(channels=channels,final_module=final_module,in_channel=c,activation_type=activation)
    tensorflow_net = TFNet(channels=channels,final_module=final_module,in_channel=c)
    tensorflow_start_time = time.perf_counter()
    tensorflow_output = tensorflow_net(tensorflow_input)
    tensorflow_elapsed = time.perf_counter() - tensorflow_start_time

    tensorflow_output_numpy = tf.transpose(tensorflow_output,[0,3,1,2]).numpy()


    #mindspore

    mindspore_input = get_input_tensor(input_corpus, dtype = mindspore.float32, environment = "mindspore")
    # mindspore_net = MindsporeNet(channels=channels,final_module=final_module,in_channel=c,activation_type=activation)
    mindspore_net = MindsporeNet(channels=channels,final_module=final_module,in_channel=c)
    mindspore_start_time = time.perf_counter()
    mindspore_output = mindspore_net(mindspore_input)
    mindspore_elapsed = time.perf_counter() - mindspore_start_time

    mindspore_output_numpy = mindspore_output.asnumpy()

    # elapsed是以秒为单位的时间浮点数，根据elapsed可以计算每秒能够处理的图片数目fps(frame_per_second)。
    tensorflow_fps = 1.0/(tensorflow_elapsed/(GlobalConfig.batch*1.0))
    torch_fps = 1.0/(torch_elapsed/(GlobalConfig.batch*1.0))
    mindspore_fps = 1.0/(mindspore_elapsed/(GlobalConfig.batch*1.0))

    diff_numpy_1 = torch_output_numpy - tensorflow_output_numpy
    diff_numpy_2 = torch_output_numpy - mindspore_output_numpy
    diff_numpy_3 = tensorflow_output_numpy - mindspore_output_numpy

    diff_1_max = np.max(np.abs(diff_numpy_1))
    # diff_1_mean = np.mean(np.abs(diff_numpy_1))
    diff_2_max = np.max(np.abs(diff_numpy_2))
    # diff_2_mean = np.mean(np.abs(diff_numpy_2))
    diff_3_max = np.max(np.abs(diff_numpy_3))
    # diff_3_mean = np.mean(np.abs(diff_numpy_3))
    # avg_diff_max = (diff_1_max+diff_2_max+diff_3_max)/3.0
    # avg_diff_mean = (diff_1_mean + diff_2_mean + diff_3_mean)/3.0

    # 将本次模型的相关状态加入判断矩阵judge_matrix
    GlobalConfig.judge_matrix.append([tensorflow_fps,torch_fps,mindspore_fps,complexity,diff_1_max,diff_2_max,diff_3_max])
    print([tensorflow_fps,torch_fps,mindspore_fps,complexity,diff_1_max,diff_2_max,diff_3_max])
    tensor_average = torch.mean(torch_input)
    return tensor_average,tensorflow_elapsed,torch_elapsed,mindspore_elapsed,tensorflow_fps,torch_fps,mindspore_fps,complexity,diff_1_max,diff_2_max,diff_3_max
# print(exe_module())

