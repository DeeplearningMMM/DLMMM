import copy
import csv
import os
import torch
import torch.nn as nn
from torchstat import stat
# 注：需要将torchstat库中的print(report)改成 return report
import re
import time
import numpy as np

from DataStruct.globalConfig import GlobalConfig
from DataStruct.flatOperatorMap import FlatOperatorMap
from DataStruct.operation import Operator
from Method.module_executor import exe_module

class edge:
    fromIndex = 0
    toIndex = 0
    # 为了方便格式转化，且涉及的操作仅为基本操作，故只保留操作号，不保留层号
    operator = 0
    index = ""
    def __init__(self, FromIndex, ToIndex, Operator):
        self.fromIndex = FromIndex
        self.toIndex = ToIndex
        self.operator = Operator


def parse_Type(Type):
    if Type in GlobalConfig.basicOps:
        res = GlobalConfig.basicOps.index(Type) - 1
    else:
        res = -1
    return res

def Decode(type, ch):
    res = 1
    same_channel_operators = [-1, 2, 4, 5, 8, 9, 10, 11, 12, 13, 14]
    if type in same_channel_operators:
        res = ch

    return res

def search_zero(in_degree, size):
    for i in range(size):
        if in_degree[i] == 0:
            return i
    return -1

def decodeChannel(f):
    global mainPath
    global branches
    #注：输入类型为flatOperaotrMap

    #先把f.chanels扩大
    f.channels = [0]*f.size
    f.channels[0] = 1
    in_degree = [0]*f.size
    for j in range(f.size):
        for i in range(f.size):
            if f.Map[i][j].m != 0:
                in_degree[j] += 1

    #最多拓扑f.size轮
    for times in range(f.size):
        # 找到入度为0的点
        target = search_zero(in_degree, f.size)
        if target < 0:
            print("Error! Circle exits!")
            return


        in_degree[target] = -1
        for j in range(f.size):
            if f.Map[target][j].m != 0:

                in_degree[j] -= 1
                f.channels[j] += Decode(f.Map[target][j].m, f.channels[target])
                Operation = f.Map[target][j].m
    return


def find_next_float(text, keyword):
    # 构建正则表达式，查找特定关键词后的浮点数
    pattern = rf'{keyword}\s*([-+]?\d*\.\d+|\d+)'
    match = re.search(pattern, text)

    if match:
        return float(match.group(1))  # 提取并转换为浮点数
    else:
        return None  # 如果未找到匹配的浮点数，返回 None

class thisModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
def str2model(model_str):
    # 统计模型基本信息（e.g., channels）
    operators = model_str.split(" ")
    if operators[-1][-1] in [',','\n']:
        operators=operators[:-1]
    node_num = int(operators[-1].split(',')[1][3:])
    model_flatmap = FlatOperatorMap(size=node_num + 1)
    final_model = []
    for x in range(model_flatmap.size):
        for y in range(model_flatmap.size):
            model_flatmap.Map[x][y] = Operator(0, 0)
    for each_operator in operators:
        if each_operator == '"\n':
            continue
        eachstr = each_operator.split(',')
        if eachstr[0]=="\n":
            continue
        fromIndex = int(eachstr[0][5:])
        toIndex = int(eachstr[1][3:])
        type = parse_Type(eachstr[2][9:])
        model_flatmap.Map[fromIndex][toIndex] = Operator(0, type)
        final_model.append(eachstr[2][9:])
    decodeChannel(model_flatmap)
    channels = model_flatmap.channels

    # 构建模型
    operatornum = 0
    model = torch.nn.Sequential()
    for operator_type in final_model:
        if operator_type=='identity':
            operatornum += 1
            continue
        elif operator_type=='1*1':
            model.add_module(str(operatornum),nn.Conv2d(in_channels=channels[operatornum]*3,out_channels=3,kernel_size=1))
            operatornum += 1
            continue
        elif operator_type=='depthwise_conv2D':
            model.add_module(str(operatornum),nn.Conv2d(in_channels=channels[operatornum]*3,out_channels=3,kernel_size=3,groups=channels[operatornum]*3,padding=1))
            operatornum += 1
            continue
        elif operator_type=='separable_conv2D':
            model.add_module(str(operatornum),nn.Conv2d(in_channels=channels[operatornum]*3,out_channels=3,kernel_size=3,groups=channels[operatornum]*3,padding=1))
            model.add_module(str(operatornum)+'point',nn.Conv2d(in_channels=channels[operatornum]*3,out_channels=3,kernel_size=1))
            operatornum += 1
            continue
        elif operator_type=='max_pooling2D':
            model.add_module(str(operatornum),nn.MaxPool2d(kernel_size=3,padding=1))
            operatornum += 1
            continue
        elif operator_type=='average_pooling2D':
            model.add_module(str(operatornum),nn.AvgPool2d(kernel_size=3,padding=1))
            operatornum += 1
            continue
        elif operator_type=='conv2D':
            model.add_module(str(operatornum),nn.Conv2d(in_channels=channels[operatornum]*3,out_channels=3,kernel_size=3,padding=1))
            operatornum += 1
            continue
        elif operator_type=='conv2D_transpose':
            model.add_module(str(operatornum),nn.ConvTranspose2d(in_channels=channels[operatornum]*3,out_channels=3,kernel_size=3,padding=1))
            operatornum += 1
            continue
        elif operator_type=='ReLU':
            model.add_module(str(operatornum),nn.ReLU(inplace=False))
            operatornum += 1
            continue
        elif operator_type=='sigmoid':
            model.add_module(str(operatornum),nn.Sigmoid())
            operatornum += 1
            continue
        elif operator_type=='tanh':
            model.add_module(str(operatornum),nn.Tanh())
            operatornum += 1
            continue
        elif operator_type=='leakyReLU':
            model.add_module(str(operatornum),nn.LeakyReLU())
            operatornum += 1
            continue
        elif operator_type=='PReLU':
            model.add_module(str(operatornum),nn.PReLU())
            operatornum += 1
            continue
        elif operator_type=='ELU':
            model.add_module(str(operatornum),nn.ELU())
            operatornum += 1
            continue
        elif operator_type=='SELU':
            model.add_module(str(operatornum),nn.SELU())
            operatornum += 1
            continue
        elif operator_type=='batch_normalization':
            model.add_module(str(operatornum),nn.BatchNorm2d(num_features=channels[operatornum]*3))
            operatornum += 1
            continue
        else:
            operatornum += 1
            continue
    return operatornum, model

def count_measurement(model_str):
    operatornum, model = str2model(model_str)
    report = stat(model, (3, 224, 224)) # 统计模型的参数量和FLOPs，（3,224,224）是输入图像的size
    report = report.replace(",", "")
    #params
    params = find_next_float(report, "Total params: ")
    # memory(MB)
    memory = find_next_float(report, "Total memory: ")
    # FLOPs(MFlops)
    FLOPs = find_next_float(report, "Total Flops: ")
    # MemR+W(MB)
    memoryRW = find_next_float(report, "Total MemR\+W: ")
    input_corpus = np.random.randn(1, 3, 224, 224)
    torch_input = torch.Tensor(input_corpus).type(dtype=torch.float32)
    torch_start_time = time.perf_counter()
    model(torch_input)
    # time (seconds)
    torch_elapsed = time.perf_counter() - torch_start_time
    print(operatornum, params, memory, FLOPs, memoryRW, torch_elapsed)
    return operatornum, params, memory, FLOPs, memoryRW, torch_elapsed

GlobalConfig.dataset = 'random'
f1 = open('./gandalf_model.csv', encoding = 'utf-8')
with open('./gandalf_measurement.csv', "a+", encoding='utf-8', newline='') as f2:
    csv_writer = csv.writer(f2)
    # data = ["operatornum", "params", "memory", "FLOPs", "memoryRW", "torch_elapsed", "bug_performance"]
    data = ["time","model_variety","bug_performance"]
    csv_writer.writerow(data)
model_num = 0
while True:
    print(model_num)
    model_num += 1
    this_model_str = f1.readline()
    # operatornum, params, memory, FLOPs, memoryRW, torch_elapsed = count_measurement(this_model_str)
    if this_model_str == "":
        break
    operators = this_model_str.split(" ")
    if operators[-1][-1] in [',','\n']:
        operators = operators[:-1]
    node_num = int(operators[-1].split(',')[1][3:])
    this_model = FlatOperatorMap(size=node_num+1)
    final_model = []
    for x in range(this_model.size):
        for y in range(this_model.size):
            this_model.Map[x][y] = Operator(0, 0)
    for each_operator in operators:
        if each_operator =='\n':
            continue
        eachstr = each_operator.split(',')
        fromIndex = int(eachstr[0][5:])
        toIndex = int(eachstr[1][3:])
        type = parse_Type(eachstr[2][9:])
        this_model.Map[fromIndex][toIndex] = Operator(0,type)
        final_model.append(edge(FromIndex=fromIndex,ToIndex=toIndex,Operator=type))
    decodeChannel(this_model)
    GlobalConfig.final_module = copy.deepcopy(final_model)
    GlobalConfig.channels = copy.deepcopy(this_model.channels)

    (tensor_average,tensorflow_elapsed,torch_elapsed,mindspore_elapsed,
     tensorflow_fps,torch_fps,mindspore_fps,complexity,diff_1_max,diff_2_max,diff_3_max) = exe_module()

    current_path = os.path.dirname(__file__)
    os.chdir(current_path)

    result_csv = './gandalf_measurement.csv'
    bug_performance = max([diff_1_max,diff_2_max,diff_3_max])
    print(bug_performance)
    with open(result_csv, "a+", encoding='utf-8', newline='') as f2:
        csv_writer = csv.writer(f2)
        # data = [operatornum, params, memory, FLOPs, memoryRW, torch_elapsed, bug_performance]
        data = [torch_elapsed, complexity, bug_performance]
        csv_writer.writerow(data)
# count_measurement("from:0,to:1,operator:identity from:1,to:2,operator:1*1 from:2,to:3,operator:depthwise_conv2D from:3,to:4,operator:separable_conv2D from:4,to:5,operator:max_pooling2D from:5,to:6,operator:average_pooling2D from:6,to:7,operator:conv2D from:7,to:8,operator:conv2D_transpose from:8,to:9,operator:ReLU from:9,to:10,operator:sigmoid from:10,to:11,operator:tanh from:11,to:12,operator:leakyReLU from:12,to:13,operator:PReLU from:13,to:14,operator:ELU from:14,to:15,operator:SELU from:15,to:16,operator:batch_normalization ")
# count_measurement("from:0 to:1 operator:identity  from:1 to:2 operator:max_pooling2D  from:1 to:3 operator:identity  from:2 to:6 operator:identity  from:3 to:4 operator:identity  from:4 to:5 operator:identity  from:5 to:10 operator:separable_conv2D  from:6 to:7 operator:identity  from:7 to:8 operator:identity  from:8 to:10 operator:separable_conv2D  from:10 to:11 operator:identity  from:11 to:12 operator:identity  from:12 to:13 operator:identity  ")


