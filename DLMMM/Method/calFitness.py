# coding=utf-8
import copy
import math

import numpy
import numpy as np

from .flatMap import toFlatMap
from DataStruct.globalConfig import GlobalConfig

def cal_fitness_value(matrix):
    thismatrix = copy.deepcopy(matrix)
    '''计算对比强度'''
    V = np.std(thismatrix, axis=0)

    '''计算冲突性'''
    A2 = numpy.transpose(thismatrix)  # 矩阵转置
    r = np.corrcoef(A2)  # 求皮尔逊相关系数
    # 如果数据全都相同，那么皮尔逊相关系数为NaN，这时证明这个error一点新的信息没有，不往下计算了，error_score直接返回0.
    if numpy.isnan(r).any():
        return 0.0

    f = np.sum(1 - r, axis=1)

    '''计算信息承载量'''
    C = V * f

    '''计算权重'''
    w = C / np.sum(C)
    #注：当所有数值均为0.0时，皮尔逊相关系数均为1，此时信息承载量C为NaN，直接返回error_score为1即可。(赋值0而非1是为了保持算法稳定性，因为全是相同的极小量时会归一化到1.0)
    if numpy.isnan(w).any():
        return 1.0

    '''计算得分(0,1之间)'''
    s = np.dot(thismatrix, w)
    Score = s / max(s)
    return Score[-1],s

# 根据本轮的信息和GlobalConfig中的判断矩阵，计算本轮的fitness。
def exe_calculate():
    # 先进行初始化initMutateTime次，fitness值一律赋值为1e-9,既初始化了种子模型池，又初始化了判断矩阵。
    if len(GlobalConfig.judge_matrix)<=GlobalConfig.initMutateTime:
        print("初始轮次，本轮fitness为:1e-9")
        return 1e-9
    # Step0:逐列数据标准化
    judge_matrix = copy.deepcopy(np.array(GlobalConfig.judge_matrix))
    for j in range(0,len(judge_matrix[0])):
        square_sum = 0.0
        #  求平方和
        for i in range(0,len(judge_matrix)):
            # NaN不参与标准化，最后直接赋值
            if math.isnan(judge_matrix[i][j]):
                continue
            # 0.0不参与标准化，最后直接赋值
            if judge_matrix[i][j]<=1e-9:
                continue
            square_sum += judge_matrix[i][j]**2
        # 按列标准化,0.0就是0.0,
        for i in range(0,len(judge_matrix)):
            # NaN不参与标准化，直接赋1
            if math.isnan(judge_matrix[i][j]):
                judge_matrix[i][j]=1.0
                continue
            # 0.0不参与标准化，直接赋值1e-9
            if judge_matrix[i][j]<=1e-9:
                judge_matrix[i][j]=1e-9
                continue
            judge_matrix[i][j] = judge_matrix[i][j]/(1.0*math.sqrt(square_sum))
    # Step1:从judge_matrix分别提取出两类指标的状态矩阵
    fps_matrix = copy.deepcopy(judge_matrix[:,0:3])
    error_matrix = copy.deepcopy(judge_matrix[:,4:7])

    fps_score,fps_fused_matrix = cal_fitness_value(fps_matrix)
    complexity_score = judge_matrix[-1][3]
    error_score,error_fused_matrix = cal_fitness_value(error_matrix)

    total_fused_matrix = numpy.vstack((fps_fused_matrix,judge_matrix[:,3],error_fused_matrix))
    total_fitness_score,total_fused_result_matrix = cal_fitness_value(total_fused_matrix)
    if GlobalConfig.ablation_study_mode == "random":
        fitness = 1e-9
    elif GlobalConfig.ablation_study_mode == "DLMMM":
        fitness = total_fitness_score
    elif GlobalConfig.ablation_study_mode == "performance":
        fitness = error_score
    elif GlobalConfig.ablation_study_mode == "operator":
        fitness = complexity_score
    elif GlobalConfig.ablation_study_mode == "time":
        fitness = fps_score
    else:
        fitness = 1e-9
        print("unsupported abalation mode!")
    # fitness = (fps_score+complexity_score+error_score)/3.0
    # print("本轮fps_score为:"+str(fps_score))
    # print("本轮complexity_score为:"+str(complexity_score))
    # print("本轮error_score为:"+str(error_score))
    print("本轮fitness为:"+str(fitness))
    # print("random模式fitness不反馈")
    return fitness
    # return 1.0
def calFitness(g):
    #todo
    # 执行模型前，需要解析通道数和模型结构，改写GlobalConfig里面的channel和final_module
    toFlatMap(g)
    #一定要用到的时候再import，否则会初始化错误的值
    from Method.module_executor import exe_module
    tensor_average,tensorflow_elapsed,torch_elapsed,mindspore_elapsed,tensorflow_fps,torch_fps,mindspore_fps,complexity,diff_1_max,diff_2_max,diff_3_max=exe_module()
    # 执行完之后，judge_matrix的最后一行就是本轮的数据，根据judge_matrix可以计算本轮的fitness。
    fitness = exe_calculate()
    return tensor_average,tensorflow_elapsed,torch_elapsed,mindspore_elapsed,tensorflow_fps,torch_fps,mindspore_fps,complexity,diff_1_max,diff_2_max,diff_3_max,fitness
