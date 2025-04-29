# coding=utf-8
import math
import random

from DataStruct.globalConfig import GlobalConfig
class Worker:
    a=0
    #todo
    def __init__(self):
        return
    def excute(self):
        g=GlobalConfig.Q.pop()
        #一定要运行的时候再import
        from Method.calFitness import calFitness
        tensor_average,tensorflow_elapsed,torch_elapsed,mindspore_elapsed,tensorflow_fps,torch_fps,mindspore_fps,complexity,diff_tf_torch_max,diff_torch_mindspore_max,diff_tf_mindspore_max,fitness=calFitness(g)

        #TODO feedback
        #基本算子的反馈
        if g.mutateL == 0 and GlobalConfig.mode != 1:
            # Handle with Operator 0, Operator 0 is meaningless
            # g.fitness记录的是突变前的fitness，fitness的变化量是本次突变策略的评价。
            if g.mutateM != 0:
                GlobalConfig.basicWeights[g.mutateM + 1] += fitness - g.fitness
                if GlobalConfig.basicWeights[g.mutateM + 1] < 1e-6:
                    GlobalConfig.basicWeights[g.mutateM + 1] = 1e-6
                # Give a new weight to Operator 0
                total = 0
                for i in range(len(GlobalConfig.basicWeights)):
                    if i == 1:
                        continue
                    total += GlobalConfig.basicWeights[i]
                # TODO 删边的概率占基本操作整体概率的1/n
                GlobalConfig.basicWeights[1] = total / (len(GlobalConfig.basicOps) - 1)
        #复合算子的反馈
        if g.mutateL > 0 and GlobalConfig.mode != 0:
            g.weights[g.mutateL - 1][g.mutateM - 1] += fitness - g.fitness
            if g.weights[g.mutateL - 1][g.mutateM - 1] < 1e-6:
                g.weights[g.mutateL - 1][g.mutateM - 1] = 1e-6
# 更新genetype中记录的fitness值
        g.fitness = fitness
        g.mutateM = -2
        g.mutateL = -2
        # 将突变后的genetype加入population
        GlobalConfig.P.append(g)
        return tensor_average,tensorflow_elapsed,torch_elapsed,mindspore_elapsed,tensorflow_fps,torch_fps,mindspore_fps,complexity,diff_tf_torch_max,diff_torch_mindspore_max,diff_tf_mindspore_max,fitness