# coding=utf-8
import os
import traceback
import time

from DataStruct.population import Population
from DataStruct.genetypeQueue import GenetypeQueue
from DataStruct.globalConfig import GlobalConfig
from DataStruct.worker import Worker
from DataStruct.controller import Controller
from Method.initialize import initialize
from Method.util import getFinalModule_in_str,getChannels_in_str
import csv

def globalInit():
    # step1:配置globalConfig
    print("正在初始化globalConfig")
    out = open(file='./' + 'new_result.csv' , mode='w', newline='')
    writer = csv.writer(out,delimiter = ",")
    GlobalConfig.N = 0
    GlobalConfig.flatOperatorMaps = []
    GlobalConfig.resGenetype = []
    GlobalConfig.P = Population()
    GlobalConfig.Q = GenetypeQueue()
    GlobalConfig.final_module = []
    GlobalConfig.channels = []
    GlobalConfig.writer = writer
    writer.writerow(["No","tensorflow_elapsed","torch_elapsed",
                    "mindspore_elapsed","tensorflow_fps",
                     "torch_fps","mindspore_fps",
                     "complexity","diff_tf_torch_max","diff_torch_mindspore_max","diff_tf_mindspore_max","fitness","channels","model","fail_time"])

print("计时开始")
duration = time.time()
globalInit()
print("正在初始化种群")
initialize(GlobalConfig.P)
print("种群初始化完成")
print("开始构建controller节点")
controller = Controller()
print("controller节点构建完成")
print("开始构建worker节点")
worker = Worker()
print("worker节点构建完成")

#主流程
t = 0
print("开始进行突变")
while(t < GlobalConfig.maxMutateTime and (time.time()-duration) < 21600):
    controller.excute()
    try:
        tensor_average, tensorflow_elapsed,torch_elapsed,mindspore_elapsed,tensorflow_fps,torch_fps,mindspore_fps,complexity,diff_tf_torch_max,diff_torch_mindspore_max,diff_tf_mindspore_max,fitness = worker.excute()
        print("第" + str(t) + "轮已经完成")
        GlobalConfig.writer.writerow([str(t),str(tensorflow_elapsed),
                                      str(torch_elapsed),
                                      str(mindspore_elapsed),
                                      str(tensorflow_fps),
                                      str(torch_fps),
                                      str(mindspore_fps),
                                      str(complexity),
                                      str(diff_tf_torch_max),
                                      str(diff_torch_mindspore_max),
                                      str(diff_tf_mindspore_max),
                                      str(fitness),
                                      getChannels_in_str(),
                                      getFinalModule_in_str(),
                                      str(GlobalConfig.fail_time)])
    except Exception as e:
        print("本轮突变失败")
        GlobalConfig.fail_time += 1
        print(traceback.format_exc())
    t = t + 1

print("计时结束")
duration = time.time()-duration
print("耗时:"+str(duration)+"秒")
# #最后的筛选
# while(len(GlobalConfig.resGenetype) < GlobalConfig.resultNum):
#     controller.excute()
#     thisg=GlobalConfig.Q.pop()
#     GlobalConfig.resGenetype.append(thisg)