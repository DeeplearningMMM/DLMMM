# coding=utf-8

from DataStruct.globalConfig import GlobalConfig
def getChannels_in_str():
    result=""
    for i in range(len(GlobalConfig.channels)):
        result=result+str(i)+":"
        result=result+str(GlobalConfig.channels[i])+" "
    return result
#方法中的算子用数字描述
def getFinalModule_in_str():
    result=""
    for eachEdge in GlobalConfig.final_module:
        result=result+"from:"+str(eachEdge.fromIndex)+" to:"+str(eachEdge.toIndex)+" operator:"+str(eachEdge.operator)+"  "
    return result

#方法中的算子用名称描述

def getFinalModule_in_str_2():
    result=""
    for eachEdge in GlobalConfig.final_module:
        result=result+"from:"+str(eachEdge.fromIndex)+" to:"+str(eachEdge.toIndex)+" operator:"+GlobalConfig.basicOps[eachEdge.operator+1]+"  "
    return result