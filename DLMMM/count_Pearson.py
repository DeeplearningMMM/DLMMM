import csv

import numpy

time = []
operator_combination_variety = []
bug_detection_performance = []
# operatornum = []
# params = []
# memory = []
# FLOPs = []
# memoryRW = []
# torch_elapsed = []
# bug_performance = []

# with open('./lemon_result.csv', 'r') as csvfile:
#     reader = csv.reader(csvfile)
#     next(reader)  # 跳过标题行
#     for row in reader:
#         operatornum.append(float(row[0]))
#         params.append(float(row[1]))
#         memory.append(float(row[2]))
#         FLOPs.append(float(row[3]))
#         memoryRW.append(float(row[4]))
#         torch_elapsed.append(float(row[5]))
#         bug_performance.append(float(row[6]))
#     print(numpy.corrcoef(operatornum,bug_performance))
#     print(numpy.corrcoef(params,bug_performance))
#     print(numpy.corrcoef(memory,bug_performance))
#     print(numpy.corrcoef(FLOPs,bug_performance))
#     print(numpy.corrcoef(memoryRW,bug_performance))
#     print(numpy.corrcoef(torch_elapsed,bug_performance))

with open('./gandalf_measurement.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # 跳过标题行
    for row in reader:
        time.append(float(row[0]))
        operator_combination_variety.append(float(row[1]))
        bug_detection_performance.append(float(row[2]))
    print(numpy.corrcoef(time,operator_combination_variety))
    print(numpy.corrcoef(time,bug_detection_performance))
    print(numpy.corrcoef(operator_combination_variety,bug_detection_performance))
