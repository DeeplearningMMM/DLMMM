import json
import os

import copy
import os
import csv
from DataStruct.flatOperatorMap import FlatOperatorMap
from DataStruct.operation import Operator
from DataStruct.globalConfig import GlobalConfig
from DataStruct.edge import edge

def parse_Type(Type):
    if Type in GlobalConfig.basicOps:
        res = GlobalConfig.basicOps.index(Type) - 1
    else:
        res = -1
    return res

def tuopu(Map, size, qidian, dep, target):
    global in_degree

    if qidian[0] == target:
        return dep
    else:
        this_list = []
        for i in qidian:
            for j in range(size):
                if f.Map[i][j].m != 0:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        this_list.append(j)

        if len(this_list) == 0:
            return -1
        else:
            return tuopu(Map, size, this_list, dep + 1, target)


if __name__ == '__main__':
    global in_degree

    # TODO 修改为用于保存结果的文件目录
    result_csv = '../DLMMM_graph_and_precision_bugs.csv'

    with open(result_csv, 'w', encoding= 'utf-8', newline='') as f:
        csv_writer = csv.writer(f)
        data = ['id', '最长通路', '总边数', '最长通路长度与总边数的比例', '总节点数', 'bug来源框架']
        csv_writer.writerow(data)

    # TODO 修改为result.csv所在目录

    file_path = '../new_result.csv'
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        model_structrues = [row[13] for row in reader]
    del model_structrues[0]

    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        diff_tf_torch_max = [row[8] for row in reader]
    del diff_tf_torch_max[0]

    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        diff_torch_mindspore_max = [row[9] for row in reader]
    del diff_torch_mindspore_max[0]

    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        diff_tf_mindspore_max = [row[10] for row in reader]
    del diff_tf_mindspore_max[0]

    # # TODO 取最后K组数据
    # K = 10000
    # model_structrues = model_structrues[len(model_structrues) - K:]
    # diff_tf_torch_max = diff_tf_torch_max[len(diff_tf_torch_max) - K:]
    # diff_torch_mindspore_max = diff_torch_mindspore_max[len(diff_torch_mindspore_max) - K:]
    # diff_tf_mindspore_max = diff_tf_mindspore_max[len(diff_tf_mindspore_max) - K:]

    Round = 0
    tensorflow_bug_num = 0
    pytorch_bug_num = 0
    mindspore_bug_num = 0
    bug_framework = "tensorflow"

    single_bug_num = 0
    multi_bug_num = 0
    for model_structure in model_structrues:

        # 未触发bug，直接下一轮
        if float(max([diff_tf_torch_max[Round],diff_torch_mindspore_max[Round],diff_tf_mindspore_max[Round]])) < 0.15:
            Round += 1
            continue

        # 投票机制判断bug来源。
        if diff_tf_torch_max[Round] >= diff_torch_mindspore_max[Round] and diff_tf_mindspore_max[Round] >= \
                diff_torch_mindspore_max[Round]:
            bug_framework = "tensorflow"
            tensorflow_bug_num += 1
        elif diff_torch_mindspore_max[Round] >= diff_tf_mindspore_max[Round] and diff_tf_torch_max[Round] >= \
                diff_tf_mindspore_max[Round]:
            bug_framework = "torch"
            pytorch_bug_num += 1
        else:
            bug_framework = "mindspore"
            mindspore_bug_num += 1

        Round += 1

        edge_infos = model_structure.split("  ")
        del edge_infos[len(edge_infos) - 1]

        edges = []
        point_num = 0

        for edge_info in edge_infos:
            elements = edge_info.split(" ")
            from_id = 0
            to_id = 0
            op = 0
            for element in elements:
                key = int(element[element.find(":") + 1:])
                if element.__contains__("from"):
                    from_id = key
                    if key > point_num:
                        point_num = key
                if element.__contains__("operator"):
                    op = key
                else:
                    if element.__contains__("to"):
                        to_id = key
                        if key > point_num:
                            point_num = key
            this_edge = edge(from_id, to_id, op)
            edges.append(this_edge)


        point_num += 1
        total_edge = len(edges)

        f = FlatOperatorMap(size=point_num)
        for x in range(f.size):
            for y in range(f.size):
                f.Map[x][y] = Operator(0, 0)

        for each_edge in edges:
            this_i = each_edge.fromIndex
            this_j = each_edge.toIndex
            this_op = each_edge.operator
            f.Map[this_i][this_j] = Operator(0, this_op)

        in_degree = [0] * f.size
        for i in range(f.size):
            for j in range(f.size):
                if f.Map[i][j].m != 0:
                    in_degree[j] += 1

        ans = tuopu(f.Map, f.size, [0], 0, edges[len(edges) - 1].toIndex)
    #
        with open(result_csv, "a+", encoding='utf-8', newline='') as f:
            csv_writer = csv.writer(f)
            data = [Round, ans, total_edge, ans / total_edge, point_num, bug_framework]
            csv_writer.writerow(data)

        print(model_structure)
        print(Round,  ": ", ans, " vs ", total_edge, " rate: ", ans / total_edge, bug_framework)

    float_tf_torch = [float(x) for x in diff_tf_torch_max]
    float_tf_mindspore = [float(x) for x in diff_tf_mindspore_max]
    float_torch_mindspore = [float(x) for x in diff_torch_mindspore_max]
    max_tf_torch_diff = max(float_tf_torch)
    max_tf_mindspore_diff = max(float_tf_mindspore)
    max_torch_mindspore_diff = max(float_torch_mindspore)
    print("max_tf_torch_diff:"+str(max_tf_torch_diff))
    print("max_tf_mindspore_diff:"+str(max_tf_mindspore_diff))
    print("max_torch_mindspore_diff:"+str(max_torch_mindspore_diff))
    print("tensorflow_bug_num:"+str(tensorflow_bug_num))
    print("pytorch_bug_num:"+str(pytorch_bug_num))
    print("mindspore_bug_num:"+str(mindspore_bug_num))

