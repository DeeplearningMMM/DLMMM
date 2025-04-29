import networkx as nx
import itertools
# (0,1), (0,2), (0,3), (2, 5), (5,4), (4,1)
# e = []
# G = nx.DiGraph()
# G.add_edge(0,1,score='1')
# G.add_edge(0,2,score='1')
# G.add_edge(0,3,score='6')
# G.add_edge(0,4,score='4')
# G.add_edge(0,5,score='1')
# G.add_edge(4,1,score='2')

# 从globalconfig的final_module edge列表中解析出networkx格式的图:
def get_networkx_graph():
    import DataStruct.globalConfig as globalConfig
    this_module = globalConfig.GlobalConfig.final_module
    G = nx.DiGraph()
    edge_num = len(this_module)
    for each_edge in this_module:
        G.add_edge(each_edge.fromIndex,each_edge.toIndex,score = each_edge.operator)
    return G,edge_num

# 统计各层次非同构子图个数
# 输入：networkx带权重有向图G，G的总边数edge_edge
# 输出：列表长度为edge_num，表示大小从1到subgraph_Level的各层次非同构子图个数。
def get_all_insomorphic_connected_subgraphs_number(G,edge_num):
    all_insomorphic_connected_subgraphs = []
    from DataStruct.globalConfig import GlobalConfig
    # 求全部非同构子图
    # here we ask for all connected subgraphs that have at least 2 nodes AND have no more nodes than subgraph_Level+1
    for nb_nodes in range(2, GlobalConfig.subgraph_Level + 2):
        for SG in (G.subgraph(selected_nodes) for selected_nodes in itertools.combinations(G, nb_nodes)):
            if nx.is_weakly_connected(SG):
                # 判断是否已经存在同构子图
                already_have = False
                for each_graph in all_insomorphic_connected_subgraphs:
                    GM = nx.algorithms.isomorphism.is_isomorphic(each_graph, SG, edge_match= lambda e1,e2: e1['score'] == e2['score'])
                    if GM:
                        already_have = True
                        break
                if already_have == False:
                    # 如果不存在带权重同构子图，则将该子图其加入结果集

                    # 查看结果边集
                    # print(SG.edges())
                    all_insomorphic_connected_subgraphs.append(SG)

    # 统计不同大小的非同构子图个数,键为边数，值为个数。
    subgraph_num = {}
    for each_subgraph in all_insomorphic_connected_subgraphs:
        if len(each_subgraph.edges()) not in subgraph_num:
            subgraph_num[len(each_subgraph.edges())] = 1
        else:
            subgraph_num[len(each_subgraph.edges())] += 1
    # print(subgraph_num)

    # 字典转数组返回，列表长度为subgraph_Level，表示大小从1到subgraph_Level的各层次子图个数。
    res = []
    for i in range(1,GlobalConfig.subgraph_Level+1):
        if i in subgraph_num.keys():
            res.append(subgraph_num[i])
        else:
            res.append(0)
    # 查看结果
    # print(res)
    return res
# get_all_insomorphic_connected_subgraphs_number(G=G,edge_num=6)

def get_complexity():
    G,edge_num = get_networkx_graph()
    # 最后返回结果是subgraph_Level层次非同构子结构数的总和
    return sum(get_all_insomorphic_connected_subgraphs_number(G,edge_num))
