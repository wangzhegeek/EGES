
import pandas as pd
import numpy as np
from itertools import chain
import pickle
import time
import networkx as nx
from node2vec import Node2Vec

def cnt_session(data, time_cut=30, cut_type=2):
    sku_list = data['sku_id']
    time_list = data['action_time']
    type_list = data['type']
    session = []
    tmp_session = []
    for i, item in enumerate(sku_list):
        if type_list[i] == cut_type or (i < len(sku_list)-1 and (time_list[i+1] - time_list[i]).seconds/60 > time_cut)\
                or i == len(sku_list)-1:
            tmp_session.append(item)
            session.append(tmp_session)
            tmp_session = []
        else:
            tmp_session.append(item)
    return session


def get_session(action_data, use_type=None):
    if use_type is None:
        use_type = [1, 2, 3, 5]
    action_data = action_data[action_data['type'].isin(use_type)]
    action_data = action_data.sort_values(by=['user_id', 'action_time'], ascending=True)
    group_action_data = action_data.groupby('user_id').agg(list)
    session_list = group_action_data.apply(cnt_session, axis=1)
    return session_list.to_numpy()

if __name__ == '__main__':
    data_path = './data/'
    action_data = pd.read_csv(data_path + 'action_head.csv', parse_dates=['action_time']).drop('module_id', axis=1).dropna()
    print('make session list\n')
    start_time = time.time()
    session_list = get_session(action_data, use_type=[1, 2, 3, 5])
    session_list_all = []
    for item_list in session_list:
        for session in item_list:
            if len(session) > 1:
                session_list_all.append(session)
    with open('data_cache/session_list', 'wb') as f:
        pickle.dump(session_list_all, f)

    print('make session list done, time cost {0}'.format(str(time.time() - start_time)))

    # session2graph
    node_pair = dict()
    for session in session_list_all:
        for i in range(1, len(session)):
            if (session[i-1], session[i]) not in node_pair.keys():
                node_pair[(session[i - 1], session[i])] = 1
            else:
                node_pair[(session[i-1], session[i])] += 1

    in_node_list = list(map(lambda x: x[0], list(node_pair.keys())))
    out_node_list = list(map(lambda x: x[1], list(node_pair.keys())))
    weight_list = list(node_pair.values())
    graph_df = pd.DataFrame({'in_node': in_node_list, 'out_node': out_node_list, 'weight': weight_list})
    graph_df.to_csv('./data_cache/graph.csv', sep=' ', index=False, header=False)

    G = nx.read_edgelist('./data_cache/graph.csv', create_using=nx.DiGraph(), nodetype=None,
                         data=[('weight', int)])
    model = Node2Vec(G, 10, 80, workers=4, p=0.25, q=2)

    session_reproduce = model.get_sentences()
    session_reproduce = list(filter(lambda x: len(x) > 2, session_reproduce))

    # add side info
    product_data = pd.read_csv(data_path + 'jdata_product.csv').drop('market_time', axis=1).dropna()

    all_skus = set()
    for session in session_reproduce:
        for item in session:
            all_skus.add(int(item))
    all_nodes_df = pd.DataFrame({'sku_id': list(all_skus)})
    all_nodes_df = pd.merge(all_nodes_df, product_data, on='sku_id', how='left').fillna(0)
    all_nodes_df[all_nodes_df.columns] = all_nodes_df[all_nodes_df.columns].astype(int)

    # id2index
    sku_dict = dict(enumerate(item for item in all_nodes_df['sku_id'].unique()))
    sku_dict = {v: k for k, v in sku_dict.items()}
    brand_dict = dict(enumerate(item for item in all_nodes_df['brand'].unique()))
    brand_dict = {v: k for k, v in brand_dict.items()}
    shop_dict = dict(enumerate(item for item in all_nodes_df['shop_id'].unique()))
    shop_dict = {v: k for k, v in shop_dict.items()}
    cate_dict = dict(enumerate(item for item in all_nodes_df['cate'].unique()))
    cate_dict = {v: k for k, v in cate_dict.items()}

    side_info_dict = dict(map(lambda x: (sku_dict[x[0]], (brand_dict[x[1]], shop_dict[x[2]], cate_dict[x[3]])), all_nodes_df.to_numpy()))

    session_reproduce2 = []
    for session in session_reproduce:
        tmp_lst = []
        for item in session:
            tmp_lst.append(sku_dict[int(item)])
        session_reproduce2.append(tmp_lst)

    with open('./data_cache/session_reproduce', 'wb') as f:
        pickle.dump(session_reproduce2, f)

    with open('./data_cache/side_info', 'wb') as f:
        pickle.dump(side_info_dict, f)
