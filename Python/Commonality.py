import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import ntpath
import random
from pymongo import UpdateOne
from Mongo import *
from datetime import datetime


class Commonality:
    def __init__(self):
        self._graph = nx.Graph()
        self._mongo = Mongo()

    @classmethod
    def load_from_graph(cls, graph, executions_folder):
        commonality = cls()

        commonality._graph = graph

        if commonality._graph.name == '':
            commonality._graph.name = 'NoName'

        commonality._post_load(executions_folder)

        return commonality

    @classmethod
    def load_from_file(cls, graph_file_path, graph_delimiter, executions_folder):
        commonality = cls()

        file = open(graph_file_path, 'r')

        for line in file:
            split = line.split(graph_delimiter)

            split0 = int(split[0])
            split1 = int(split[1].replace('\n', ''))

            assert split0 != split, 'loop: ' + split0 + ' ' + split1

            commonality._graph.add_edge(split0, split1)

        file.close()

        file_name = ntpath.basename(graph_file_path)
        commonality._graph.name = file_name[:file_name.rfind('.')]

        commonality._post_load(executions_folder)

        return commonality

    def _post_load(self, executions_folder):
        self._graph = nx.k_core(self._graph, 2)

        self._output_path = executions_folder + '/' + self._graph.name + '_' + str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
        os.makedirs(self._output_path)

        self._commonality_init()

    def _commonality_init(self):
        self._d1_count = 1
        self._d2_count = 1

        self._d1_arr_for_mongo = []
        self._d2_arr_for_mongo = []

        self._d1_additions = self._get_commonality_additions(1)
        self._d2_additions = self._get_commonality_additions(2)

    def _get_tuple(self, a, b):
        if a < b:
            return a, b
        return b, a

    def _get_commonality_additions(self, d):
        if d == 1:
            return 2, 2
        return 0, 2

    def _check_folder(self, folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    def _save_d(self, d_list, d):
        file_name_txt = self._graph.name + '_D=' + str(d) + '.txt'
        file_name_csv = self._graph.name + '_D=' + str(d) + '.csv'

        file_txt = open(self._output_path + '/' + file_name_txt, 'a+')
        file_csv = open(self._output_path + '/' + file_name_csv, 'a+')

        file_csv.write('Node1, Node2, Commonality (Union), Inner/Outer\n')

        for d_row in d_list:
            node1 = str(d_row['first'])
            node2 = str(d_row['second'])
            commonality_union = str(round(d_row['commonality_union'], 3))

            is_inner_txt = ''
            is_inner_csv = ''

            if 'is_inner' in d_row:
                if d_row['is_inner'] == 1:
                    is_inner_txt = '\tIs Inner: Yes'
                    is_inner_csv = 'Inner'
                else:
                    is_inner_txt = '\tIs Inner: No'
                    is_inner_csv = 'Outer'

            file_txt.write('Nodes: (' + node1 + ', ' + node2 + ')\tCommonality: Union = ' + commonality_union + is_inner_txt + '\n')
            file_csv.write(node1 + ', ' + node2 + ', ' + commonality_union + is_inner_csv + '\n')

        file_txt.close()
        file_csv.close()

    def _save_distribution(self, values_arr, d, column_name):
        plt.clf()

        counts, bins = np.histogram(values_arr, bins=np.arange(0, 1, 0.01))
        counts = counts / sum(counts)

        plt.hist(bins[:-1], bins, weights=counts)
        plt.xticks(np.arange(0, 1, 0.01))
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Distribution D=' + str(d) + ', Compute By=' + column_name)

        folder_name = 'Distribution'
        self._check_folder(self._output_path + '/' + folder_name)

        plt.savefig(self._output_path + '/' + folder_name + '/' + self._graph.name + '_Distribution_D=' + str(d) + '_' + column_name + '.png')

    def _save_arr_k_compare(self, values_arr, k_big_arr, k_small_arr, d, column_name):
        folder_name = 'Compare_To_K'
        self._check_folder(self._output_path + '/' + folder_name)

        plt.clf()
        plt.plot(values_arr, k_big_arr, 'bo', ms=1)

        plt.title(column_name + ' vs K Bigger, D=' + str(d) + ', Data=all', fontsize=18)
        plt.xlabel('Values By ' + column_name, fontsize=18)
        plt.ylabel('K Bigger', fontsize=18)

        plt.savefig(self._output_path + '/' + folder_name + '/' + self._graph.name + '_Compare_' + column_name + '_K_Bigger_All_D=' + str(d) + '.png')

        plt.clf()
        plt.plot(values_arr, k_small_arr, 'bo', ms=1)

        plt.title(column_name + ' vs K Smaller, D=' + str(d) + ', Data=all', fontsize=18)
        plt.xlabel('Values By ' + column_name, fontsize=18)
        plt.ylabel('K Smaller', fontsize=18)

        plt.savefig(self._output_path + '/' + folder_name + '/' + self._graph.name + '_Compare_' + column_name + '_K_Smaller_All_D=' + str(d) + '.png')

        values_arr = values_arr[:1000]
        k_big_arr = k_big_arr[:1000]
        k_small_arr = k_small_arr[:1000]

        plt.clf()
        plt.plot(values_arr, k_big_arr, 'bo', ms=1)

        plt.title(column_name + ' vs K Bigger, D=' + str(d) + ', Data=1000', fontsize=18)
        plt.xlabel('Values By ' + column_name, fontsize=18)
        plt.ylabel('K Bigger', fontsize=18)

        plt.savefig(self._output_path + '/' + folder_name + '/' + self._graph.name + '_Compare_' + column_name + '_K_Bigger_1000_D=' + str(d) + '.png')

        plt.clf()
        plt.plot(values_arr, k_small_arr, 'bo', ms=1)

        plt.title(column_name + ' vs K Smaller, D=' + str(d) + ', Data=1000', fontsize=18)
        plt.xlabel('Values By ' + column_name, fontsize=18)
        plt.ylabel('K Smaller', fontsize=18)

        plt.savefig(self._output_path + '/' + folder_name + '/' + self._graph.name + '_Compare_' + column_name + '_K_Smaller_1000_D=' + str(d) + '.png')

    def _box_plot(self, df, d, a, b):

        df['calculation'] = (df['numerator'] ** a) / (df['denominator_union'] ** b)

        values_arr = df['calculation'].tolist()

        plt.clf()

        inner_df = df[df['is_inner'] == 1][['calculation']]
        outer_df = df[df['is_inner'] == 0][['calculation']]

        values_inner_arr = inner_df['calculation'].tolist()
        values_outer_arr = outer_df['calculation'].tolist()

        data = [values_arr, values_inner_arr, values_outer_arr]

        plt.boxplot(data)
        plt.title('Box Plot a=' + str(a) + ' b=' + str(b) + ' D=' + str(d), fontsize=18)
        plt.xticks([1, 2, 3], ['All Data', 'Inner', 'Outer'])

        folder_name = 'Box_Plot'
        self._check_folder(self._output_path + '/' + folder_name)

        plt.savefig(self._output_path + '/' + folder_name + '/' + self._graph.name + '_Box_Plot_a=' + str(a) + '_b=' + str(b) + '_D=' + str(d) + '.png')

    def _process_d_columns(self, d_list, d):
        random.shuffle(d_list)

        df = pd.DataFrame(d_list)

        if 'is_inner' in df.columns:
            self._box_plot(df, d, 1, 1)
            self._box_plot(df, d, 1, 0.5)
            self._box_plot(df, d, 1, 0)

            self._box_plot(df, d, 0.5, 1)
            self._box_plot(df, d, 0.5, 0.5)
            self._box_plot(df, d, 0.5, 0)

            self._box_plot(df, d, 0, 1)
            self._box_plot(df, d, 0, 0.5)
            self._box_plot(df, d, 0, 0)

        count_for_avg = 1

        avg_union_list = []

        k_big_arr = []
        k_small_arr = []

        avg_union_total = 0

        for d_row in d_list:
            avg_union_total = avg_union_total + d_row['commonality_union']

            avg_union_list.append(avg_union_total / count_for_avg)

            count_for_avg = count_for_avg + 1

            len_neighbors1 = len(list(self._graph.neighbors(d_row['first'])))
            len_neighbors2 = len(list(self._graph.neighbors(d_row['second'])))

            if len_neighbors1 > len_neighbors2:
                k_big_arr.append(len_neighbors1)
                k_small_arr.append(len_neighbors2)
            else:
                k_big_arr.append(len_neighbors2)
                k_small_arr.append(len_neighbors1)

        d_list.clear()

        values_union_arr = df['commonality_union']

        self._save_arr_k_compare(values_union_arr, k_big_arr, k_small_arr, d, 'union')

        self._save_distribution(values_union_arr, d, 'union')

    def _process_d(self, d_list, d):
        self._save_d(d_list, d)

        self._process_d_columns(d_list, d)

    def _process_partial_d(self, d_count, d, mongo_get_index_range_ptr, get_by_index_list_ptr):
        interval = 25000000

        for i in range(1, int(d_count / interval) + 1):
            partial_d = list(mongo_get_index_range_ptr(interval * (i - 1) + 1, i * interval + 1))
            self._save_d(partial_d, d)
            partial_d.clear()
        partial_d = list(mongo_get_index_range_ptr(d_count - (d_count % interval) + 1, d_count + 1))
        self._save_d(partial_d, d)

        random_index = {}
        while len(random_index) < 1000000:
            rand = random.randint(0, d_count)
            if rand not in random_index:
                random_index[rand] = True

        random_index_list = list(random_index.keys())
        random_index.clear()

        d_partial_random_list = list(get_by_index_list_ptr(random_index_list))
        self._process_d_columns(d_partial_random_list, d)

    def save_to_file(self):
        max_size_for_full_process = 100000000

        d1_count = self._mongo.d1_get_count()

        if d1_count <= max_size_for_full_process:
            d_list = list(self._mongo.d1_get())
            self._process_d(d_list, 1)
        else:
            self._process_partial_d(d1_count, 1, self._mongo.d1_get_index_range, self._mongo.d1_get_by_index_list)

        d2_count = self._mongo.d2_get_count()

        if d2_count <= max_size_for_full_process:
            d_list = list(self._mongo.d2_get())
            self._process_d(d_list, 2)
        else:
            self._process_partial_d(d2_count, 2, self._mongo.d2_get_index_range, self._mongo.d2_get_by_index_list)

    def _compute_commonality(self, node_from, list_node_from_neighbors, node_to, d, additions_arr):  # add parameters arr_for_mongo, d1_insert_many
        node_to_neighbors = self._graph.neighbors(node_to)

        list_node_to_neighbors = list(node_to_neighbors)

        len_l1 = len(list_node_from_neighbors)
        len_l2 = len(list_node_to_neighbors)

        numerator = len(list(set(list_node_from_neighbors).intersection(list_node_to_neighbors))) + additions_arr[0]

        denominator_union = len_l1 + len_l2 - numerator + additions_arr[1]

        if d == 1:
            arr_for_mongo = self._d1_arr_for_mongo

        elif d == 2:
            arr_for_mongo = self._d2_arr_for_mongo

        arr_for_mongo.append(UpdateOne({'first': node_from, 'second': node_to}, {'$set': {'numerator': numerator, 'denominator_union': denominator_union}}))

        if len(arr_for_mongo) == 1000000:
            if d == 1:
                self._mongo.d1_bulk_write_update_many(arr_for_mongo)
            elif d == 2:
                self._mongo.d2_bulk_write_update_many(arr_for_mongo)

            arr_for_mongo.clear()

    def process_commonality(self):
        shortest_path_gen = nx.all_pairs_shortest_path(self._graph, 2)

        for shortest_path in shortest_path_gen:
            node_from = shortest_path[0]
            node_from_neighbors = list(self._graph.neighbors(node_from))

            nodes_to = shortest_path[1]

            for node_to in nodes_to:
                if node_from < node_to:
                    len_path = len(nodes_to[node_to])

                    if len_path == 2:
                        self._compute_commonality(node_from, node_from_neighbors, node_to, 1, self._d1_additions)
                    elif len_path == 3:
                        self._compute_commonality(node_from, node_from_neighbors, node_to, 2, self._d2_additions)

        if len(self._d1_arr_for_mongo) > 0:
            self._mongo.d1_bulk_write_update_many(self._d1_arr_for_mongo)

        if len(self._d2_arr_for_mongo) > 0:
            self._mongo.d2_bulk_write_update_many(self._d2_arr_for_mongo)
