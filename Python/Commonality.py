import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ntpath

from statistics import mean, median
from Utils import *
from datetime import datetime


class Commonality:
    def __init__(self, log):
        self._graph = nx.Graph()
        self._mongo = Mongo()

        self._log = log
        self._log_space = 0

    @classmethod
    def load_from_graph(cls, graph, executions_folder, log=True):
        commonality = cls(log)

        commonality._graph = graph

        if commonality._graph.name == '':
            commonality._graph.name = 'NoName'

        commonality._post_load(executions_folder)

        return commonality

    @classmethod
    def load_from_file(cls, graph_file_path, graph_delimiter, executions_folder, log=True):
        commonality = cls(log)

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
        self._count = [1, 1]

        self._arr_for_mongo = [[], []]

        self._community_graph = nx.Graph()

        self._k_array = []

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
        check_folder(self._output_path + '/' + folder_name)

        plt.savefig(self._output_path + '/' + folder_name + '/' + self._graph.name + '_Distribution_D=' + str(d) + '_' + column_name + '.png')

    def _save_arr_k_compare(self, values_arr, k_big_arr, k_small_arr, d, column_name):
        folder_name = 'Compare_To_K'
        check_folder(self._output_path + '/' + folder_name)

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

    def _box_plot(self, df, title, folder_name, file_name):

        values_arr = df['calculation'].tolist()

        inner_df = df[df['is_inner'] == 1][['calculation']]
        outer_df = df[df['is_inner'] == 0][['calculation']]

        values_inner_arr = inner_df['calculation'].tolist()
        values_outer_arr = outer_df['calculation'].tolist()

        data = [values_arr, values_inner_arr, values_outer_arr]

        plt.clf()

        plt.boxplot(data, showfliers=False)
        plt.title(title, fontsize=18)
        plt.xticks([1, 2, 3], ['All Data', 'Inner', 'Outer'])

        check_folder(folder_name)

        plt.savefig(folder_name + '/' + file_name)

    def _box_plot_k(self, df, d, mean, median):
        df['calculation'] = (df['numerator'] ** 2) / (df['denominator_union'] * mean)

        mean_str = str(format(mean, '.2f'))

        title = 'Box Plot Mean=' + mean_str + ' D=' + str(d)
        folder_name = self._output_path + '/' + 'Box_Plot'
        file_name = self._graph.name + '_Box_Plot_Mean=' + mean_str + '_D=' + str(d) + '.png'

        self._box_plot(df, title, folder_name, file_name)

        df['calculation'] = (df['numerator'] ** 2) / (df['denominator_union'] * median)

        title = 'Box Plot Median=' + str(median) + ' D=' + str(d)
        folder_name = self._output_path + '/' + 'Box_Plot'
        file_name = self._graph.name + '_Box_Plot_Median=' + str(median) + '_D=' + str(d) + '.png'

        self._box_plot(df, title, folder_name, file_name)

    def _box_plot_ab(self, df, d, a, b):
        df['calculation'] = (df['numerator'] ** a) / (df['denominator_union'] ** b)

        title = 'Box Plot a=' + str(a) + ' b=' + str(b) + ' D=' + str(d)
        folder_name = self._output_path + '/' + 'Box_Plot'
        file_name = self._graph.name + '_Box_Plot_a=' + str(a) + '_b=' + str(b) + '_D=' + str(d) + '.png'

        self._box_plot(df, title, folder_name, file_name)

    def _process_d_columns(self, d_list, d):
        random.shuffle(d_list)

        df = pd.DataFrame(d_list)

        if 'is_inner' in df.columns:
            self._box_plot_ab(df, d, 2, 2)
            self._box_plot_ab(df, d, 2, 1)
            self._box_plot_ab(df, d, 2, 0)

            self._box_plot_ab(df, d, 1, 2)
            self._box_plot_ab(df, d, 1, 1)
            self._box_plot_ab(df, d, 1, 0)

            self._box_plot_ab(df, d, 0, 2)
            self._box_plot_ab(df, d, 0, 1)
            self._box_plot_ab(df, d, 0, 0)

            self._box_plot_k(df, d, mean(self._k_array), median(self._k_array))

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

    def _calculate_commonality(self, node1, node2):
        return calculate_commonality(self._graph, node1, node2)

    def _compute_commonality(self, node_from, node_to, d):
        numerator, denominator = self._calculate_commonality(node_from, node_to)

        self._arr_for_mongo[d - 1].append({'first': node_from, 'second': node_to, 'index': self._count[d - 1], 'numerator': numerator, 'denominator': denominator})

        if len(self._arr_for_mongo[d - 1]) == 1000000:
            if d == 1:
                self._mongo.d1_insert_many(self._arr_for_mongo[d - 1])
            elif d == 2:
                self._mongo.d2_insert_many(self._arr_for_mongo[d - 1])

            self._arr_for_mongo[d - 1].clear()

        self._count[d - 1] += 1

    def process_commonality(self):
        shortest_path_gen = nx.all_pairs_shortest_path(self._graph, 2)

        for shortest_path in shortest_path_gen:
            node_from = shortest_path[0]
            nodes_to = shortest_path[1]

            for node_to in nodes_to:
                if node_from < node_to:
                    len_path = len(nodes_to[node_to])

                    if len_path == 2:
                        self._compute_commonality(node_from, node_to, 1)

                        # self._k_array.append(len(node_from_neighbors))

                    elif len_path == 3:
                        self._compute_commonality(node_from, node_to, 2)

        if len(self._arr_for_mongo[0]) > 0:
            self._mongo.d1_insert_many(self._arr_for_mongo[0])

        if len(self._arr_for_mongo[1]) > 0:
            self._mongo.d2_insert_many(self._arr_for_mongo[1])
