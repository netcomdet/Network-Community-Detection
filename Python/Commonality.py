import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ntpath

from statistics import mean, median
from Utils import *
from datetime import datetime


class Commonality:
    def __init__(self):
        self._graph = nx.Graph()
        self._mongo = Mongo()

        self.MAX_SIZE_FOR_FULL_PROCESS = 100000000

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

        self._output_path = executions_folder + self._graph.name + '_' + str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S")) + '\\'
        os.makedirs(self._output_path)

        self._commonality_init()

    def output_path(self):
        return self._output_path

    def get_graph(self):
        return self._graph

    def _commonality_init(self):
        self._count = [1, 1]

        self._arr_for_mongo = [[], []]

    def _save_d(self, d_list, d):
        file_name_txt = self._graph.name + '_D=' + str(d) + '.txt'
        file_name_csv = self._graph.name + '_D=' + str(d) + '.csv'

        file_txt = open(self._output_path + '/' + file_name_txt, 'a+')
        file_csv = open(self._output_path + '/' + file_name_csv, 'a+')

        file_csv.write('Node1, Node2, Commonality, Inner/Outer\n')

        for d_row in d_list:
            node1 = str(d_row['first'])
            node2 = str(d_row['second'])

            commonality = str(round(d_row['numerator'] / d_row['denominator'], 3))

            is_inner_txt = ''
            is_inner_csv = ''

            if 'is_inner' in d_row:
                if d_row['is_inner'] == 1:
                    is_inner_txt = '\tIs Inner: Yes'
                    is_inner_csv = ',Inner'
                else:
                    is_inner_txt = '\tIs Inner: No'
                    is_inner_csv = ',Outer'

            file_txt.write('Nodes: (' + node1 + ', ' + node2 + ')\tCommonality = ' + commonality + is_inner_txt + '\n')
            file_csv.write(node1 + ',' + node2 + ',' + commonality + is_inner_csv + '\n')

        file_txt.close()
        file_csv.close()

    def _save_distribution(self, values_arr, d):
        plt.clf()

        counts, bins = np.histogram(values_arr, bins=np.arange(0, 1, 0.01))
        counts = counts / sum(counts)

        plt.hist(bins[:-1], bins, weights=counts)
        plt.xticks(np.arange(0, 1, 0.01))
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Distribution D=' + str(d))

        folder_name = 'Distribution'
        check_folder(self._output_path + '/' + folder_name)

        plt.savefig(self._output_path + '/' + folder_name + '/' + self._graph.name + '_Distribution_D=' + str(d) + '.png')

    def _save_arr_k_compare(self, values_arr, k_big_arr, k_small_arr, d):
        folder_name = 'Compare_To_K'
        check_folder(self._output_path + '/' + folder_name)

        plt.clf()
        plt.plot(values_arr, k_big_arr, 'bo', ms=1)

        plt.title('Commonality vs K Bigger, D=' + str(d) + ', Data=all', fontsize=18)
        plt.xlabel('Commonality', fontsize=18)
        plt.ylabel('K Bigger', fontsize=18)

        plt.savefig(self._output_path + '/' + folder_name + '/' + self._graph.name + '_Compare_K_Bigger_All_D=' + str(d) + '.png')

        plt.clf()
        plt.plot(values_arr, k_small_arr, 'bo', ms=1)

        plt.title('Commonality vs K Smaller, D=' + str(d) + ', Data=all', fontsize=18)
        plt.xlabel('Commonality', fontsize=18)
        plt.ylabel('K Smaller', fontsize=18)

        plt.savefig(self._output_path + '/' + folder_name + '/' + self._graph.name + '_Compare_K_Smaller_All_D=' + str(d) + '.png')

        values_arr = values_arr[:1000]
        k_big_arr = k_big_arr[:1000]
        k_small_arr = k_small_arr[:1000]

        plt.clf()
        plt.plot(values_arr, k_big_arr, 'bo', ms=1)

        plt.title('Commonality vs K Bigger, D=' + str(d) + ', Data=1000', fontsize=18)
        plt.xlabel('Commonality', fontsize=18)
        plt.ylabel('K Bigger', fontsize=18)

        plt.savefig(self._output_path + '/' + folder_name + '/' + self._graph.name + '_Compare_K_Bigger_1000_D=' + str(d) + '.png')

        plt.clf()
        plt.plot(values_arr, k_small_arr, 'bo', ms=1)

        plt.title('Commonality vs K Smaller, D=' + str(d) + ', Data=1000', fontsize=18)
        plt.xlabel('Commonality', fontsize=18)
        plt.ylabel('K Smaller', fontsize=18)

        plt.savefig(self._output_path + '/' + folder_name + '/' + self._graph.name + '_Compare_K_Smaller_1000_D=' + str(d) + '.png')

    def _box_plot(self, df, title, folder_name, file_name):

        # values_arr = df['calculation'].tolist()

        inner_df = df[df['is_inner'] == 1][['calculation']]
        outer_df = df[df['is_inner'] == 0][['calculation']]

        values_inner_arr = inner_df['calculation'].tolist()
        values_outer_arr = outer_df['calculation'].tolist()

        # data = [values_arr, values_inner_arr, values_outer_arr]
        data = [values_inner_arr, values_outer_arr]

        plt.clf()

        # plt.boxplot(data, showfliers=False) - with outliers
        plt.boxplot(data)
        plt.title(title, fontsize=18)
        # plt.xticks([1, 2, 3], ['All Data', 'Inner', 'Outer'])
        plt.xticks([1, 2], ['In Community', 'Not In Community'])

        check_folder(folder_name)

        plt.savefig(folder_name + '/' + file_name)

    def _box_plot_k(self, df, d, k_mean, k_median):
        df['calculation'] = (df['numerator'] ** 2) / (df['denominator'] * mean)

        k_mean_str = str(format(k_mean, '.2f'))

        title = 'Box Plot Mean=' + k_mean_str + ' D=' + str(d)
        folder_name = self._output_path + '/' + 'Box_Plot'
        file_name = self._graph.name + '_Box_Plot_Mean=' + k_mean_str + '_D=' + str(d) + '.png'

        self._box_plot(df, title, folder_name, file_name)

        df['calculation'] = (df['numerator'] ** 2) / (df['denominator'] * k_median)

        title = 'Box Plot Median=' + str(k_median) + ' D=' + str(d)
        folder_name = self._output_path + '/' + 'Box_Plot'
        file_name = self._graph.name + '_Box_Plot_Median=' + str(k_median) + '_D=' + str(d) + '.png'

        self._box_plot(df, title, folder_name, file_name)

    def _box_plot_ab(self, df, d, a, b):
        df['calculation'] = (df['numerator'] ** a) / (df['denominator'] ** b)

        # title = 'Box Plot a=' + str(a) + ' b=' + str(b) + ' D=' + str(d)
        if d == 1:
            title = 'A. Commonality Box Plot Distance = ' + str(d)
        else:
            title = 'B. Commonality Box Plot Distance = ' + str(d)
        folder_name = self._output_path + '/' + 'Box_Plot'
        file_name = self._graph.name + '_Box_Plot_a=' + str(a) + '_b=' + str(b) + '_D=' + str(d) + '.png'

        self._box_plot(df, title, folder_name, file_name)

    def _process_d_columns(self, d_list, d):
        random.shuffle(d_list)

        df = pd.DataFrame(d_list)

        if 'is_inner' in df.columns:
            ''''self._box_plot_ab(df, d, 2, 2)
            self._box_plot_ab(df, d, 2, 1)
            self._box_plot_ab(df, d, 2, 0)

            self._box_plot_ab(df, d, 1, 2)'''
            self._box_plot_ab(df, d, 1, 1)
            '''self._box_plot_ab(df, d, 1, 0)

            self._box_plot_ab(df, d, 0, 2)
            self._box_plot_ab(df, d, 0, 1)
            self._box_plot_ab(df, d, 0, 0)'''

            # self._box_plot_k(df, d, mean(self._k_array), median(self._k_array))

        count_for_avg = 1

        avg_list = []

        k_big_arr = []
        k_small_arr = []

        avg_total = 0

        for d_row in d_list:
            avg_total += d_row['numerator'] / d_row['denominator']

            avg_list.append(avg_total / count_for_avg)

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

        values_arr = df['numerator'] / df['denominator']

        self._save_arr_k_compare(values_arr, k_big_arr, k_small_arr, d)

        self._save_distribution(values_arr, d)

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
        d1_count = self._mongo.d1_get_count()

        if d1_count <= self.MAX_SIZE_FOR_FULL_PROCESS:
            d_list = list(self._mongo.d1_get())
            self._process_d(d_list, 1)
        else:
            self._process_partial_d(d1_count, 1, self._mongo.d1_get_index_range, self._mongo.d1_get_by_index_list)

        d2_count = self._mongo.d2_get_count()

        if d2_count <= self.MAX_SIZE_FOR_FULL_PROCESS:
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

                    elif len_path == 3:
                        self._compute_commonality(node_from, node_to, 2)

        if len(self._arr_for_mongo[0]) > 0:
            self._mongo.d1_insert_many(self._arr_for_mongo[0])

        if len(self._arr_for_mongo[1]) > 0:
            self._mongo.d2_insert_many(self._arr_for_mongo[1])

    def _plot_alg(self, algorithm_and_d, both, inner, outer):
        plt.clf()

        data = [both, inner, outer]
        plt.boxplot(data, showfliers=False)
        plt.title(self._graph.name + '_' + algorithm_and_d, fontsize=18)
        plt.xticks([1, 2, 3], ['All', 'Inner', 'Outer'])

        plt.savefig(self._graph.name + '_' + algorithm_and_d + '.png')

        plt.clf()

        data = [both, inner, outer]
        plt.boxplot(data, showfliers=True)
        plt.title(self._graph.name + '_' + algorithm_and_d, fontsize=18)
        plt.xticks([1, 2, 3], ['All', 'Inner', 'Outer'])

        plt.savefig(self._graph.name + '_' + algorithm_and_d + '_with_outliers.png')

    def box_plot_3_commonality(self):
        z = round(2 * len(self._graph.edges) / len(self._graph.nodes), 2)

        alg1_d_1_inner = []
        alg1_d_1_outer = []
        alg1_d_1_all = []

        alg1_d_2_inner = []
        alg1_d_2_outer = []
        alg1_d_2_all = []

        alg1_d_all_inner = []
        alg1_d_all_outer = []
        alg1_d_all_all = []

        alg2_d_1_inner = []
        alg2_d_1_outer = []
        alg2_d_1_all = []

        alg2_d_2_inner = []
        alg2_d_2_outer = []
        alg2_d_2_all = []

        alg2_d_all_inner = []
        alg2_d_all_outer = []
        alg2_d_all_all = []

        alg3_d_1_inner = []
        alg3_d_1_outer = []
        alg3_d_1_all = []

        alg3_d_2_inner = []
        alg3_d_2_outer = []
        alg3_d_2_all = []

        alg3_d_all_inner = []
        alg3_d_all_outer = []
        alg3_d_all_all = []

        d1_count = self._mongo.d1_get_count()

        if d1_count <= self.MAX_SIZE_FOR_FULL_PROCESS:
            d1 = self._mongo.d1_get()
        else:
            random_index = {}
            while len(random_index) < 1000000:
                rand = random.randint(0, d1_count)
                if rand not in random_index:
                    random_index[rand] = True

            random_index_list = list(random_index.keys())
            random_index.clear()

            d1 = list(self._mongo.d1_get_by_index_list(random_index_list))

        for d1_row in d1:
            numerator = d1_row['numerator']
            # denominator = d1_row['denominator_union']
            denominator = d1_row['denominator']

            alg1 = numerator
            alg2 = numerator / denominator
            alg3 = (numerator * numerator) / (z * denominator)

            alg1_d_1_all.append(alg1)
            alg1_d_all_all.append(alg1)

            alg2_d_1_all.append(alg2)
            alg2_d_all_all.append(alg2)

            alg3_d_1_all.append(alg3)
            alg3_d_all_all.append(alg3)

            if d1_row['is_inner'] == 1:
                alg1_d_1_inner.append(alg1)
                alg1_d_all_inner.append(alg1)

                alg2_d_1_inner.append(alg2)
                alg2_d_all_inner.append(alg2)

                alg3_d_1_inner.append(alg3)
                alg3_d_all_inner.append(alg3)
            else:
                alg1_d_1_outer.append(alg1)
                alg1_d_all_outer.append(alg1)

                alg2_d_1_outer.append(alg2)
                alg2_d_all_outer.append(alg2)

                alg3_d_1_outer.append(alg3)
                alg3_d_all_outer.append(alg3)

        d2_count = self._mongo.d2_get_count()

        if d2_count <= self.MAX_SIZE_FOR_FULL_PROCESS:
            d2 = self._mongo.d2_get()
        else:
            random_index = {}
            while len(random_index) < 1000000:
                rand = random.randint(0, d1_count)
                if rand not in random_index:
                    random_index[rand] = True

            random_index_list = list(random_index.keys())
            random_index.clear()

            d2 = list(self._mongo.d2_get_by_index_list(random_index_list))

        for d2_row in d2:
            numerator = d2_row['numerator']
            # denominator = d2_row['denominator_union']
            denominator = d2_row['denominator']

            alg1 = numerator
            alg2 = numerator / denominator
            alg3 = (numerator * numerator) / (z * denominator)

            alg1_d_2_all.append(alg1)
            alg1_d_all_all.append(alg1)

            alg2_d_2_all.append(alg2)
            alg2_d_all_all.append(alg2)

            alg3_d_2_all.append(alg3)
            alg3_d_all_all.append(alg3)

            if d2_row['is_inner'] == 1:
                alg1_d_2_inner.append(alg1)
                alg1_d_all_inner.append(alg1)

                alg2_d_2_inner.append(alg2)
                alg2_d_all_inner.append(alg2)

                alg3_d_2_inner.append(alg3)
                alg3_d_all_inner.append(alg3)
            else:
                alg1_d_2_outer.append(alg1)
                alg1_d_all_outer.append(alg1)

                alg2_d_2_outer.append(alg2)
                alg2_d_all_outer.append(alg2)

                alg3_d_2_outer.append(alg3)
                alg3_d_all_outer.append(alg3)

        print('Alg 1:')
        print(median(alg1_d_all_all))
        print(max(alg1_d_all_all))
        print(median(alg1_d_all_inner))
        print(max(alg1_d_all_inner))
        print(median(alg1_d_all_outer))
        print(max(alg1_d_all_outer))
        print('Alg 2:')
        print(median(alg2_d_all_all))
        print(max(alg2_d_all_all))
        print(median(alg2_d_all_inner))
        print(max(alg2_d_all_inner))
        print(median(alg2_d_all_outer))
        print(max(alg2_d_all_outer))
        print('Alg 3:')
        print(median(alg3_d_all_all))
        print(max(alg3_d_all_all))
        print(median(alg3_d_all_inner))
        print(max(alg3_d_all_inner))
        print(median(alg3_d_all_outer))
        print(max(alg3_d_all_outer))

        '''self._plot_alg('Algorithm1_dall', alg1_d_all_all, alg1_d_all_inner, alg1_d_all_outer)
        self._plot_alg('Algorithm1_d1', alg1_d_1_all, alg1_d_1_inner, alg1_d_1_outer)
        self._plot_alg('Algorithm1_d2', alg1_d_2_all, alg1_d_2_inner, alg1_d_2_outer)

        self._plot_alg('Algorithm2_dall', alg2_d_all_all, alg2_d_all_inner, alg2_d_all_outer)
        self._plot_alg('Algorithm2_d1', alg2_d_1_all, alg2_d_1_inner, alg2_d_1_outer)
        self._plot_alg('Algorithm2_d2', alg2_d_2_all, alg2_d_2_inner, alg2_d_2_outer)

        self._plot_alg('Algorithm3_dall_z_' + str(z), alg3_d_all_all, alg3_d_all_inner, alg3_d_all_outer)
        self._plot_alg('Algorithm3_d1_z_' + str(z), alg3_d_1_all, alg3_d_1_inner, alg3_d_1_outer)
        self._plot_alg('Algorithm3_d2_z_' + str(z), alg3_d_2_all, alg3_d_2_inner, alg3_d_2_outer)'''

    def box_plot_3_commonality2(self):
        z = 2 * len(self._graph.edges) / len(self._graph.nodes)

        alg1_d_1_inner = []
        alg1_d_1_outer = []
        alg1_d_1_all = []

        alg2_d_1_inner = []
        alg2_d_1_outer = []
        alg2_d_1_all = []

        alg3_d_1_inner = []
        alg3_d_1_outer = []
        alg3_d_1_all = []

        for edge in self._graph.edges:
            if edge[0] < edge[1]:
                node1 = edge[0]
                node2 = edge[1]
            else:
                node1 = edge[1]
                node2 = edge[0]

            numerator, denominator = calculate_commonality(self._graph, node1, node2)

            alg1 = numerator
            alg2 = numerator / denominator
            alg3 = (numerator * numerator) / (z * denominator)

            alg1_d_1_all.append(alg1)

            alg2_d_1_all.append(alg2)

            alg3_d_1_all.append(alg3)

            path = 'c:\\University\\Thesis\\Network-Community-Detection\\Data\\' + self._graph.name + 'Cmty\\1\\' + str(node1) + '.txt'

            if os.path.exists(path):
                file = open(path, mode='r')
                community = file.read()
                file.close()

                len_community = len(community)

                first_index = community.index('\n')
                first_node = community[0: first_index]

                if first_index + 1 < len_community:
                    last_index = community[0: -1].rindex('\n')
                    last_node = community[last_index + 1: len(community) - 1]
                else:
                    last_node = ''
            else:
                community = ''
                first_node = ''
                last_node = ''

            str_neighbor = str(node2)

            if str_neighbor == first_node or '\n' + str_neighbor + '\n' in community or str_neighbor == last_node:
                alg1_d_1_inner.append(alg1)
                alg2_d_1_inner.append(alg2)
                alg3_d_1_inner.append(alg3)

            else:
                alg1_d_1_outer.append(alg1)
                alg2_d_1_outer.append(alg2)
                alg3_d_1_outer.append(alg3)

        self._plot_alg('Algorithm1_d1', alg1_d_1_all, alg1_d_1_inner, alg1_d_1_outer)
        self._plot_alg('Algorithm2_d1', alg2_d_1_all, alg2_d_1_inner, alg2_d_1_outer)
        self._plot_alg('Algorithm3_d1_', alg3_d_1_all, alg3_d_1_inner, alg3_d_1_outer)
