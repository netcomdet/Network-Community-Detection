import matplotlib.pyplot as plt
import itertools

from Commonality import *
import random
import os
from community import community_louvain
from communities.algorithms import hierarchical_clustering, bron_kerbosch, girvan_newman
from Mongo import Mongo
from pymongo import UpdateOne
from copy import copy, deepcopy
import networkx as nx

PRINT_LOG = False


def plot_distribution(distribution):
    tt = {}
    for i in distribution:
        if i in tt:
            tt[i] = tt[i] + 1
        else:
            tt[i] = 1

    counts, bins = np.histogram(list(tt.values()))
    plt.stairs(counts, bins)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.show()


def check_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def calculate_commonality(g, node1, node2):
    node1_neighbors = list(g.neighbors(node1))
    node2_neighbors = list(g.neighbors(node2))

    if g.has_edge(node1, node2):
        numerator_addition = 2
    else:
        numerator_addition = 0

    numerator = len(list(set(node1_neighbors).intersection(node2_neighbors))) + numerator_addition

    denominator = len(node1_neighbors) + len(node2_neighbors) - numerator + 2

    return numerator, denominator


def preprocess_community(file_path, output_folder):
    file = open(file_path, 'r')

    for line in file:

        split = line.replace('\n', '').split('\t')

        if len(split) > 2:
            node1 = split.pop(0)

            while split:
                file_to_write = open(output_folder + node1 + '.txt', 'a')

                for i in range(len(split)):
                    node2 = split[i]
                    file_to_write.write(node1 + ' ' + node2 + '\n')

                file_to_write.close()
                node1 = split.pop(0)

    file.close()


def distinct_community(source_folder, destination_folder):
    for file in os.listdir(source_folder):
        file_community = open(source_folder + file, 'r')

        community_list = []
        community_list_dict = {}

        for line in file_community:
            community = int(line.replace('\n', '').split(' ')[1])
            if community not in community_list_dict:
                community_list_dict[community] = True
                community_list.append(community)

        file_community.close()

        community_list.sort()

        file_new = open(destination_folder + file, 'a')

        for community in community_list:
            file_new.write(str(community) + '\n')

        file_new.close()


def prepare_community_by_d(graph_path, delimiter, community_folder, community_new_folder):
    g = nx.Graph()

    file = open(graph_path, 'r')

    for line in file:
        split = line.split(delimiter)

        split0 = int(split[0])
        split1 = int(split[1].replace('\n', ''))

        assert split0 != split, 'loop: ' + split0 + ' ' + split1

        g.add_edge(split0, split1)

    file.close()

    g = nx.k_core(g, 2)

    shortest_path_gen = nx.all_pairs_shortest_path(g, 2)

    count = 0

    for shortest_path in shortest_path_gen:
        node_from = shortest_path[0]
        node_from_str = str(node_from)
        node_from_file_name = community_folder + node_from_str + '.txt'

        if count % 10000 == 0:
            print(count)

        count = count + 1

        if os.path.isfile(node_from_file_name):

            file = open(node_from_file_name, 'r')

            community = []

            for line in file:
                community.append(int(line.replace('\n', '')))

            nodes_to = shortest_path[1]

            d1_community_file = None
            d2_community_file = None

            for node_to in nodes_to:
                if node_from < node_to:
                    len_path = len(nodes_to[node_to])

                    if len_path == 2:
                        if node_to in community:
                            if d1_community_file is None:
                                d1_community_file = open(community_new_folder + '1\\' + node_from_str + '.txt', 'a')
                            d1_community_file.write(str(node_to) + '\n')
                    elif len_path == 3:
                        if node_to in community:
                            if d2_community_file is None:
                                d2_community_file = open(community_new_folder + '2\\' + node_from_str + '.txt', 'a')
                            d2_community_file.write(str(node_to) + '\n')

            if d1_community_file is not None:
                d1_community_file.close()
            if d2_community_file is not None:
                d2_community_file.close()


def update_mongo_community(source_folder):
    m = Mongo()

    def process_d(source_folder, community_update_many_ptr):

        list_to_mongo = []

        count = 1

        for file in os.listdir(source_folder):
            file_community = open(source_folder + file, 'r')
            node1 = int(file.split('.')[0])

            for line in file_community:
                community = int(line.replace('\n', ''))
                list_to_mongo.append(UpdateOne({'first': node1, 'second': community}, {'$set': {'is_inner': 1}}))

                if len(list_to_mongo) == 100000:
                    community_update_many_ptr(list_to_mongo)
                    list_to_mongo.clear()

            file_community.close()

            if count % 1000 == 0:
                print(count)
            count = count + 1

        if len(list_to_mongo) > 0:
            community_update_many_ptr(list_to_mongo)

    process_d(source_folder + '1\\', m.d1_bulk_write_update_many)
    process_d(source_folder + '2\\', m.d2_bulk_write_update_many)


def get_node_name_for_lattice(i, j):
    return str(i) + '-' + str(j)


def create_distribution(g):
    distribution = []
    for node in g.nodes:
        distribution = distribution + [node] * len(g.edges(node))

    return distribution


def randomize_in(g, p, i, lattice_size, z, distribution=None):
    i_nodes = [node for node in g.nodes if node.split('-')[0] == str(i)]

    if distribution is not None:
        i_distribution = [node for node in distribution if node.split('-')[0] == str(i)]

    for z in range(int(p * lattice_size * (z / 2) / 100)):
        neighbours = []

        while len(neighbours) == 0:
            f = random.choice(i_nodes)
            '''while True:
                f = get_node_name_for_lattice(i, random.randint(0, lattice_size - 1))
                if f in list(g.nodes):
                    break'''

            # neighbours = []
            # for n in list(g.neighbors(f)):
                # if str(n).split('-')[0] == str(i):
                    # neighbours.append(n)
            neighbours = [node for node in g.neighbors(f) if node.split('-')[0] == str(i)]

        # fn = neighbours[random.randint(0, len(neighbours) - 1)]
        fn = random.choice(neighbours)
        t = f

        if distribution is None:
            while g.has_edge(t, f) or t == f:
                t = get_node_name_for_lattice(i, random.randint(0, lattice_size - 1))

            g.remove_edge(f, fn)
            g.add_edge(f, t)
        else:
            distribution_copy = copy(i_distribution)

            t_i = i

            while (g.has_edge(t, f) or i != t_i or t == f) and len(distribution_copy) > 0:
                distribution_copy.remove(t)
                if len(distribution_copy) > 0:
                    t = random.choice(distribution_copy)
                    t_i = int(t[0:t.index('-')])

            if (not g.has_edge(t, f)) and i == t_i and t != f:
                g.remove_edge(f, fn)
                g.add_edge(f, t)

                distribution.remove(fn)
                i_distribution.remove(fn)

                distribution.append(t)
                i_distribution.append(t)
            else:
                print('else', t, distribution_copy)


def randomize_out(g, p, n, z, distribution=None):
    for _ in range(int(p * n * (z / 2) / 100)):
        neighbours = []

        while len(neighbours) == 0:
            f = list(g.nodes)[random.randint(0, len(g.nodes) - 1)]
            neighbours = list(g.neighbors(str(f)))

        fn = neighbours[random.randint(0, len(neighbours) - 1)]

        f_i = int(f[0:f.index('-')])

        while True:
            t = list(g.nodes)[random.randint(0, len(g.nodes) - 1)]
            if len(g.edges(t)) > 0:
                break

        t_i = int(t[0:t.index('-')])

        if distribution is None:
            while g.has_edge(t, f) or f_i == t_i:
                t = list(g.nodes)[random.randint(0, len(g.nodes) - 1)]
                t_i = int(t[0:t.index('-')])

            g.remove_edge(f, fn)
            g.add_edge(f, t)
        else:
            distribution_copy = copy(distribution)

            while (g.has_edge(t, f) or f_i == t_i) and len(distribution) > 0:
                distribution_copy.remove(t)
                if len(distribution_copy) > 0:
                    t = random.choice(distribution_copy)
                    t_i = int(t[0:t.index('-')])

            if (not g.has_edge(t, f)) and f_i != t_i:
                g.remove_edge(f, fn)
                g.add_edge(f, t)
                distribution.remove(fn)
                distribution.append(t)
            else:
                print('else', t, distribution_copy)


def randomize_both(g, p, distribution=None):
    for i in range(int(len(g.nodes) * p / 100)):
        neighbours = []

        while len(neighbours) == 0:
            f = list(g.nodes)[random.randint(0, len(g.nodes) - 1)]
            neighbours = list(g.neighbors(str(f)))
        f_len = len(g.edges(f))
        # fn = neighbours[random.randint(0, len(neighbours) - 1)]
        fn = random.choice(neighbours)
        fn_len = len(g.edges(fn))
        g.remove_edge(f, fn)

        if distribution is None:
            nodes_list = list(g.nodes)
        else:
            distribution.remove(fn)
            nodes_list = copy(distribution)

        t = f

        while g.has_edge(f, t) or f == t:
            t = random.choice(nodes_list)

        if distribution is not None:
            distribution.append(t)
        t_len = len(g.edges(t))
        # print(f_len, fn_len, t_len)
        g.add_edge(f, t)


def create_overlap(g, l, n):
    nodes_removed = []

    for _ in range(int(l * n / 100)):
        f = list(g.nodes)[random.randint(0, len(g.nodes) - 1)]
        neighbours = list(g.neighbors(str(f)))

        t = list(g.nodes)[random.randint(0, len(g.nodes) - 1)]
        while g.has_edge(t, f) or t == f or str(t)[0] == str(f)[0]:
            t = list(g.nodes)[random.randint(0, len(g.nodes) - 1)]

        g.remove_node(f)

        for nn in neighbours:
            g.add_edge(t, nn)

        nodes_removed.append((f, t))

    return nodes_removed


def create_lattice(n, z, m, p_in, p_out, l):
    if n % m != 0:
        print('not z|m')
        return
    if z % 2 != 0:
        print('not 2|z')
        return

    lattice_size = int(n / m)

    g = nx.Graph()

    ground_truth = []

    for i in range(m):
        lattice_ground_truth = []

        for j in range(lattice_size):
            current_node = get_node_name_for_lattice(i, j)
            lattice_ground_truth.append(current_node)

            for k in range(1, int(z / 2) + 1):
                g.add_edge(current_node, get_node_name_for_lattice(i, (j + k) % lattice_size))

        ground_truth.append(lattice_ground_truth)

        for _ in range(int(p_in * lattice_size * (z / 2) / 100)):
            randomize_in(g, p_in, i, lattice_size, z)

    if m > 1:
        randomize_out(g, p_out, n, z)

        if l > 0:
            nodes_removed = create_overlap(g, l, n)

            for node_removed in nodes_removed:
                for community in ground_truth:
                    if node_removed[0] in community:
                        community.remove(node_removed[0])
                        if node_removed[1] not in community:
                            community.append(node_removed[1])

    attributes = {'n': n, 'z': z, 'm': m}
    g.graph.update(attributes)

    return g, ground_truth


def randomize_lattice(g, p, randomize_type, l, distribution=None):
    n = g.graph['n']
    z = g.graph['z']
    m = g.graph['m']

    lattice_size = int(n / m)

    if randomize_type == 0:
        for i in range(m):
            randomize_in(g, p, i, lattice_size, z, distribution)
    elif randomize_type == 1:
        if m > 1:
            randomize_out(g, p, n, z, distribution)

            create_overlap(g, l, n)
    else: # elif randomize_type == 2
        randomize_both(g, p, distribution)


def print_log(s):
    if PRINT_LOG:
        # print(' ' * indent + s)
        print(s)


def get_similarity(communities, ground_truth, nodes_number):
    delta = 0
    for gt in ground_truth:
        gt_set = set(gt)

        max_index = -1
        max_intersection = None
        max_intersection_len = 0

        for i in range(len(communities)):
            current_intersection = gt_set.intersection(communities[i])
            current_intersection_len = len(current_intersection)
            if len(current_intersection) > max_intersection_len:
                max_index = i
                max_intersection = current_intersection
                max_intersection_len = current_intersection_len

        if max_index == -1:
            t = {}
        else:
            t = list((gt_set | set(communities[max_index])) - max_intersection)

        delta += len(t)

    return 1 - (delta / nodes_number)


def print_communities(communities):
    for c in communities:
        print(c)


def draw(g):
    nx.draw(g, with_labels=True)

    plt.draw()
    plt.show()


def louvain_community(g):
    communities = community_louvain.best_partition(g)
    communities_dict = {}
    communities_list = []

    for node in communities:
        node_community = communities[node]
        if node_community in communities_dict:
            communities_dict[node_community].append(node)
        else:
            communities_dict[node_community] = [node]

    for communities_dict_community in communities_dict:
        communities_list.append(communities_dict[communities_dict_community])

    return communities_list


def ground_truth_to_index(ground_truth, g):
    t = []
    nodes_list = list(g.nodes)

    for community in ground_truth:
        c = []

        for node in community:
            c.append(nodes_list.index(node))

        t.append(c)
    return t


def convert_iter_to_list(community_iter):
    for iter_slice in itertools.islice(community_iter, 1):
        communities = iter_slice

    return convert_adj_communities_sets_to_list(communities)


def convert_adj_communities_sets_to_list(community_set):
    community_list = []

    for community in community_set:
        community_list.append(list(community))

    return community_list


def get_adj_matrix(g):
    return np.array(nx.adjacency_matrix(g).todense())


def plot_histogram(g, iteration, n, z, m, l, title, t):
    plt.clf()
    degrees = [g.degree(n) for n in g.nodes()]
    plt.hist(degrees)
    plt.title(title + ' n=' + str(n) + ' z=' + str(z) + ' m=' + str(m) + ' l=' + str(l) + ' Iteration=' + str(iteration), fontsize=18)
    plt.xticks(np.arange(min(degrees), max(degrees) + 1, 1))
    plt.savefig('Algorithm Compare\\Histogram\\' + title + '_n=' + str(n) + ' z=' + str(z) + ' m=' + str(m) + ' l=' + str(l) + ' t=' + str(t) + ' Iteration=' + str(iteration) + '.png')


def plot_similarity(similarity_list, n, z, m, l, file_name, title):
    x_ticks = list(range(0, 21, 1))

    plt.clf()
    plt.xticks(x_ticks)
    plt.plot(x_ticks, similarity_list)
    plt.title(title + ' n=' + str(n) + ' z=' + str(z) + ' m=' + str(m) + ' l=' + str(l), fontsize=18)
    plt.savefig('Algorithm Compare\\Separate\\' + file_name + '_n=' + str(n) + ' z=' + str(z) + ' m=' + str(m) + ' l=' + str(l) + '.png')


def remove_bad_values(arr, min_value=-0.5):
    for i in range(len(arr)):
        if arr[i] <= min_value:
            return arr[0: i]
    return arr


# def plot_all_similarities(similarity_list_commonality1, similarity_list_commonality2, similarity_list_commonality3, similarity_list_louvain_method, similarity_list_hierarchical_clustering, similarity_list_bron_kerbosch, similarity_list_girvan_newman, similarity_list_clique_percolation, n, z, m, l, file_name, title, type):
# def plot_all_similarities(similarity_list_commonality1, similarity_list_commonality2, similarity_list_commonality3, similarity_list_louvain_method, similarity_list_hierarchical_clustering, similarity_list_bron_kerbosch, similarity_list_clique_percolation, n, z, m, l, file_name, title, type):
# def plot_all_similarities(similarity_list_commonality1, similarity_list_commonality2, similarity_list_commonality3, similarity_list_louvain_method, similarity_list_hierarchical_clustering, similarity_list_clique_percolation, n, z, m, l, file_name, title, type):
# def plot_all_similarities(similarity_list_commonality1, similarity_list_commonality2, similarity_list_commonality3, similarity_list_louvain_method, similarity_list_clique_percolation, n, z, m, l, file_name, title, type):
def plot_all_similarities(similarity_list_commonality1, similarity_list_commonality2, similarity_list_commonality3, similarity_list_louvain_method, n, z, m, l, file_name, title, type):
    x_ticks = list(range(0, 21, 1))
    plt.clf()
    plot1, = plt.plot(x_ticks, similarity_list_commonality1, 'r', label='f1')
    plot2, = plt.plot(x_ticks, similarity_list_commonality2, 'g', label='f2')
    plot3, = plt.plot(x_ticks, similarity_list_commonality3, 'hotpink', label='f3')
    plot4, = plt.plot(x_ticks, similarity_list_louvain_method, 'b', label='Louvain')
    # plot5, = plt.plot(x_ticks, similarity_list_hierarchical_clustering, 'y', label='Hierarchical Clustering')
    # plot6, = plt.plot(x_ticks, similarity_list_bron_kerbosch, 'm', label='Bron Kerbosch')
    # plot7, = plt.plot(x_ticks, similarity_list_girvan_newman, 'tab:orange', label='Girvan Newman')
    # plot8, = plt.plot(x_ticks, similarity_list_clique_percolation, 'c', label='Clique Percolation')
    plt.title(title + ' n=' + str(n) + ' z=' + str(z) + ' m=' + str(m) + ' l=' + str(l), fontsize=18)
    plt.xticks(x_ticks)
    # plt.legend(handles=[plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8])
    plt.legend(handles=[plot1, plot2, plot3, plot4])
    plt.savefig('Algorithm Compare\\' + file_name + '_n=' + str(n) + ' z=' + str(z) + ' m=' + str(m) + ' l=' + str(l) + '_' + type + '.png')

    plots = []

    plt.clf()

    similarity_list_commonality1 = remove_bad_values(similarity_list_commonality1)
    similarity_list_commonality2 = remove_bad_values(similarity_list_commonality2)
    similarity_list_commonality3 = remove_bad_values(similarity_list_commonality3)
    similarity_list_louvain_method = remove_bad_values(similarity_list_louvain_method)
    # similarity_list_hierarchical_clustering = remove_bad_values(similarity_list_hierarchical_clustering)
    # similarity_list_bron_kerbosch = remove_bad_values(similarity_list_bron_kerbosch)
    # similarity_list_girvan_newman = remove_bad_values(similarity_list_girvan_newman)
    # similarity_list_clique_percolation = remove_bad_values(similarity_list_clique_percolation)

    if len(similarity_list_commonality1) > 0:
        plot, = plt.plot(x_ticks[0: len(similarity_list_commonality1)], similarity_list_commonality1, 'r', label='f1')
        plots.append(plot)

    if len(similarity_list_commonality2) > 0:
        plot, = plt.plot(x_ticks[0: len(similarity_list_commonality2)], similarity_list_commonality2, 'g', label='f2')
        plots.append(plot)

    if len(similarity_list_commonality3) > 0:
        plot, = plt.plot(x_ticks[0: len(similarity_list_commonality3)], similarity_list_commonality3, 'hotpink', label='f3')
        plots.append(plot)

    if len(similarity_list_louvain_method) > 0:
        plot, = plt.plot(x_ticks[0: len(similarity_list_louvain_method)], similarity_list_louvain_method, 'b', label='Louvain')
        plots.append(plot)

    '''if len(similarity_list_hierarchical_clustering) > 0:
        plot, = plt.plot(x_ticks[0: len(similarity_list_hierarchical_clustering)], similarity_list_hierarchical_clustering, 'y', label='Hierarchical Clustering')
        plots.append(plot)'''

    '''if len(similarity_list_bron_kerbosch) > 0:
        plot, = plt.plot(x_ticks[0: len(similarity_list_bron_kerbosch)], similarity_list_bron_kerbosch, 'm', label='Bron Kerbosch')
        plots.append(plot)'''

    '''if len(similarity_list_girvan_newman) > 0:
        plot, = plt.plot(x_ticks[0: len(similarity_list_girvan_newman)], similarity_list_girvan_newman, 'tab:orange', label='Girvan Newman')
        plots.append(plot)'''

    '''if len(similarity_list_clique_percolation) > 0:
        plot, = plt.plot(x_ticks[0: len(similarity_list_clique_percolation)], similarity_list_clique_percolation, 'c', label='Clique Percolation')
        plots.append(plot)'''

    plt.title(title + ' n=' + str(n) + ' z=' + str(z) + ' m=' + str(m) + ' l=' + str(l), fontsize=18)
    plt.xticks(x_ticks)
    plt.legend(handles=plots)
    plt.savefig('Algorithm Compare\\' + file_name + '_n=' + str(n) + ' z=' + str(z) + ' m=' + str(m) + ' l=' + str(l) + '_' + type + '_2.png')


def algorithm_compare(n, z, m, p, l, iterations):
    times = 10

    from Community import Community

    '''similarity_list_in_commonality1_GTvsC = [0] * (iterations + 1)
    similarity_list_in_commonality2_GTvsC = [0] * (iterations + 1)
    similarity_list_in_commonality3_GTvsC = [0] * (iterations + 1)

    similarity_list_in_commonality1_CvsGT = [0] * (iterations + 1)
    similarity_list_in_commonality2_CvsGT = [0] * (iterations + 1)
    similarity_list_in_commonality3_CvsGT = [0] * (iterations + 1)

    similarity_list_in_louvain_method = [0] * (iterations + 1)
    similarity_list_in_hierarchical_clustering = [0] * (iterations + 1)
    similarity_list_in_bron_kerbosch = [0] * (iterations + 1)

    similarity_list_out_commonality1_GTvsC = [0] * (iterations + 1)
    similarity_list_out_commonality2_GTvsC = [0] * (iterations + 1)
    similarity_list_out_commonality3_GTvsC = [0] * (iterations + 1)

    similarity_list_out_commonality1_CvsGT = [0] * (iterations + 1)
    similarity_list_out_commonality2_CvsGT = [0] * (iterations + 1)
    similarity_list_out_commonality3_CvsGT = [0] * (iterations + 1)

    similarity_list_out_louvain_method = [0] * (iterations + 1)
    similarity_list_out_hierarchical_clustering = [0] * (iterations + 1)
    similarity_list_out_bron_kerbosch = [0] * (iterations + 1)'''

    similarity_list_both_commonality1_GTvsC = [0] * (iterations + 1)
    similarity_list_both_commonality2_GTvsC = [0] * (iterations + 1)
    similarity_list_both_commonality3_GTvsC = [0] * (iterations + 1)

    similarity_list_both_commonality1_CvsGT = [0] * (iterations + 1)
    similarity_list_both_commonality2_CvsGT = [0] * (iterations + 1)
    similarity_list_both_commonality3_CvsGT = [0] * (iterations + 1)

    similarity_list_both_louvain_method = [0] * (iterations + 1)
    # similarity_list_both_hierarchical_clustering = [0] * (iterations + 1)
    # similarity_list_both_bron_kerbosch = [0] * (iterations + 1)
    # similarity_list_both_girvan_newman = [0] * (iterations + 1)
    # similarity_list_both_clique_percolation = [0] * (iterations + 1)

    for t in range(times):
        print('t =', t)
        # g_in, ground_truth_in = create_lattice(n, z, m, 0, 0, l)
        # g_out, ground_truth_out = create_lattice(n, z, m, 0, 0, l)
        g_both, ground_truth_both = create_lattice(n, z, m, 0, 0, l)

        # g_in_distribution = create_distribution(g_in)
        # g_out_distribution = create_distribution(g_out)
        g_both_distribution = create_distribution(g_both)

        # ground_truth_in_index = ground_truth_to_index(ground_truth_in, g_in)
        # ground_truth_out_index = ground_truth_to_index(ground_truth_out, g_out)
        # ground_truth_both_index = ground_truth_to_index(ground_truth_both, g_both)

        # community_in1 = Community(g_in, 1)
        # community_out1 = Community(g_out, 1)
        community_both1 = Community(g_both, 1)

        # community_in2 = Community(g_in, 2)
        # community_out2 = Community(g_out, 2)
        community_both2 = Community(g_both, 2)

        # community_in3 = Community(g_in, 3, z)
        # community_out3 = Community(g_out, 3, z)
        community_both3 = Community(g_both, 3, z)

        # nodes_number_in = len(g_in.nodes)
        # nodes_number_out = len(g_out.nodes)
        nodes_number_both = len(g_both.nodes)

        # adj_matrix_in = get_adj_matrix(g_in)
        # adj_matrix_out = get_adj_matrix(g_out)
        # adj_matrix_both = get_adj_matrix(g_both)

        '''communities_in_commonality1 = community_in1.get_communities()
        communities_in_commonality2 = community_in2.get_communities()
        communities_in_commonality3 = community_in3.get_communities()
        communities_in_louvain_method = louvain_community(g_in)
        communities_in_hierarchical_clustering = convert_adj_communities_sets_to_list(hierarchical_clustering(adj_matrix_in))
        print(6, datetime.now())
        communities_in_bron_kerbosch = convert_adj_communities_sets_to_list(bron_kerbosch(adj_matrix_in))
        print(7, datetime.now())

        communities_out_commonality1 = community_out1.get_communities()
        communities_out_commonality2 = community_out2.get_communities()
        communities_out_commonality3 = community_out3.get_communities()
        communities_out_louvain_method = louvain_community(g_out)
        communities_out_hierarchical_clustering = convert_adj_communities_sets_to_list(hierarchical_clustering(adj_matrix_out))
        communities_out_bron_kerbosch = convert_adj_communities_sets_to_list(bron_kerbosch(adj_matrix_out))'''

        print('start\t\t\t\t\t', datetime.now())
        communities_both_commonality1 = community_both1.get_communities()
        print('community_both1\t\t\t', datetime.now())
        communities_both_commonality2 = community_both2.get_communities()
        print('community_both2\t\t\t', datetime.now())
        communities_both_commonality3 = community_both3.get_communities()
        print('community_both3\t\t\t', datetime.now())
        communities_both_louvain_method = louvain_community(g_both)
        print('louvain\t\t\t\t\t', datetime.now())
        # communities_both_hierarchical_clustering = convert_adj_communities_sets_to_list(hierarchical_clustering(adj_matrix_both))
        # print('hierarchical_clustering\t', datetime.now())
        # communities_both_bron_kerbosch = convert_adj_communities_sets_to_list(bron_kerbosch(adj_matrix_both))
        # print('bron_kerbosch\t\t\t', datetime.now())
        # communities_both_girvan_newman = convert_iter_to_list(nx.community.girvan_newman(g_both))
        # print('girvan_newman', datetime.now())
        # communities_both_clique_percolation = convert_adj_communities_sets_to_list(list(nx.community.k_clique_communities(g_both, 3)))
        # print('k_clique_communities\t', datetime.now())
        print('end', datetime.now())

        '''similarity_list_in_commonality1_GTvsC[0] += get_similarity(ground_truth_in, communities_in_commonality1, nodes_number_in) / times
        similarity_list_in_commonality2_GTvsC[0] += get_similarity(ground_truth_in, communities_in_commonality2, nodes_number_in) / times
        similarity_list_in_commonality3_GTvsC[0] += get_similarity(ground_truth_in, communities_in_commonality3, nodes_number_in) / times

        similarity_list_in_commonality1_CvsGT[0] += get_similarity(communities_in_commonality1, ground_truth_in, nodes_number_in) / times
        similarity_list_in_commonality2_CvsGT[0] += get_similarity(communities_in_commonality2, ground_truth_in, nodes_number_in) / times
        similarity_list_in_commonality3_CvsGT[0] += get_similarity(communities_in_commonality3, ground_truth_in, nodes_number_in) / times

        similarity_list_in_louvain_method[0] += get_similarity(ground_truth_in, communities_in_louvain_method, nodes_number_in) / times
        similarity_list_in_hierarchical_clustering[0] += get_similarity(ground_truth_in_index, communities_in_hierarchical_clustering, nodes_number_in) / times
        similarity_list_in_bron_kerbosch[0] += get_similarity(ground_truth_in_index, communities_in_bron_kerbosch, nodes_number_in) / times

        similarity_list_out_commonality1_GTvsC[0] += get_similarity(ground_truth_out, communities_out_commonality1, nodes_number_out) / times
        similarity_list_out_commonality2_GTvsC[0] += get_similarity(ground_truth_out, communities_out_commonality2, nodes_number_out) / times
        similarity_list_out_commonality3_GTvsC[0] += get_similarity(ground_truth_out, communities_out_commonality3, nodes_number_out) / times

        similarity_list_out_commonality1_CvsGT[0] += get_similarity(communities_out_commonality1, ground_truth_out, nodes_number_out) / times
        similarity_list_out_commonality2_CvsGT[0] += get_similarity(communities_out_commonality2, ground_truth_out, nodes_number_out) / times
        similarity_list_out_commonality3_CvsGT[0] += get_similarity(communities_out_commonality3, ground_truth_out, nodes_number_out) / times

        similarity_list_out_louvain_method[0] += get_similarity(ground_truth_out, communities_out_louvain_method, nodes_number_out) / times
        similarity_list_out_hierarchical_clustering[0] += get_similarity(ground_truth_out_index, communities_out_hierarchical_clustering, nodes_number_out) / times
        similarity_list_out_bron_kerbosch[0] += get_similarity(ground_truth_out_index, communities_out_bron_kerbosch, nodes_number_out) / times'''

        similarity_list_both_commonality1_GTvsC[0] += get_similarity(ground_truth_both, communities_both_commonality1, nodes_number_both) / times
        similarity_list_both_commonality2_GTvsC[0] += get_similarity(ground_truth_both, communities_both_commonality2, nodes_number_both) / times
        similarity_list_both_commonality3_GTvsC[0] += get_similarity(ground_truth_both, communities_both_commonality3, nodes_number_both) / times

        similarity_list_both_commonality1_CvsGT[0] += get_similarity(communities_both_commonality1, ground_truth_both, nodes_number_both) / times
        similarity_list_both_commonality2_CvsGT[0] += get_similarity(communities_both_commonality2, ground_truth_both, nodes_number_both) / times
        similarity_list_both_commonality3_CvsGT[0] += get_similarity(communities_both_commonality3, ground_truth_both, nodes_number_both) / times

        similarity_list_both_louvain_method[0] += get_similarity(ground_truth_both, communities_both_louvain_method, nodes_number_both) / times
        # similarity_list_both_hierarchical_clustering[0] += get_similarity(ground_truth_both_index, communities_both_hierarchical_clustering, nodes_number_both) / times
        # similarity_list_both_bron_kerbosch[0] += get_similarity(ground_truth_both_index, communities_both_bron_kerbosch, nodes_number_both) / times
        # similarity_list_both_girvan_newman[0] += get_similarity(ground_truth_both, communities_both_girvan_newman, nodes_number_both) / times
        # similarity_list_both_clique_percolation[0] += get_similarity(ground_truth_both, communities_both_clique_percolation, nodes_number_both) / times

        for i in range(1, iterations + 1):
            print('i =', i)

            # randomize_lattice(g_in, p, 0, 0, g_in_distribution)
            # randomize_lattice(g_out, p, 1, 0, g_out_distribution)

            # print('start Randomize\t\t\t\t\t', datetime.now())
            randomize_lattice(g_both, p / 2, 0, 0, g_both_distribution)
            # print('start Randomize2\t\t\t\t\t', datetime.now())
            randomize_lattice(g_both, p / 2, 1, 0, g_both_distribution)
            # print('end Randomize\t\t\t\t\t', datetime.now())

            # if i in (5, 10, 15, 20):
                # plot_histogram(g_in, i, n, z, m, l, 'in', t)
                # plot_histogram(g_out, i, n, z, m, l, 'out', t)
                # plot_histogram(g_both, i, n, z, m, l, 'both', t)

            # adj_matrix_in = get_adj_matrix(g_in)
            # adj_matrix_out = get_adj_matrix(g_out)
            # adj_matrix_both = get_adj_matrix(g_both)

            # communities_in_commonality1 = community_in1.get_communities()
            # communities_in_commonality2 = community_in2.get_communities()
            # communities_in_commonality3 = community_in3.get_communities()
            # communities_in_louvain_method = louvain_community(g_in)
            # communities_in_hierarchical_clustering = convert_adj_communities_sets_to_list(hierarchical_clustering(adj_matrix_in))
            # communities_in_bron_kerbosch = convert_adj_communities_sets_to_list(bron_kerbosch(adj_matrix_in))


            '''communities_out_commonality1 = community_out1.get_communities()
            communities_out_commonality2 = community_out2.get_communities()
            communities_out_commonality3 = community_out3.get_communities()
            communities_out_louvain_method = louvain_community(g_out)
            communities_out_hierarchical_clustering = convert_adj_communities_sets_to_list(hierarchical_clustering(adj_matrix_out))
            communities_out_bron_kerbosch = convert_adj_communities_sets_to_list(bron_kerbosch(adj_matrix_out))'''

            print('start\t\t\t\t\t', datetime.now())
            communities_both_commonality1 = community_both1.get_communities()
            print('community_both1\t\t\t', datetime.now())
            communities_both_commonality2 = community_both2.get_communities()
            print('community_both2\t\t\t', datetime.now())
            communities_both_commonality3 = community_both3.get_communities()
            print('community_both3\t\t\t', datetime.now())
            communities_both_louvain_method = louvain_community(g_both)
            print('louvain\t\t\t\t\t', datetime.now())
            # communities_both_hierarchical_clustering = convert_adj_communities_sets_to_list(hierarchical_clustering(adj_matrix_both))
            # communities_both_bron_kerbosch = convert_adj_communities_sets_to_list(bron_kerbosch(adj_matrix_both))
            # communities_both_girvan_newman = convert_iter_to_list(nx.community.girvan_newman(g_both))
            # communities_both_clique_percolation = convert_adj_communities_sets_to_list(list(nx.community.k_clique_communities(g_both, 3)))
            # print('k_clique_communities\t', datetime.now())
            print('end', datetime.now())


            '''similarity_list_in_commonality1_GTvsC[i] += get_similarity(ground_truth_in, communities_in_commonality1, nodes_number_in) / times
            similarity_list_in_commonality2_GTvsC[i] += get_similarity(ground_truth_in, communities_in_commonality2, nodes_number_in) / times
            similarity_list_in_commonality3_GTvsC[i] += get_similarity(ground_truth_in, communities_in_commonality3, nodes_number_in) / times

            similarity_list_in_commonality1_CvsGT[i] += get_similarity(communities_in_commonality1, ground_truth_in, nodes_number_in) / times
            similarity_list_in_commonality2_CvsGT[i] += get_similarity(communities_in_commonality2, ground_truth_in, nodes_number_in) / times
            similarity_list_in_commonality3_CvsGT[i] += get_similarity(communities_in_commonality3, ground_truth_in, nodes_number_in) / times

            similarity_list_in_louvain_method[i] += get_similarity(ground_truth_in, communities_in_louvain_method, nodes_number_in) / times
            similarity_list_in_hierarchical_clustering[i] += get_similarity(ground_truth_in_index, communities_in_hierarchical_clustering, nodes_number_in) / times
            similarity_list_in_bron_kerbosch[i] += get_similarity(ground_truth_in_index, communities_in_bron_kerbosch, nodes_number_in) / times

            similarity_list_out_commonality1_GTvsC[i] += get_similarity(ground_truth_out, communities_out_commonality1, nodes_number_out) / times
            similarity_list_out_commonality2_GTvsC[i] += get_similarity(ground_truth_out, communities_out_commonality2, nodes_number_out) / times
            similarity_list_out_commonality3_GTvsC[i] += get_similarity(ground_truth_out, communities_out_commonality3, nodes_number_out) / times

            similarity_list_out_commonality1_CvsGT[i] += get_similarity(communities_out_commonality1, ground_truth_out, nodes_number_out) / times
            similarity_list_out_commonality2_CvsGT[i] += get_similarity(communities_out_commonality2, ground_truth_out, nodes_number_out) / times
            similarity_list_out_commonality3_CvsGT[i] += get_similarity(communities_out_commonality3, ground_truth_out, nodes_number_out) / times

            similarity_list_out_louvain_method[i] += get_similarity(ground_truth_out, communities_out_louvain_method, nodes_number_out) / times
            similarity_list_out_hierarchical_clustering[i] += get_similarity(ground_truth_out_index, communities_out_hierarchical_clustering, nodes_number_out) / times
            similarity_list_out_bron_kerbosch[i] += get_similarity(ground_truth_out_index, communities_out_bron_kerbosch, nodes_number_out) / times'''

            similarity_list_both_commonality1_GTvsC[i] += get_similarity(ground_truth_both, communities_both_commonality1, nodes_number_both) / times
            similarity_list_both_commonality2_GTvsC[i] += get_similarity(ground_truth_both, communities_both_commonality2, nodes_number_both) / times
            similarity_list_both_commonality3_GTvsC[i] += get_similarity(ground_truth_both, communities_both_commonality3, nodes_number_both) / times

            similarity_list_both_commonality1_CvsGT[i] += get_similarity(communities_both_commonality1, ground_truth_both, nodes_number_both) / times
            similarity_list_both_commonality2_CvsGT[i] += get_similarity(communities_both_commonality2, ground_truth_both, nodes_number_both) / times
            similarity_list_both_commonality3_CvsGT[i] += get_similarity(communities_both_commonality3, ground_truth_both, nodes_number_both) / times

            similarity_list_both_louvain_method[i] += get_similarity(ground_truth_both, communities_both_louvain_method, nodes_number_both) / times
            # similarity_list_both_hierarchical_clustering[i] += get_similarity(ground_truth_both_index, communities_both_hierarchical_clustering, nodes_number_both) / times
            # similarity_list_both_bron_kerbosch[i] += get_similarity(ground_truth_both_index, communities_both_bron_kerbosch, nodes_number_both) / times
            # similarity_list_both_girvan_newman[i] += get_similarity(ground_truth_both, communities_both_girvan_newman, nodes_number_both) / times
            # similarity_list_both_clique_percolation[i] += get_similarity(ground_truth_both, communities_both_clique_percolation, nodes_number_both) / times

    '''plot_similarity(similarity_list_in_commonality1_GTvsC, n, z, m, l, 'In_Commonality_Alg1_GTvsC', 'In_Commonality_Alg1')
    plot_similarity(similarity_list_in_commonality2_GTvsC, n, z, m, l, 'In_Commonality_Alg2_GTvsC', 'In_Commonality_Alg2')
    plot_similarity(similarity_list_in_commonality3_GTvsC, n, z, m, l, 'In_Commonality_Alg3_GTvsC', 'In_Commonality_Alg3')

    plot_similarity(similarity_list_in_commonality1_CvsGT, n, z, m, l, 'In_Commonality_Alg1_CvsGT', 'In_Commonality_Alg1')
    plot_similarity(similarity_list_in_commonality2_CvsGT, n, z, m, l, 'In_Commonality_Alg2_CvsGT', 'In_Commonality_Alg2')
    plot_similarity(similarity_list_in_commonality3_CvsGT, n, z, m, l, 'In_Commonality_Alg3_CvsGT', 'In_Commonality_Alg3')

    plot_similarity(similarity_list_in_louvain_method, n, z, m, l, 'In_Louvain_Method', 'In_Louvain_Method')
    plot_similarity(similarity_list_in_hierarchical_clustering, n, z, m, l, 'In_Hierarchical_Clustering', 'In_Hierarchical_Clustering')
    plot_similarity(similarity_list_in_bron_kerbosch, n, z, m, l, 'In_Bron_Kerbosch', 'In_Bron_Kerbosch')'''

    # plot_all_similarities(similarity_list_in_commonality1_GTvsC, similarity_list_in_commonality2_GTvsC, similarity_list_in_commonality3_GTvsC, similarity_list_in_louvain_method, similarity_list_in_hierarchical_clustering, similarity_list_in_bron_kerbosch, n, z, m, l, 'In_All_GTvsC', 'In_All')
    # plot_all_similarities(similarity_list_in_commonality1_CvsGT, similarity_list_in_commonality2_CvsGT, similarity_list_in_commonality3_CvsGT, similarity_list_in_louvain_method, similarity_list_in_hierarchical_clustering, similarity_list_in_bron_kerbosch, n, z, m, l, 'In_All_CvsGT', 'In_All')

    '''plot_similarity(similarity_list_out_commonality1_GTvsC, n, z, m, l, 'Out_Commonality_Alg1_GTvsC', 'Out_Commonality_Alg1')
    plot_similarity(similarity_list_out_commonality2_GTvsC, n, z, m, l, 'Out_Commonality_Alg2_GTvsC', 'Out_Commonality_Alg2')
    plot_similarity(similarity_list_out_commonality3_GTvsC, n, z, m, l, 'Out_Commonality_Alg3_GTvsC', 'Out_Commonality_Alg3')

    plot_similarity(similarity_list_out_commonality1_CvsGT, n, z, m, l, 'Out_Commonality_Alg1_CvsGT', 'Out_Commonality_Alg1')
    plot_similarity(similarity_list_out_commonality2_CvsGT, n, z, m, l, 'Out_Commonality_Alg2_CvsGT', 'Out_Commonality_Alg2')
    plot_similarity(similarity_list_out_commonality3_CvsGT, n, z, m, l, 'Out_Commonality_Alg3_CvsGT', 'Out_Commonality_Alg3')

    plot_similarity(similarity_list_out_louvain_method, n, z, m, l, 'Out_Louvain_Method', 'Out_Louvain_Method')
    plot_similarity(similarity_list_out_hierarchical_clustering, n, z, m, l, 'Out_Hierarchical_Clustering', 'Out_Hierarchical_Clustering')
    plot_similarity(similarity_list_out_bron_kerbosch, n, z, m, l, 'Out_Bron_Kerbosch', 'Out_Bron_Kerbosch')'''

    # plot_all_similarities(similarity_list_out_commonality1_GTvsC, similarity_list_out_commonality2_GTvsC, similarity_list_out_commonality3_GTvsC, similarity_list_out_louvain_method, similarity_list_out_hierarchical_clustering, similarity_list_out_bron_kerbosch, n, z, m, l, 'Out_All_GTvsC', 'Out_All')
    # plot_all_similarities(similarity_list_out_commonality1_CvsGT, similarity_list_out_commonality2_CvsGT, similarity_list_out_commonality3_CvsGT, similarity_list_out_louvain_method, similarity_list_out_hierarchical_clustering, similarity_list_out_bron_kerbosch, n, z, m, l, 'Out_All_CvsGT', 'Out_All')

    '''plot_similarity(similarity_list_both_commonality1_GTvsC, n, z, m, l, 'Both_Commonality_Alg1_GTvsC', 'Both_Commonality_Alg1')
    plot_similarity(similarity_list_both_commonality2_GTvsC, n, z, m, l, 'Both_Commonality_Alg2_GTvsC', 'Both_Commonality_Alg2')
    plot_similarity(similarity_list_both_commonality3_GTvsC, n, z, m, l, 'Both_Commonality_Alg3_GTvsC', 'Both_Commonality_Alg3')

    plot_similarity(similarity_list_both_commonality1_CvsGT, n, z, m, l, 'Both_Commonality_Alg1_CvsGT', 'Both_Commonality_Alg1')
    plot_similarity(similarity_list_both_commonality2_CvsGT, n, z, m, l, 'Both_Commonality_Alg2_CvsGT', 'Both_Commonality_Alg2')
    plot_similarity(similarity_list_both_commonality3_CvsGT, n, z, m, l, 'Both_Commonality_Alg3_CvsGT', 'Both_Commonality_Alg3')

    plot_similarity(similarity_list_both_louvain_method, n, z, m, l, 'Both_Louvain_Method', 'Both_Louvain_Method')
    plot_similarity(similarity_list_both_hierarchical_clustering, n, z, m, l, 'Both_Hierarchical_Clustering', 'Both_Hierarchical_Clustering')
    plot_similarity(similarity_list_both_bron_kerbosch, n, z, m, l, 'Both_Bron_Kerbosch', 'Both_Bron_Kerbosch')'''

    # plot_all_similarities(similarity_list_both_commonality1_GTvsC, similarity_list_both_commonality2_GTvsC, similarity_list_both_commonality3_GTvsC, similarity_list_both_louvain_method, similarity_list_both_hierarchical_clustering, similarity_list_both_bron_kerbosch, similarity_list_both_girvan_newman, similarity_list_both_clique_percolation, n, z, m, l, 'Both_All', 'Both_All', 'GTvsC')
    # plot_all_similarities(similarity_list_both_commonality1_CvsGT, similarity_list_both_commonality2_CvsGT, similarity_list_both_commonality3_CvsGT, similarity_list_both_louvain_method, similarity_list_both_hierarchical_clustering, similarity_list_both_bron_kerbosch, similarity_list_both_girvan_newman, similarity_list_both_clique_percolation, n, z, m, l, 'Both_All', 'Both_All', 'CvsGT')
    # plot_all_similarities(similarity_list_both_commonality1_GTvsC, similarity_list_both_commonality2_GTvsC, similarity_list_both_commonality3_GTvsC, similarity_list_both_louvain_method, similarity_list_both_hierarchical_clustering, similarity_list_both_bron_kerbosch, similarity_list_both_clique_percolation, n, z, m, l, 'Both_All', 'Both_All', 'GTvsC')
    # plot_all_similarities(similarity_list_both_commonality1_CvsGT, similarity_list_both_commonality2_CvsGT, similarity_list_both_commonality3_CvsGT, similarity_list_both_louvain_method, similarity_list_both_hierarchical_clustering, similarity_list_both_bron_kerbosch, similarity_list_both_clique_percolation, n, z, m, l, 'Both_All', 'Both_All', 'CvsGT')
    # plot_all_similarities(similarity_list_both_commonality1_GTvsC, similarity_list_both_commonality2_GTvsC, similarity_list_both_commonality3_GTvsC, similarity_list_both_louvain_method, similarity_list_both_hierarchical_clustering, similarity_list_both_clique_percolation, n, z, m, l, 'Both_All', 'Both_All', 'GTvsC')
    # plot_all_similarities(similarity_list_both_commonality1_CvsGT, similarity_list_both_commonality2_CvsGT, similarity_list_both_commonality3_CvsGT, similarity_list_both_louvain_method, similarity_list_both_hierarchical_clustering, similarity_list_both_clique_percolation, n, z, m, l, 'Both_All', 'Both_All', 'CvsGT')
    # plot_all_similarities(similarity_list_both_commonality1_GTvsC, similarity_list_both_commonality2_GTvsC, similarity_list_both_commonality3_GTvsC, similarity_list_both_louvain_method, similarity_list_both_clique_percolation, n, z, m, l, 'Both_All', 'Both_All', 'GTvsC')
    # plot_all_similarities(similarity_list_both_commonality1_CvsGT, similarity_list_both_commonality2_CvsGT, similarity_list_both_commonality3_CvsGT, similarity_list_both_louvain_method, similarity_list_both_clique_percolation, n, z, m, l, 'Both_All', 'Both_All', 'CvsGT')
    plot_all_similarities(similarity_list_both_commonality1_GTvsC, similarity_list_both_commonality2_GTvsC, similarity_list_both_commonality3_GTvsC, similarity_list_both_louvain_method, n, z, m, l, 'Both_All', 'Both_All', 'GTvsC')
    plot_all_similarities(similarity_list_both_commonality1_CvsGT, similarity_list_both_commonality2_CvsGT, similarity_list_both_commonality3_CvsGT, similarity_list_both_louvain_method, n, z, m, l, 'Both_All', 'Both_All', 'CvsGT')


def plot_algorithm_sensitivity(x_ticks, list, title, save_folder, file_name):
    plt.clf()
    plt.xticks(x_ticks)
    plt.plot(x_ticks, list)
    plt.title(title, fontsize=18)
    plt.savefig(save_folder + file_name + '.png')


def plot_all_algorithms_sensitivity(alg1_list, alg2_list, alg3_list, title, save_folder, file_name):
    x_ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    plt.clf()
    plot1, = plt.plot(x_ticks, alg1_list, 'r', label='f1')
    plot2, = plt.plot(x_ticks, alg2_list, 'g', label='f2')
    plot3, = plt.plot(x_ticks, alg3_list, 'b', label='f3')
    plt.title(title, fontsize=18)
    plt.xticks(x_ticks)
    plt.legend(handles=[plot1, plot2, plot3])
    plt.savefig(save_folder + file_name + '.png')


def remove_outliers(l):
    an_array = np.array(l)
    mean = np.mean(an_array)
    standard_deviation = np.std(an_array)
    distance_from_mean = abs(an_array - mean)
    max_deviations = 1.5
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    return an_array[not_outlier]


def similarity(data_name):
    community_file_destination = 'c:\\University\\Thesis\\Network-Community-Detection\\Data\\' + data_name + 'Cmty\\1\\'
    # c = Commonality.load_from_file('C:\\University\\Thesis\\Network-Community-Detection\\Data\\' + data_name + '.txt', '\t', 'C:\\University\\Thesis\\Network-Community-Detection\\Executions\\')
    g_karate_club = nx.karate_club_graph()
    g_karate_club.name = data_name
    c = Commonality.load_from_graph(g_karate_club, 'C:\\University\\Thesis\\Network-Community-Detection\\Executions\\')
    g = c.get_graph()

    different_thresholds = 11

    alg1 = [0, None]
    alg2 = [0, None]
    alg3 = [0, None]

    commonality = {}

    z = 2 * len(g.edges) / len(g.nodes)

    alg1_values = []
    alg2_values = []
    alg3_values = []

    for edge in g.edges:
        if edge[0] < edge[1]:
            node1 = edge[0]
            node2 = edge[1]
        else:
            node1 = edge[1]
            node2 = edge[0]

        path = community_file_destination + str(node1) + '.txt'

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

        str_node2 = str(node2)

        if str_node2 == first_node or '\n' + str_node2 + '\n' in community or str_node2 == last_node:
            true_neighbors = True
        else:
            true_neighbors = False

        numerator, denominator = calculate_commonality(g, node1, node2)

        alg1_values.append(numerator)
        alg2_values.append(numerator / denominator)
        alg3_values.append((numerator * numerator) / denominator)

        commonality[(node1, node2)] = [numerator, denominator, true_neighbors]

    alg1_values = remove_outliers(alg1_values)
    alg2_values = remove_outliers(alg2_values)
    alg3_values = remove_outliers(alg3_values)

    alg1[1] = max(alg1_values)
    alg2[1] = max(alg2_values)
    alg3[1] = max(alg3_values)

    alg1.append(np.linspace(alg1[0], alg1[1], different_thresholds))
    alg2.append(np.linspace(alg2[0], alg2[1], different_thresholds))
    alg3.append(np.linspace(alg3[0], alg3[1], different_thresholds))

    positive_arr = [0] * different_thresholds

    positive_mat = [deepcopy(positive_arr), deepcopy(positive_arr), deepcopy(positive_arr), deepcopy(positive_arr)]

    alg1.append(deepcopy(positive_mat))
    alg2.append(deepcopy(positive_mat))
    alg3.append(deepcopy(positive_mat))

    true_positive = 0
    false_positive = 1
    false_negative = 2
    true_negative = 3
    fp_divided_by_tp = 4
    fpr = 5
    fdr = 6
    tpr = 7

    for key in commonality.keys():
        value = commonality[key]

        numerator, denominator, true_neighbors = value[0], value[1], value[2]

        for i in range(different_thresholds):
            if numerator >= alg1[2][i]:
                if true_neighbors is True:
                    alg1[3][true_positive][i] += 1
                else:
                    alg1[3][false_positive][i] += 1
            else:
                if true_neighbors is True:
                    alg1[3][false_negative][i] += 1
                else:
                    alg1[3][true_negative][i] += 1

            if numerator / denominator >= alg2[2][i]:
                if true_neighbors is True:
                    alg2[3][true_positive][i] += 1
                else:
                    alg2[3][false_positive][i] += 1
            else:
                if true_neighbors is True:
                    alg2[3][false_negative][i] += 1
                else:
                    alg2[3][true_negative][i] += 1

            if (numerator * numerator) / (z * denominator) >= alg3[2][i]:
                if true_neighbors is True:
                    alg3[3][true_positive][i] += 1
                else:
                    alg3[3][false_positive][i] += 1
            else:
                if true_neighbors is True:
                    alg3[3][false_negative][i] += 1
                else:
                    alg3[3][true_negative][i] += 1

    alg1[3].append([])
    alg1[3].append([])
    alg1[3].append([])
    alg1[3].append([])

    alg2[3].append([])
    alg2[3].append([])
    alg2[3].append([])
    alg2[3].append([])

    alg3[3].append([])
    alg3[3].append([])
    alg3[3].append([])
    alg3[3].append([])

    for i in range(different_thresholds):
        if alg1[3][true_positive][i] == 0:
            alg1[3][fp_divided_by_tp].append(0)
        else:
            alg1[3][fp_divided_by_tp].append(alg1[3][false_positive][i] / alg1[3][true_positive][i])

        if alg1[3][true_negative][i] == 0 and alg1[3][false_positive][i] == 0:
            alg1[3][fpr].append(0)
        else:
            alg1[3][fpr].append(alg1[3][false_positive][i] / (alg1[3][false_positive][i] + alg1[3][true_negative][i]))

        if alg2[3][true_positive][i] == 0:
            alg2[3][fp_divided_by_tp].append(0)
        else:
            alg2[3][fp_divided_by_tp].append(alg2[3][false_positive][i] / alg2[3][true_positive][i])

        if alg2[3][true_negative][i] == 0 and alg2[3][false_positive][i] == 0:
            alg2[3][fpr].append(0)
        else:
            alg2[3][fpr].append(alg2[3][false_positive][i] / (alg2[3][false_positive][i] + alg2[3][true_negative][i]))

        if alg3[3][true_positive][i] == 0:
            alg3[3][fp_divided_by_tp].append(0)
        else:
            alg3[3][fp_divided_by_tp].append(alg3[3][false_positive][i] / alg3[3][true_positive][i])

        if alg3[3][true_negative][i] == 0 and alg3[3][false_positive][i] == 0:
            alg3[3][fpr].append(0)
        else:
            alg3[3][fpr].append(alg3[3][false_positive][i] / (alg3[3][false_positive][i] + alg3[3][true_negative][i]))

        if alg1[3][true_positive][i] == 0 and alg1[3][false_positive][i] == 0:
            alg1[3][fdr].append(0)
        else:
            alg1[3][fdr].append(alg1[3][false_positive][i] / (alg1[3][false_positive][i] + alg1[3][true_positive][i]))

        if alg2[3][true_positive][i] == 0 and alg2[3][false_positive][i] == 0:
            alg2[3][fdr].append(0)
        else:
            alg2[3][fdr].append(alg2[3][false_positive][i] / (alg2[3][false_positive][i] + alg2[3][true_positive][i]))

        if alg3[3][true_positive][i] == 0 and alg3[3][false_positive][i] == 0:
            alg3[3][fdr].append(0)
        else:
            alg3[3][fdr].append(alg3[3][false_positive][i] / (alg3[3][false_positive][i] + alg3[3][true_positive][i]))

        if alg1[3][true_positive][i] == 0 and alg1[3][false_negative][i] == 0:
            alg1[3][tpr].append(0)
        else:
            alg1[3][tpr].append(alg1[3][true_positive][i] / (alg1[3][true_positive][i] + alg1[3][false_negative][i]))

        if alg2[3][true_positive][i] == 0 and alg2[3][false_negative][i] == 0:
            alg2[3][tpr].append(0)
        else:
            alg2[3][tpr].append(alg2[3][true_positive][i] / (alg2[3][true_positive][i] + alg2[3][false_negative][i]))

        if alg3[3][true_positive][i] == 0 and alg3[3][false_negative][i] == 0:
            alg3[3][tpr].append(0)
        else:
            alg3[3][tpr].append(alg3[3][true_positive][i] / (alg3[3][true_positive][i] + alg3[3][false_negative][i]))

    plot_algorithm_sensitivity(alg1[2], alg1[3][false_positive], data_name + ' Alg1 FP', c.output_path(), data_name + '_Alg1_FP')
    plot_algorithm_sensitivity(alg1[2], alg1[3][fp_divided_by_tp], data_name + ' Alg1 FP / TP', c.output_path(), data_name + '_Alg1_FP_Divided_By_TP')
    plot_algorithm_sensitivity(alg1[2], alg1[3][fpr], data_name + ' Alg1 FPR', c.output_path(), data_name + '_Alg1_FPR')
    plot_algorithm_sensitivity(alg1[2], alg1[3][fdr], data_name + ' Alg1 FDR', c.output_path(), data_name + '_Alg1_FDR')

    plot_algorithm_sensitivity(alg2[2], alg2[3][false_positive], data_name + ' Alg2 FP', c.output_path(), data_name + '_Alg2_FP')
    plot_algorithm_sensitivity(alg2[2], alg2[3][fp_divided_by_tp], data_name + ' Alg2 FP / TP', c.output_path(), data_name + '_Alg2_FP_Divided_By_TP')
    plot_algorithm_sensitivity(alg2[2], alg2[3][fpr], data_name + ' Alg2 FPR', c.output_path(), data_name + '_Alg2_FPR')
    plot_algorithm_sensitivity(alg2[2], alg2[3][fdr], data_name + ' Alg2 FDR', c.output_path(), data_name + '_Alg2_FDR')

    plot_algorithm_sensitivity(alg3[2], alg3[3][false_positive], data_name + ' Alg3 FP', c.output_path(), data_name + '_Alg3_FP')
    plot_algorithm_sensitivity(alg3[2], alg3[3][fp_divided_by_tp], data_name + ' Alg3 FP / TP', c.output_path(), data_name + '_Alg3_FP_Divided_By_TP')
    plot_algorithm_sensitivity(alg3[2], alg3[3][fpr], data_name + ' Alg3 FPR', c.output_path(), data_name + '_Alg3_FPR')
    plot_algorithm_sensitivity(alg3[2], alg3[3][fdr], data_name + ' Alg3 FDR', c.output_path(), data_name + '_Alg3_FDR')

    plot_all_algorithms_sensitivity(alg1[3][true_positive], alg2[3][true_positive], alg3[3][true_positive], 'True Positive', c.output_path(), data_name + '_True_Positive')
    plot_all_algorithms_sensitivity(alg1[3][false_positive], alg2[3][false_positive], alg3[3][false_positive], 'False Positive', c.output_path(), data_name + '_False_Positive')
    plot_all_algorithms_sensitivity(alg1[3][fp_divided_by_tp], alg2[3][fp_divided_by_tp], alg3[3][fp_divided_by_tp], 'False Positive Divided by True Positive', c.output_path(), data_name + '_FP_Divided_By_Tp')
    plot_all_algorithms_sensitivity(alg1[3][true_negative], alg2[3][true_negative], alg3[3][true_negative], 'True Negative', c.output_path(), data_name + '_True_Negative')
    plot_all_algorithms_sensitivity(alg1[3][fpr], alg2[3][fpr], alg3[3][fpr], 'False Positive Rate', c.output_path(), data_name + '_False_Positive_Rate')
    plot_all_algorithms_sensitivity(alg1[3][fdr], alg2[3][fdr], alg3[3][fdr], 'False Discovery Rate', c.output_path(), data_name + '_False_Discovery_Rate')
    plot_all_algorithms_sensitivity(alg1[3][tpr], alg2[3][tpr], alg3[3][tpr], 'True Positive Rate', c.output_path(), data_name + '_True_Positive_Rate')