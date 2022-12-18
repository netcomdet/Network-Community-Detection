import matplotlib.pyplot as plt
import networkx as nx
import random
import os

from Mongo import Mongo
from pymongo import UpdateOne

PRINT_LOG = False


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


def randomize_in(g, p_in, i, lattice_size, z):
    for _ in range(int(p_in * lattice_size * (z / 2) / 100)):
        neighbours = []
        while len(neighbours) == 0:
            while True:
                f = get_node_name_for_lattice(i, random.randint(0, lattice_size - 1))
                if f in list(g.nodes):
                    break
            neighbours = []
            for n in list(g.neighbors(str(f))):
                if str(n)[0] == str(i):
                    neighbours.append(n)

        fn = neighbours[random.randint(0, len(neighbours) - 1)]

        t = f

        while g.has_edge(t, f) or t == f:
            t = get_node_name_for_lattice(i, random.randint(0, lattice_size - 1))

        g.remove_edge(f, fn)
        g.add_edge(f, t)


def randomize_out(g, p_out, n, z):
    for _ in range(int(p_out * n * (z / 2) / 100)):
        neighbours = []

        while len(neighbours) == 0:
            f = list(g.nodes)[random.randint(0, len(g.nodes) - 1)]
            neighbours = list(g.neighbors(str(f)))

        fn = neighbours[random.randint(0, len(neighbours) - 1)]

        t = list(g.nodes)[random.randint(0, len(g.nodes) - 1)]

        while g.has_edge(t, f) or t == f or str(t)[0] == str(f)[0]:
            t = list(g.nodes)[random.randint(0, len(g.nodes) - 1)]

        g.remove_edge(f, fn)
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
                        community.append(node_removed[1])

    attributes = {'n': n, 'z': z, 'm': m}
    g.graph.update(attributes)

    return g, ground_truth


def randomize_lattice(g, p_in, p_out, l):
    n = g.graph['n']
    z = g.graph['z']
    m = g.graph['m']

    lattice_size = int(n / m)

    for i in range(m):
        randomize_in(g, p_in, i, lattice_size, z)

    if m > 1:
        randomize_out(g, p_out, n, z)

        create_overlap(g, l, n)


def print_log(s):
    if PRINT_LOG:
        # print(' ' * indent + s)
        print(s)


def get_similarity(ground_truth, communities, nodes_number):
    delta = 0
    for c in communities:
        c_set = set(c)

        max_index = -1
        max_intersection = 0
        max_intersection_len = 0

        for i in range(len(ground_truth)):
            current_intersection = c_set.intersection(ground_truth[i])
            current_intersection_len = len(current_intersection)
            if len(current_intersection) > max_intersection_len:
                max_index = i
                max_intersection = current_intersection
                max_intersection_len = current_intersection_len

        t = list((c_set | set(ground_truth[max_index])) - max_intersection)

        delta += len(t)

    return 1 - (delta / nodes_number)


def print_communities(communities):
    for c in communities:
        print(c)


def draw(g):
    nx.draw(g, with_labels=True)

    plt.draw()
    plt.show()
