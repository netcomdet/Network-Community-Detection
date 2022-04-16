from Mongo import Mongo
from pymongo import UpdateOne
import networkx as nx
import os


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
