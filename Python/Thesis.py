import numpy as np

from Community import *
from community import community_louvain
from communities.algorithms import girvan_newman, hierarchical_clustering, bron_kerbosch


different = [(100, 4, 2), (100, 8, 2), (500, 16, 2), (500, 16, 5), (500, 16, 10), (1000, 16, 5), (1000, 32, 5)]


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


def convert_adj_communities_sets_to_list(community_list):
    l = []

    for community in community_list:
        l.append(list(community))

    return l


def get_adj_matrix(g):
    return np.array(nx.adjacency_matrix(g).todense())


def plot_similarity(similarity_list, n, z, m, title):
    x_ticks = list(range(0, 21, 1))

    plt.clf()
    plt.xticks(x_ticks)
    plt.plot(x_ticks, similarity_list)
    plt.title(title + ' n=' + str(n) + ' z=' + str(z) + ' m=' + str(m), fontsize=18)
    plt.savefig(title + '_n=' + str(n) + ' z=' + str(z) + ' m=' + str(m) + '.png')


def plot_all_similarities(similarity_list_commonality, similarity_list_louvain_method, similarity_list_girvan_newman, similarity_list_hierarchical_clustering, similarity_list_bron_kerbosch, n, z, m, title):
    x_ticks = list(range(0, 21, 1))

    plt.clf()
    plt.xticks(x_ticks)
    plt.plot(x_ticks, similarity_list_commonality, 'r')
    plt.plot(x_ticks, similarity_list_louvain_method, 'b')
    plt.plot(x_ticks, similarity_list_girvan_newman, 'g')
    plt.plot(x_ticks, similarity_list_hierarchical_clustering, 'y')
    plt.plot(x_ticks, similarity_list_bron_kerbosch, 'm')
    plt.title(title + ' n=' + str(n) + ' z=' + str(z) + ' m=' + str(m), fontsize=18)
    plt.savefig(title + '_n=' + str(n) + ' z=' + str(z) + ' m=' + str(m) + '.png')


def lattice_similarity(n, z, m, p_in, p_out, p_both, l, iterations):
    times = 10

    similarity_list_in_commonality = [0] * (iterations + 1)
    similarity_list_in_louvain_method = [0] * (iterations + 1)
    similarity_list_in_girvan_newman = [0] * (iterations + 1)
    similarity_list_in_hierarchical_clustering = [0] * (iterations + 1)
    similarity_list_in_bron_kerbosch = [0] * (iterations + 1)

    similarity_list_out_commonality = [0] * (iterations + 1)
    similarity_list_out_louvain_method = [0] * (iterations + 1)
    similarity_list_out_girvan_newman = [0] * (iterations + 1)
    similarity_list_out_hierarchical_clustering = [0] * (iterations + 1)
    similarity_list_out_bron_kerbosch = [0] * (iterations + 1)

    similarity_list_both_commonality = [0] * (iterations + 1)
    similarity_list_both_louvain_method = [0] * (iterations + 1)
    similarity_list_both_girvan_newman = [0] * (iterations + 1)
    similarity_list_both_hierarchical_clustering = [0] * (iterations + 1)
    similarity_list_both_bron_kerbosch = [0] * (iterations + 1)

    for t in range(times):
        # print('t =', t)
        g_in, ground_truth_in = create_lattice(n, z, m, 0, 0, l)
        g_out, ground_truth_out = create_lattice(n, z, m, 0, 0, l)
        g_both, ground_truth_both = create_lattice(n, z, m, 0, 0, l)

        ground_truth_in_index = ground_truth_to_index(ground_truth_in, g_in)
        ground_truth_out_index = ground_truth_to_index(ground_truth_out, g_out)
        ground_truth_both_index = ground_truth_to_index(ground_truth_both, g_both)

        community_in = Community(g_in)
        community_out = Community(g_out)
        community_both = Community(g_both)

        nodes_number_in = len(g_in.nodes)
        nodes_number_out = len(g_out.nodes)
        nodes_number_both = len(g_both.nodes)

        adj_matrix_in = get_adj_matrix(g_in)
        adj_matrix_out = get_adj_matrix(g_out)
        adj_matrix_both = get_adj_matrix(g_both)

        # print(1, datetime.now())
        communities_in_commonality = community_in.get_communities()
        # print(2, datetime.now())
        communities_in_louvain_method = louvain_community(g_in)
        # print(3, datetime.now())
        communities_in_girvan_newman = convert_adj_communities_sets_to_list(girvan_newman(adj_matrix_in)[0])
        # print(4, datetime.now())
        communities_in_hierarchical_clustering = convert_adj_communities_sets_to_list(hierarchical_clustering(adj_matrix_in))
        # print(5, datetime.now())
        communities_in_bron_kerbosch = convert_adj_communities_sets_to_list(bron_kerbosch(adj_matrix_in))
        # print(6, datetime.now())

        communities_out_commonality = community_out.get_communities()
        communities_out_louvain_method = louvain_community(g_out)
        communities_out_girvan_newman = convert_adj_communities_sets_to_list(girvan_newman(adj_matrix_out)[0])
        communities_out_hierarchical_clustering = convert_adj_communities_sets_to_list(hierarchical_clustering(adj_matrix_out))
        communities_out_bron_kerbosch = convert_adj_communities_sets_to_list(bron_kerbosch(adj_matrix_out))

        communities_both_commonality = community_both.get_communities()
        communities_both_louvain_method = louvain_community(g_both)
        communities_both_girvan_newman = convert_adj_communities_sets_to_list(girvan_newman(adj_matrix_both)[0])
        communities_both_hierarchical_clustering = convert_adj_communities_sets_to_list(hierarchical_clustering(adj_matrix_both))
        communities_both_bron_kerbosch = convert_adj_communities_sets_to_list(bron_kerbosch(adj_matrix_both))

        similarity_list_in_commonality[0] += get_similarity(communities_in_commonality, ground_truth_in, nodes_number_in) / times
        similarity_list_in_louvain_method[0] += get_similarity(communities_in_louvain_method, ground_truth_in, nodes_number_in) / times
        similarity_list_in_girvan_newman[0] += get_similarity(communities_in_girvan_newman, ground_truth_in_index, nodes_number_in) / times
        similarity_list_in_hierarchical_clustering[0] += get_similarity(communities_in_hierarchical_clustering, ground_truth_in_index, nodes_number_in) / times
        similarity_list_in_bron_kerbosch[0] += get_similarity(communities_in_bron_kerbosch, ground_truth_in_index, nodes_number_in) / times

        similarity_list_out_commonality[0] += get_similarity(communities_out_commonality, ground_truth_out, nodes_number_out) / times
        similarity_list_out_louvain_method[0] += get_similarity(communities_out_louvain_method, ground_truth_out, nodes_number_out) / times
        similarity_list_out_girvan_newman[0] += get_similarity(communities_out_girvan_newman, ground_truth_out, nodes_number_out) / times
        similarity_list_out_hierarchical_clustering[0] += get_similarity(communities_out_hierarchical_clustering, ground_truth_out_index, nodes_number_out) / times
        similarity_list_out_bron_kerbosch[0] += get_similarity(communities_out_bron_kerbosch, ground_truth_out_index, nodes_number_out) / times

        similarity_list_both_commonality[0] += get_similarity(communities_both_commonality, ground_truth_both, nodes_number_both) / times
        similarity_list_both_louvain_method[0] += get_similarity(communities_both_louvain_method, ground_truth_both, nodes_number_both) / times
        similarity_list_both_girvan_newman[0] += get_similarity(communities_both_girvan_newman, ground_truth_both, nodes_number_both) / times
        similarity_list_both_hierarchical_clustering[0] += get_similarity(communities_both_hierarchical_clustering, ground_truth_both_index, nodes_number_both) / times
        similarity_list_both_bron_kerbosch[0] += get_similarity(communities_both_bron_kerbosch, ground_truth_both_index, nodes_number_both) / times

        for i in range(1, iterations + 1):
            randomize_lattice(g_in, p_in, 0, 0)
            randomize_lattice(g_out, 0, p_out, 0)
            randomize_lattice(g_both, p_both, p_both, 0)

            adj_matrix_in = get_adj_matrix(g_in)
            adj_matrix_out = get_adj_matrix(g_out)
            adj_matrix_both = get_adj_matrix(g_both)

            # communities_in_spectral_clustering = spectral_clustering()

            communities_in_commonality = community_in.get_communities()
            communities_in_louvain_method = louvain_community(g_in)
            communities_in_girvan_newman = convert_adj_communities_sets_to_list(girvan_newman(adj_matrix_in)[0])
            communities_in_hierarchical_clustering = convert_adj_communities_sets_to_list(hierarchical_clustering(adj_matrix_in))
            communities_in_bron_kerbosch = convert_adj_communities_sets_to_list(bron_kerbosch(adj_matrix_in))

            communities_out_commonality = community_out.get_communities()
            communities_out_louvain_method = louvain_community(g_out)
            communities_out_girvan_newman = convert_adj_communities_sets_to_list(girvan_newman(adj_matrix_out)[0])
            communities_out_hierarchical_clustering = convert_adj_communities_sets_to_list(hierarchical_clustering(adj_matrix_out))
            communities_out_bron_kerbosch = convert_adj_communities_sets_to_list(bron_kerbosch(adj_matrix_out))

            communities_both_commonality = community_both.get_communities()
            communities_both_louvain_method = louvain_community(g_both)
            communities_both_girvan_newman = convert_adj_communities_sets_to_list(girvan_newman(adj_matrix_both)[0])
            communities_both_hierarchical_clustering = convert_adj_communities_sets_to_list(hierarchical_clustering(adj_matrix_both))
            communities_both_bron_kerbosch = convert_adj_communities_sets_to_list(bron_kerbosch(adj_matrix_both))

            similarity_in = get_similarity(communities_in_commonality, ground_truth_in, nodes_number_in)
            similarity_in_louvain_method = get_similarity(communities_in_louvain_method, ground_truth_in, nodes_number_in)
            similarity_in_girvan_newman = get_similarity(communities_in_girvan_newman, ground_truth_in, nodes_number_in)
            similarity_in_hierarchical_clustering = get_similarity(communities_in_hierarchical_clustering, ground_truth_in_index, nodes_number_in)
            similarity_in_bron_kerbosch = get_similarity(communities_in_bron_kerbosch, ground_truth_in_index, nodes_number_in)

            similarity_out = get_similarity(communities_out_commonality, ground_truth_out, nodes_number_out)
            similarity_out_louvain_method = get_similarity(communities_out_louvain_method, ground_truth_out, nodes_number_out)
            similarity_out_girvan_newman = get_similarity(communities_out_girvan_newman, ground_truth_out, nodes_number_out)
            similarity_out_hierarchical_clustering = get_similarity(communities_out_hierarchical_clustering, ground_truth_out_index, nodes_number_out)
            similarity_out_bron_kerbosch = get_similarity(communities_out_bron_kerbosch, ground_truth_out_index, nodes_number_out)

            similarity_both = get_similarity(communities_both_commonality, ground_truth_both, nodes_number_both)
            similarity_both_louvain_method = get_similarity(communities_both_louvain_method, ground_truth_both, nodes_number_both)
            similarity_both_girvan_newman = get_similarity(communities_both_girvan_newman, ground_truth_both, nodes_number_both)
            similarity_both_hierarchical_clustering = get_similarity(communities_both_hierarchical_clustering, ground_truth_both_index, nodes_number_both)
            similarity_both_bron_kerbosch = get_similarity(communities_both_bron_kerbosch, ground_truth_both_index, nodes_number_both)

            similarity_list_in_commonality[i] += similarity_in / times
            similarity_list_in_louvain_method[i] += similarity_in_louvain_method / times
            similarity_list_in_girvan_newman[i] += similarity_in_girvan_newman / times
            similarity_list_in_hierarchical_clustering[i] += similarity_in_hierarchical_clustering / times
            similarity_list_in_bron_kerbosch[i] += similarity_in_bron_kerbosch / times

            similarity_list_out_commonality[i] += similarity_out / times
            similarity_list_out_louvain_method[i] += similarity_out_louvain_method / times
            similarity_list_out_girvan_newman[i] += similarity_out_girvan_newman / times
            similarity_list_out_hierarchical_clustering[i] += similarity_out_hierarchical_clustering / times
            similarity_list_out_bron_kerbosch[i] += similarity_out_bron_kerbosch / times

            similarity_list_both_commonality[i] += similarity_both / times
            similarity_list_both_louvain_method[i] += similarity_both_louvain_method / times
            similarity_list_both_girvan_newman[i] += similarity_both_girvan_newman / times
            similarity_list_both_hierarchical_clustering[i] += similarity_both_hierarchical_clustering / times
            similarity_list_both_bron_kerbosch[i] += similarity_both_bron_kerbosch / times

    plot_similarity(similarity_list_in_commonality, n, z, m, 'In_Commonality')
    plot_similarity(similarity_list_in_louvain_method, n, z, m, 'In_Louvain_Method')
    plot_similarity(similarity_list_in_girvan_newman, n, z, m, 'In_Girvan_Newman')
    plot_similarity(similarity_list_in_hierarchical_clustering, n, z, m, 'In_Hierarchical_Clustering')
    plot_similarity(similarity_list_in_bron_kerbosch, n, z, m, 'In_Bron_Kerbosch')
    plot_all_similarities(similarity_list_in_commonality, similarity_list_in_louvain_method, similarity_list_in_hierarchical_clustering, n, z, m, 'In_All')

    plot_similarity(similarity_list_out_commonality, n, z, m, 'Out_Commonality')
    plot_similarity(similarity_list_out_louvain_method, n, z, m, 'Out_Louvain_Method')
    plot_similarity(similarity_list_out_girvan_newman, n, z, m, 'Out_Girvan_Newman')
    plot_similarity(similarity_list_out_hierarchical_clustering, n, z, m, 'Out_Hierarchical_Clustering')
    plot_similarity(similarity_list_out_bron_kerbosch, n, z, m, 'Out_Bron_Kerbosch')
    plot_all_similarities(similarity_list_out_commonality, similarity_list_out_louvain_method, similarity_list_out_hierarchical_clustering, n, z, m, 'Out_All')

    plot_similarity(similarity_list_both_commonality, n, z, m, 'Both_Commonality')
    plot_similarity(similarity_list_both_louvain_method, n, z, m, 'Both_Louvain_Method')
    plot_similarity(similarity_list_both_girvan_newman, n, z, m, 'Both_Girvan_Newman')
    plot_similarity(similarity_list_both_hierarchical_clustering, n, z, m, 'Both_Hierarchical_Clustering')
    plot_similarity(similarity_list_both_bron_kerbosch, n, z, m, 'Both_Bron_Kerbosch')
    plot_all_similarities(similarity_list_both_commonality, similarity_list_both_louvain_method, similarity_list_both_hierarchical_clustering, n, z, m, 'Both_All')


for different_item in different:
    print(different_item)
    lattice_similarity(different_item[0], different_item[1], different_item[2], 10, 10, 5, 2, 20)