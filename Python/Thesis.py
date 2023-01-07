from Community import *
from community import community_louvain
# from communities.algorithms import girvan_newman, hierarchical_clustering, bron_kerbosch
from communities.algorithms import hierarchical_clustering, bron_kerbosch
from datetime import datetime
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


def convert_adj_communities_sets_to_list(community_set):
    community_list = []

    for community in community_set:
        community_list.append(list(community))

    return community_list


def get_adj_matrix(g):
    return np.array(nx.adjacency_matrix(g).todense())


def plot_similarity(similarity_list, n, z, m, title):
    x_ticks = list(range(0, 21, 1))

    plt.clf()
    plt.xticks(x_ticks)
    plt.plot(x_ticks, similarity_list)
    plt.title(title + ' n=' + str(n) + ' z=' + str(z) + ' m=' + str(m), fontsize=18)
    plt.savefig('Algorithm Compare\\Separate\\' + title + '_n=' + str(n) + ' z=' + str(z) + ' m=' + str(m) + '.png')


def plot_all_similarities(similarity_list_commonality1, similarity_list_commonality2, similarity_list_commonality3, similarity_list_louvain_method, similarity_list_hierarchical_clustering, similarity_list_bron_kerbosch, n, z, m, title):
    x_ticks = list(range(0, 21, 1))

    plt.clf()
    plt.xticks(x_ticks)
    plt.plot(x_ticks, similarity_list_commonality1, 'r')
    plt.plot(x_ticks, similarity_list_commonality2, 'g')
    plt.plot(x_ticks, similarity_list_commonality3, 'k')
    plt.plot(x_ticks, similarity_list_louvain_method, 'b')
    plt.plot(x_ticks, similarity_list_hierarchical_clustering, 'y')
    plt.plot(x_ticks, similarity_list_bron_kerbosch, 'm')
    plt.title(title + ' n=' + str(n) + ' z=' + str(z) + ' m=' + str(m), fontsize=18)
    plt.savefig('Algorithm Compare\\All\\' + title + '_n=' + str(n) + ' z=' + str(z) + ' m=' + str(m) + '.png')


def lattice_similarity(n, z, m, p_in, p_out, p_both, l, iterations):
    times = 10

    similarity_list_in_commonality1 = [0] * (iterations + 1)
    similarity_list_in_commonality2 = [0] * (iterations + 1)
    similarity_list_in_commonality3 = [0] * (iterations + 1)
    similarity_list_in_louvain_method = [0] * (iterations + 1)
    similarity_list_in_hierarchical_clustering = [0] * (iterations + 1)
    similarity_list_in_bron_kerbosch = [0] * (iterations + 1)

    similarity_list_out_commonality1 = [0] * (iterations + 1)
    similarity_list_out_commonality2 = [0] * (iterations + 1)
    similarity_list_out_commonality3 = [0] * (iterations + 1)
    similarity_list_out_louvain_method = [0] * (iterations + 1)
    similarity_list_out_hierarchical_clustering = [0] * (iterations + 1)
    similarity_list_out_bron_kerbosch = [0] * (iterations + 1)

    similarity_list_both_commonality1 = [0] * (iterations + 1)
    similarity_list_both_commonality2 = [0] * (iterations + 1)
    similarity_list_both_commonality3 = [0] * (iterations + 1)
    similarity_list_both_louvain_method = [0] * (iterations + 1)
    similarity_list_both_hierarchical_clustering = [0] * (iterations + 1)
    similarity_list_both_bron_kerbosch = [0] * (iterations + 1)

    for t in range(times):
        print('t =', t)
        g_in, ground_truth_in = create_lattice(n, z, m, 0, 0, l)
        g_out, ground_truth_out = create_lattice(n, z, m, 0, 0, l)
        g_both, ground_truth_both = create_lattice(n, z, m, 0, 0, l)

        g_in_distribution = create_distribution(g_in)
        g_out_distribution = create_distribution(g_out)
        g_both_distribution = create_distribution(g_both)

        ground_truth_in_index = ground_truth_to_index(ground_truth_in, g_in)
        ground_truth_out_index = ground_truth_to_index(ground_truth_out, g_out)
        ground_truth_both_index = ground_truth_to_index(ground_truth_both, g_both)

        community_in1 = Community(g_in, 1)
        community_out1 = Community(g_out, 1)
        community_both1 = Community(g_both, 1)

        community_in2 = Community(g_in, 2)
        community_out2 = Community(g_out, 2)
        community_both2 = Community(g_both, 2)

        community_in3 = Community(g_in, 3, z)
        community_out3 = Community(g_out, 3, z)
        community_both3 = Community(g_both, 3, z)

        nodes_number_in = len(g_in.nodes)
        nodes_number_out = len(g_out.nodes)
        nodes_number_both = len(g_both.nodes)

        adj_matrix_in = get_adj_matrix(g_in)
        adj_matrix_out = get_adj_matrix(g_out)
        adj_matrix_both = get_adj_matrix(g_both)

        print(1, datetime.now())
        communities_in_commonality1 = community_in1.get_communities()
        print(2, datetime.now())
        communities_in_commonality2 = community_in2.get_communities()
        print(3, datetime.now())
        communities_in_commonality3 = community_in3.get_communities()
        print(4, datetime.now())
        communities_in_louvain_method = louvain_community(g_in)
        print(5, datetime.now())
        communities_in_hierarchical_clustering = convert_adj_communities_sets_to_list(hierarchical_clustering(adj_matrix_in))
        print(6, datetime.now())
        communities_in_bron_kerbosch = convert_adj_communities_sets_to_list(bron_kerbosch(adj_matrix_in))
        print(7, datetime.now())

        communities_out_commonality1 = community_out1.get_communities()
        communities_out_commonality2 = community_out2.get_communities()
        communities_out_commonality3 = community_out3.get_communities()
        communities_out_louvain_method = louvain_community(g_out)
        communities_out_hierarchical_clustering = convert_adj_communities_sets_to_list(hierarchical_clustering(adj_matrix_out))
        communities_out_bron_kerbosch = convert_adj_communities_sets_to_list(bron_kerbosch(adj_matrix_out))

        communities_both_commonality1 = community_both1.get_communities()
        communities_both_commonality2 = community_both2.get_communities()
        communities_both_commonality3 = community_both3.get_communities()
        communities_both_louvain_method = louvain_community(g_both)
        communities_both_hierarchical_clustering = convert_adj_communities_sets_to_list(hierarchical_clustering(adj_matrix_both))
        communities_both_bron_kerbosch = convert_adj_communities_sets_to_list(bron_kerbosch(adj_matrix_both))

        similarity_list_in_commonality1[0] += get_similarity(communities_in_commonality1, ground_truth_in, nodes_number_in) / times
        similarity_list_in_commonality2[0] += get_similarity(communities_in_commonality2, ground_truth_in, nodes_number_in) / times
        similarity_list_in_commonality3[0] += get_similarity(communities_in_commonality3, ground_truth_in, nodes_number_in) / times
        similarity_list_in_louvain_method[0] += get_similarity(communities_in_louvain_method, ground_truth_in, nodes_number_in) / times
        similarity_list_in_hierarchical_clustering[0] += get_similarity(communities_in_hierarchical_clustering, ground_truth_in_index, nodes_number_in) / times
        similarity_list_in_bron_kerbosch[0] += get_similarity(communities_in_bron_kerbosch, ground_truth_in_index, nodes_number_in) / times

        similarity_list_out_commonality1[0] += get_similarity(communities_out_commonality1, ground_truth_out, nodes_number_out) / times
        similarity_list_out_commonality2[0] += get_similarity(communities_out_commonality2, ground_truth_out, nodes_number_out) / times
        similarity_list_out_commonality3[0] += get_similarity(communities_out_commonality3, ground_truth_out, nodes_number_out) / times
        similarity_list_out_louvain_method[0] += get_similarity(communities_out_louvain_method, ground_truth_out, nodes_number_out) / times
        similarity_list_out_hierarchical_clustering[0] += get_similarity(communities_out_hierarchical_clustering, ground_truth_out_index, nodes_number_out) / times
        similarity_list_out_bron_kerbosch[0] += get_similarity(communities_out_bron_kerbosch, ground_truth_out_index, nodes_number_out) / times

        similarity_list_both_commonality1[0] += get_similarity(communities_both_commonality1, ground_truth_both, nodes_number_both) / times
        similarity_list_both_commonality2[0] += get_similarity(communities_both_commonality2, ground_truth_both, nodes_number_both) / times
        similarity_list_both_commonality3[0] += get_similarity(communities_both_commonality3, ground_truth_both, nodes_number_both) / times
        similarity_list_both_louvain_method[0] += get_similarity(communities_both_louvain_method, ground_truth_both, nodes_number_both) / times
        similarity_list_both_hierarchical_clustering[0] += get_similarity(communities_both_hierarchical_clustering, ground_truth_both_index, nodes_number_both) / times
        similarity_list_both_bron_kerbosch[0] += get_similarity(communities_both_bron_kerbosch, ground_truth_both_index, nodes_number_both) / times

        for i in range(1, iterations + 1):
            print('i =', i)
            randomize_lattice(g_in, p_in, 0, 0, g_in_distribution)
            randomize_lattice(g_out, 0, p_out, 0, g_out_distribution)
            randomize_lattice(g_both, p_both, p_both, 0, g_both_distribution)

            adj_matrix_in = get_adj_matrix(g_in)
            adj_matrix_out = get_adj_matrix(g_out)
            adj_matrix_both = get_adj_matrix(g_both)

            communities_in_commonality1 = community_in1.get_communities()
            communities_in_commonality2 = community_in2.get_communities()
            communities_in_commonality3 = community_in3.get_communities()
            communities_in_louvain_method = louvain_community(g_in)
            communities_in_hierarchical_clustering = convert_adj_communities_sets_to_list(hierarchical_clustering(adj_matrix_in))
            communities_in_bron_kerbosch = convert_adj_communities_sets_to_list(bron_kerbosch(adj_matrix_in))

            communities_out_commonality1 = community_out1.get_communities()
            communities_out_commonality2 = community_out2.get_communities()
            communities_out_commonality3 = community_out3.get_communities()
            communities_out_louvain_method = louvain_community(g_out)
            communities_out_hierarchical_clustering = convert_adj_communities_sets_to_list(hierarchical_clustering(adj_matrix_out))
            communities_out_bron_kerbosch = convert_adj_communities_sets_to_list(bron_kerbosch(adj_matrix_out))

            communities_both_commonality1 = community_both1.get_communities()
            communities_both_commonality2 = community_both2.get_communities()
            communities_both_commonality3 = community_both3.get_communities()
            communities_both_louvain_method = louvain_community(g_both)
            communities_both_hierarchical_clustering = convert_adj_communities_sets_to_list(hierarchical_clustering(adj_matrix_both))
            communities_both_bron_kerbosch = convert_adj_communities_sets_to_list(bron_kerbosch(adj_matrix_both))

            similarity_in1 = get_similarity(communities_in_commonality1, ground_truth_in, nodes_number_in)
            similarity_in2 = get_similarity(communities_in_commonality2, ground_truth_in, nodes_number_in)
            similarity_in3 = get_similarity(communities_in_commonality3, ground_truth_in, nodes_number_in)
            similarity_in_louvain_method = get_similarity(communities_in_louvain_method, ground_truth_in, nodes_number_in)
            similarity_in_hierarchical_clustering = get_similarity(communities_in_hierarchical_clustering, ground_truth_in_index, nodes_number_in)
            similarity_in_bron_kerbosch = get_similarity(communities_in_bron_kerbosch, ground_truth_in_index, nodes_number_in)

            similarity_out1 = get_similarity(communities_out_commonality1, ground_truth_out, nodes_number_out)
            similarity_out2 = get_similarity(communities_out_commonality2, ground_truth_out, nodes_number_out)
            similarity_out3 = get_similarity(communities_out_commonality3, ground_truth_out, nodes_number_out)
            similarity_out_louvain_method = get_similarity(communities_out_louvain_method, ground_truth_out, nodes_number_out)
            similarity_out_hierarchical_clustering = get_similarity(communities_out_hierarchical_clustering, ground_truth_out_index, nodes_number_out)
            similarity_out_bron_kerbosch = get_similarity(communities_out_bron_kerbosch, ground_truth_out_index, nodes_number_out)

            similarity_both1 = get_similarity(communities_both_commonality1, ground_truth_both, nodes_number_both)
            similarity_both2 = get_similarity(communities_both_commonality2, ground_truth_both, nodes_number_both)
            similarity_both3 = get_similarity(communities_both_commonality3, ground_truth_both, nodes_number_both)
            similarity_both_louvain_method = get_similarity(communities_both_louvain_method, ground_truth_both, nodes_number_both)
            similarity_both_hierarchical_clustering = get_similarity(communities_both_hierarchical_clustering, ground_truth_both_index, nodes_number_both)
            similarity_both_bron_kerbosch = get_similarity(communities_both_bron_kerbosch, ground_truth_both_index, nodes_number_both)

            similarity_list_in_commonality1[i] += similarity_in1 / times
            similarity_list_in_commonality2[i] += similarity_in2 / times
            similarity_list_in_commonality3[i] += similarity_in3 / times
            similarity_list_in_louvain_method[i] += similarity_in_louvain_method / times
            similarity_list_in_hierarchical_clustering[i] += similarity_in_hierarchical_clustering / times
            similarity_list_in_bron_kerbosch[i] += similarity_in_bron_kerbosch / times

            similarity_list_out_commonality1[i] += similarity_out1 / times
            similarity_list_out_commonality2[i] += similarity_out2 / times
            similarity_list_out_commonality3[i] += similarity_out3 / times
            similarity_list_out_louvain_method[i] += similarity_out_louvain_method / times
            similarity_list_out_hierarchical_clustering[i] += similarity_out_hierarchical_clustering / times
            similarity_list_out_bron_kerbosch[i] += similarity_out_bron_kerbosch / times

            similarity_list_both_commonality1[i] += similarity_both1 / times
            similarity_list_both_commonality2[i] += similarity_both2 / times
            similarity_list_both_commonality3[i] += similarity_both3 / times
            similarity_list_both_louvain_method[i] += similarity_both_louvain_method / times
            similarity_list_both_hierarchical_clustering[i] += similarity_both_hierarchical_clustering / times
            similarity_list_both_bron_kerbosch[i] += similarity_both_bron_kerbosch / times

    plot_similarity(similarity_list_in_commonality1, n, z, m, 'In_Commonality_Alg1')
    plot_similarity(similarity_list_in_commonality2, n, z, m, 'In_Commonality_Alg2')
    plot_similarity(similarity_list_in_commonality3, n, z, m, 'In_Commonality_Alg3')
    plot_similarity(similarity_list_in_louvain_method, n, z, m, 'In_Louvain_Method')
    plot_similarity(similarity_list_in_hierarchical_clustering, n, z, m, 'In_Hierarchical_Clustering')
    plot_similarity(similarity_list_in_bron_kerbosch, n, z, m, 'In_Bron_Kerbosch')
    plot_all_similarities(similarity_list_in_commonality1, similarity_list_in_commonality2, similarity_list_in_commonality3, similarity_list_in_louvain_method, similarity_list_in_hierarchical_clustering, similarity_list_in_bron_kerbosch, n, z, m, 'In_All')

    plot_similarity(similarity_list_out_commonality1, n, z, m, 'Out_Commonality_Alg1')
    plot_similarity(similarity_list_out_commonality2, n, z, m, 'Out_Commonality_Alg2')
    plot_similarity(similarity_list_out_commonality3, n, z, m, 'Out_Commonality_Alg3')
    plot_similarity(similarity_list_out_louvain_method, n, z, m, 'Out_Louvain_Method')
    plot_similarity(similarity_list_out_hierarchical_clustering, n, z, m, 'Out_Hierarchical_Clustering')
    plot_similarity(similarity_list_out_bron_kerbosch, n, z, m, 'Out_Bron_Kerbosch')
    plot_all_similarities(similarity_list_out_commonality1, similarity_list_out_commonality2, similarity_list_out_commonality3, similarity_list_out_louvain_method, similarity_list_out_hierarchical_clustering, similarity_list_out_bron_kerbosch, n, z, m, 'Out_All')

    plot_similarity(similarity_list_both_commonality1, n, z, m, 'Both_Commonality_Alg1')
    plot_similarity(similarity_list_both_commonality2, n, z, m, 'Both_Commonality_Alg2')
    plot_similarity(similarity_list_both_commonality3, n, z, m, 'Both_Commonality_Alg3')
    plot_similarity(similarity_list_both_louvain_method, n, z, m, 'Both_Louvain_Method')
    plot_similarity(similarity_list_both_hierarchical_clustering, n, z, m, 'Both_Hierarchical_Clustering')
    plot_similarity(similarity_list_both_bron_kerbosch, n, z, m, 'Both_Bron_Kerbosch')
    plot_all_similarities(similarity_list_both_commonality1, similarity_list_both_commonality2, similarity_list_both_commonality3, similarity_list_both_louvain_method, similarity_list_both_hierarchical_clustering, similarity_list_both_bron_kerbosch, n, z, m, 'Both_All')


for different_item in different:
    print(different_item)
    lattice_similarity(different_item[0], different_item[1], different_item[2], 10, 10, 5, 0, 20)
