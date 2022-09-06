from Community import *

y = []

for i in range(21):
    y.append(i)

# different = [(100, 4, 2), (100, 8, 2), (500, 16, 2), (500, 16, 5), (500, 16, 10), (1000, 16, 5), (1000, 32, 5), (1000, 32, 8), (5000, 32, 10), (5000, 64, 10)]
different = [(5000, 64, 10)]


def lattice_similarity(n, z, m, p_in, p_out, l, iterations):
    g, ground_truth = create_lattice(n, z, m, p_in, p_out, l)
    print('1')
    community = Community(g)
    nodes_number = len(g.nodes)
    y = list(range(iterations+1))
    similarity_list = []

    communities = community.get_communities()
    print('2')
    similarity_list.append(get_similarity(communities, ground_truth, nodes_number))
    print('3')

    for i in range(iterations):
        print(i)
        randomize_lattice(g, 5, 5, 0)

        communities = community.get_communities()
        similarity_list.append(get_similarity(communities, ground_truth, nodes_number))

    plt.clf()
    plt.xticks(range(0, 21, 1))
    plt.plot(y, similarity_list)
    plt.title('n=' + str(n) + ' z=' + str(z) + ' m=' + str(m), fontsize=18)
    plt.savefig('Similarity_n=' + str(n) + ' z=' + str(z) + ' m=' + str(m) + '.png')


for different_item in different:
    lattice_similarity(different_item[0], different_item[1], different_item[2], 0, 0, 2, 20)
