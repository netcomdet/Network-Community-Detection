from Utils import *


class Community:
    def __init__(self, g, c1, c2):
        self._graph = g.copy()
        self._c1 = c1
        self._c2 = c2

        self._community = []
        self._commonality_calculation = {}

        self._log_space = 0

    def _get_commonality(self, node1, node2):
        if node1 > node2:
            min_node = node2
            max_node = node1
        else:
            min_node = node1
            max_node = node2

        if (min_node, max_node) not in self._commonality_calculation:
            numerator, denominator = calculate_commonality(self._graph, node1, node2)
            self._commonality_calculation[(min_node, max_node)] = numerator / denominator

        return self._commonality_calculation[(min_node, max_node)]

    def get_node_community(self, node):
        self._community = []
        print_log('_check_node_neighbors_for_commonality: ' + str(node), self._log_space)
        for node_neighbor in self._graph.neighbors(node):
            print_log('_check_node_neighbors_for_commonality node_neighbor: ' + str(node_neighbor), self._log_space)
            commonality_c1 = self._get_commonality(node, node_neighbor)
            print_log('_check_node_neighbors_for_commonality node_neighbor commonality: ' + str(commonality_c1), self._log_space)
            if commonality_c1 > self._c1:
                self._check_second_node_neighbors_for_commonality(node, node_neighbor)
                self._check_second_node_neighbors_for_commonality(node_neighbor, node)

        return self._community

    def _check_second_node_neighbors_for_commonality(self, node1, node2):
        self._log_space += 1
        print_log('_check_second_node_neighbors_for_commonality: ' + str(node1) + ' ' + str(node2), self._log_space)
        for node_neighbor in self._graph.neighbors(node2):
            print_log('_check_second_node_neighbors_for_commonality node_neighbor: ' + str(node_neighbor), self._log_space)
            if node1 != node_neighbor and (node1 not in self._community or node2 not in self._community or node_neighbor not in self._community):
                commonality_c1 = self._get_commonality(node2, node_neighbor)
                commonality_c2 = self._get_commonality(node1, node_neighbor)
                print_log('_check_second_node_neighbors_for_commonality commonality_c1: ' + str(commonality_c1), self._log_space)
                print_log('_check_second_node_neighbors_for_commonality commonality_c2: ' + str(commonality_c2), self._log_space)
                if commonality_c1 > self._c1 and commonality_c2 > self._c2:
                    print_log('_check_second_node_neighbors_for_commonality inside if', self._log_space)
                    if node1 not in self._community:
                        self._community.append(node1)
                    if node2 not in self._community:
                        self._community.append(node2)
                    if node_neighbor not in self._community:
                        self._community.append(node_neighbor)

                    self._check_second_node_neighbors_for_commonality(node2, node_neighbor)

        self._log_space -= 1

    def get_communities(self):
        temp = self._graph.copy()
        communities = []
        while len(self._graph.nodes) > 0:
            node = random.choice(list(self._graph.nodes))
            community = self.get_node_community(node)

            if not self._community:
                self._graph.remove_node(node)
            else:
                communities.append(community)
                for i in range(len(community)):
                    for j in range(i + 1, len(community)):
                        if self._graph.has_edge(community[i], community[j]):
                            self._graph.remove_edge(community[i], community[j])

            iso = list(nx.isolates(self._graph))
            for iso_node in iso:
                self._graph.remove_node(iso_node)

        self._graph = temp.copy()

        return communities
