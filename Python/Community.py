from Utils import *


class Community:
    def __init__(self):
        self._graph = None

        self._community = []
        self._commonality_calculation = {}

        self._log_space = 0

        self._reccursion_stack = []
        self._pair_checked = []

    def _init_graph(self, g):
        self._graph = g.copy()
        self._set_threshold()

    def _set_threshold(self):
        s = 0
        count = 0
        for n in list(self._graph.nodes):
            for n_n in self._graph.neighbors(n):
                s += self._get_commonality(n, n_n)
                count += 1

        self._commonality_threshold = (s / count) * 0.7

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

    def _run_reccursion(self):
        while self._reccursion_stack:
            # print(self._t)
            args = self._reccursion_stack.pop()
            # print(str(func))
            self._check_second_node_neighbors_for_commonality(args[0], args[1])

    def get_node_community(self, node):
        self._community = []
        self._pair_checked = []
        print_log('get_node_community node_neighbor: ' + str(node))
        for node_neighbor in self._graph.neighbors(node):
            if not self._community:
                print_log('get_node_community node_neighbor: ' + str(node_neighbor))
                commonality = self._get_commonality(node, node_neighbor)
                print_log('get_node_community node_neighbor commonality: ' + str(commonality))
                if commonality > self._commonality_threshold:
                    self._reccursion_stack.append([node, node_neighbor])
                    self._reccursion_stack.append([node_neighbor, node])

                    self._run_reccursion()

        return self._community

    def _check_second_node_neighbors_for_commonality(self, node1, node2):
        if (node1, node2) not in self._pair_checked:
            self._pair_checked.append((node1, node2))
            self._log_space += 1
            print_log('_check_second_node_neighbors_for_commonality: ' + str(node1) + ' ' + str(node2))
            for node_neighbor in self._graph.neighbors(node2):
                print_log('_check_second_node_neighbors_for_commonality node_neighbor: ' + str(node_neighbor))
                if node1 != node_neighbor:
                    commonality_d1 = self._get_commonality(node2, node_neighbor)
                    commonality_d2 = self._get_commonality(node1, node_neighbor)
                    print_log('_check_second_node_neighbors_for_commonality commonality: ' + str(commonality_d1))
                    print_log('_check_second_node_neighbors_for_commonality commonality: ' + str(commonality_d2))
                    if commonality_d1 > self._commonality_threshold and commonality_d2 > self._commonality_threshold:
                        print_log('_check_second_node_neighbors_for_commonality inside if')
                        if node1 not in self._community:
                            self._community.append(node1)
                        if node2 not in self._community:
                            self._community.append(node2)
                        if node_neighbor not in self._community:
                            self._community.append(node_neighbor)

                        self._reccursion_stack.append([node2, node_neighbor])

        self._log_space -= 1

    def get_communities(self, g):
        self._init_graph(g)

        temp = self._graph.copy()
        communities = []
        while len(self._graph.nodes) > 0:
            node = random.choice(list(self._graph.nodes))

            community = self.get_node_community(node)

            if not community:
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

        self._graph = temp

        return communities
