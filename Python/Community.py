from Utils import *


class Community:
    def __init__(self, g):
        self._graph = g
        self._community_graph = None

        self._community = []
        self._community_for_similarity = []
        self._commonality_calculation = {}

        self._reccursion_stack = []
        self._pair_checked = []

        self._set_threshold()

    def _set_threshold(self):
        s = 0
        count = 0

        shortest_path_gen = nx.all_pairs_shortest_path(self._graph, 2)

        for shortest_path in shortest_path_gen:
            node_from = shortest_path[0]
            nodes_to = shortest_path[1]

            for node_to in nodes_to:
                if node_from < node_to:
                    len_path = len(nodes_to[node_to])

                    if len_path == 2 or len_path == 3:
                        s += self._get_commonality(node_from, node_to)
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
            args = self._reccursion_stack.pop()
            self._check_second_node_neighbors_for_commonality(args[0], args[1], args[2])

    def _append_stack(self, node1, node2, indent):
        if (node1, node2) not in self._pair_checked:
            self._reccursion_stack.append([node1, node2, indent])
            self._pair_checked.append((node1, node2))

    def get_node_community(self, node):
        self._community = []
        self._pair_checked = []

        for node_neighbor in self._community_graph.neighbors(node):
            if not self._community:
                commonality = self._get_commonality(node, node_neighbor)

                if commonality > self._commonality_threshold:
                    self._append_stack(node, node_neighbor, 0)
                    self._append_stack(node_neighbor, node, 0)

                    self._run_reccursion()

    def _check_second_node_neighbors_for_commonality(self, node1, node2, indent):
        for node_neighbor in self._community_graph.neighbors(node2):
            if node1 != node_neighbor and (node2, node_neighbor) not in self._pair_checked:
                commonality_d1 = self._get_commonality(node2, node_neighbor)
                commonality_d2 = self._get_commonality(node1, node_neighbor)

                if commonality_d1 > self._commonality_threshold and commonality_d2 > self._commonality_threshold:
                    if node1 not in self._community:
                        self._community.append(node1)
                    if node2 not in self._community:
                        self._community.append(node2)
                    if node_neighbor not in self._community:
                        self._community.append(node_neighbor)

                    self._append_stack(node2, node_neighbor, indent + 1)
                    self._append_stack(node_neighbor, node2, indent + 1)

    def get_communities(self):
        self._community_graph = self._graph.copy()
        communities = []
        while len(self._community_graph.nodes) > 0:
            node = random.choice(list(self._community_graph.nodes))

            self.get_node_community(node)

            if not self._community:
                self._community_graph.remove_node(node)
            else:
                communities.append(self._community)
                for i in range(len(self._community)):
                    for j in range(i + 1, len(self._community)):
                        if self._community_graph.has_edge(self._community[i], self._community[j]):
                            self._community_graph.remove_edge(self._community[i], self._community[j])

            iso = list(nx.isolates(self._community_graph))
            for iso_node in iso:
                self._community_graph.remove_node(iso_node)

        self._community_graph = None

        return communities
