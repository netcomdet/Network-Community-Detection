# from Utils import calculate_commonality
from Utils import *
import networkx as nx


class Community:
    def __init__(self, g, algorithm, z=None):
        self._graph = g
        self._community_graph = None

        self._community_for_similarity = []
        self._commonality_calculation = {}

        self._reccursion_stack = []

        self._community = None
        # self._community_edges = None
        self._pair_checked = None

        match algorithm:
            case 1:
                self.get_commonality = self._get_commonality1
            case 2:
                self.get_commonality = self._get_commonality2
            case 3:
                self.get_commonality = self._get_commonality3
                self._z = z

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

                    if len_path == 2:
                        s += self.get_commonality(node_from, node_to)
                        count += 1

        self._commonality_threshold = (s / count) * 0.75

    def _get_commonality1(self, node1, node2):
        if node1 > node2:
            min_node = node2
            max_node = node1
        else:
            min_node = node1
            max_node = node2

        if (min_node, max_node) not in self._commonality_calculation:
            numerator, denominator = calculate_commonality(self._graph, node1, node2)

            self._commonality_calculation[(min_node, max_node)] = numerator

        return self._commonality_calculation[(min_node, max_node)]

    def _get_commonality2(self, node1, node2):
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

    def _get_commonality3(self, node1, node2):
        if node1 > node2:
            min_node = node2
            max_node = node1
        else:
            min_node = node1
            max_node = node2

        if (min_node, max_node) not in self._commonality_calculation:
            numerator, denominator = calculate_commonality(self._graph, node1, node2)
            self._commonality_calculation[(min_node, max_node)] = (numerator * numerator) / (denominator * self._z)

        return self._commonality_calculation[(min_node, max_node)]

    def _run_reccursion(self):
        while self._reccursion_stack:
            args = self._reccursion_stack.pop()
            self._check_second_node_neighbors_for_commonality(args[0], args[1])

    def _append_stack(self, node1, node2):
        self._reccursion_stack.append([node1, node2])

        if node1 not in self._pair_checked:
            self._pair_checked[node1] = []

        self._pair_checked[node1].append(node2)

    def get_node_community(self, node):
        self._community = set([])
        # self._community_edges = []
        self._pair_checked = {}

        node_neighbors = list(self._community_graph.neighbors(node))

        for node_neighbor in node_neighbors:
            if not self._community:
                commonality = self.get_commonality(node, node_neighbor)

                if commonality > self._commonality_threshold:
                    self._append_stack(node, node_neighbor)
                    self._append_stack(node_neighbor, node)

                    self._run_reccursion()

                else:
                    self._community_graph.remove_edge(node, node_neighbor)

    def check_pair_checked(self, node1, node2):
        if node1 not in self._pair_checked:
            return True
        else:
            if node2 not in self._pair_checked[node1]:
                return True
        return False

    def _check_second_node_neighbors_for_commonality(self, node1, node2):
        node2_neighbors = list(self._community_graph.neighbors(node2))

        if node1 in node2_neighbors:
            node2_neighbors.remove(node1)

        for node_neighbor in node2_neighbors:
            commonality_d1 = self.get_commonality(node2, node_neighbor)
            commonality_d2 = self.get_commonality(node1, node_neighbor)

            if commonality_d1 > self._commonality_threshold:
                if commonality_d2 > self._commonality_threshold:
                    self._community.add(node1)
                    self._community.add(node2)
                    self._community.add(node_neighbor)

                    if self._community_graph.has_edge(node1, node2):
                        self._community_graph.remove_edge(node1, node2)

                    self._append_stack(node2, node_neighbor)
                    self._append_stack(node_neighbor, node2)
            else:
                self._community_graph.remove_edge(node2, node_neighbor)

    def get_communities(self):
        self._community_graph = self._graph.copy()
        communities = []

        while len(self._community_graph.nodes) > 0:
            node = random.choice(list(self._community_graph.nodes))

            self.get_node_community(node)

            if not self._community:
                self._community_graph.remove_node(node)
            else:
                community_list = list(self._community)
                communities.append(community_list)

            iso = list(nx.isolates(self._community_graph))
            for iso_node in iso:
                self._community_graph.remove_node(iso_node)

        self._community_graph = None

        return communities
