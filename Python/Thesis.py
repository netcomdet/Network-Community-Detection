# from Commonality import *
import networkx as nx

from Community import *
from Utils import PRINT_LOG
import os.path

n = 10
z = 4
m = 1
p_in = 10
p_out = 20
l = 5

g, ground_truth = create_lattice(n, z, m, p_in, p_out, l)
# print(ground_truth)
# g.name = 'Test'
# executions_folder = os.path.dirname(__file__) + '/../Executions'
# commonality = Commonality.load_from_graph(g, executions_folder, False)
# commonality.get_communities()

community = Community()
c = []
rand_index = []
for i in range(1):
    # print(i)
    # for n in g.nodes:
#        print(n, list(g.neighbors(n)))
    communities = community.get_communities(g)
    c.append(communities)
    # for cc in communities:
#        print(cc)
    rand_index.append(adjusted_rand_index(ground_truth, communities, len(g.nodes)))
 #   print('adjusted_rand_index ', adjusted_rand_index(ground_truth, communities, len(g.nodes)))
    randomize_lattice(g, 10, 10, 5)
  #  print('---------')

for i in range(len(rand_index)):
    print(rand_index[i])
    print(c[i])
