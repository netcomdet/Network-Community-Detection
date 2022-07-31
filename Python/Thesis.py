# from Commonality import *
from Utils import *
from Community import *

import os.path

n = 100
z = 4
m = 4
p_in = 10
p_out = 10
l = 4

g = create_lattice(n, z, m, p_in, p_out, l)

# g.name = 'Test'
# executions_folder = os.path.dirname(__file__) + '/../Executions'
# commonality = Commonality.load_from_graph(g, executions_folder, False)
# commonality.get_communities()

community = Community(g, 0.2, 0.2)
for c in community.get_communities():
    print(c)
