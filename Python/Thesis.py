from Commonality import *
from Community import *
import os.path

n = 10
z = 4
m = 1
p_in = 0
p_out = 0
l = 0

g = create_lattice(n, z, m, p_in, p_out, l)

g.name = 'Test'
executions_folder = os.path.dirname(__file__) + '/../Executions'
commonality = Commonality.load_from_graph(g, executions_folder, False)
commonality.get_communities()
