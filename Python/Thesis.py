from Commonality import *

graph_file_path = 'C:\\University\\Comunity Detection\\Data\\Youtube.txt'
graph_delimiter = '\t'
executions_folder = 'C:\\University\\Comunity Detection\\Executions\\'

# For loading Karate Club Graph:
# import networkx as nx
# graph = nx.karate_club_graph()
# graph.name = 'Karate_Club'
# commonality = Commonality.load_from_graph(graph, executions_folder)

commonality = Commonality.load_from_file(graph_file_path, graph_delimiter, executions_folder)
commonality.process_commonality()
commonality.save_to_file()
