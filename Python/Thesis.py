from Commonality import *
import os.path

graph_file_path = os.path.dirname(__file__) + '/../Data/Amazon.txt'
graph_delimiter = '\t'
executions_folder = os.path.dirname(__file__) + '/../Executions'

# For loading Karate Club Graph:
# import networkx as nx
# graph = nx.karate_club_graph()
# graph.name = 'Karate_Club'
# commonality = Commonality.load_from_graph(graph, executions_folder)

commonality = Commonality.load_from_file(graph_file_path, graph_delimiter, executions_folder)
commonality.process_commonality()
commonality.save_to_file()
