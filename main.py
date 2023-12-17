# import bcrypt
import math
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

import scipy  # būtiai instaliuoti
import networkx as nx
import pandas as pd
from pyvis.network import Network
import pyvis
from utils import utils as ut

# Create a pandas DataFrame with edges (you can load your own data here)
# file_loc = r"C:\Users\lukasva\Desktop\Darbai\RRF\LIMS sistemos\Elektrėnu ligonine\uzklausa_2.xlsx"
# col_selection = ["table", "referenced_table"]
# file_loc = r"C:\Users\lukasva\Desktop\Darbai\RRF\HI\priklausomybiu_gydymai\ASIS.xlsx"
# col_selection=["table","referenced_table"]
file_loc = r"C:\Users\lukasva\Desktop\Darbai\RRF\Janos užduotis\ryšiai.xlsx"
col_selection = ["Įstaiga", "Pilnas pavadinimas"]

try:
    df = pd.read_excel(file_loc)
except ValueError as ve:
    print(ve)
    print("Pakeisk formata")
    exit()

df = df.loc[:, col_selection]
df.columns = ["source", "target"]
df = df.drop_duplicates()

dict_edges = df["source"].tolist() + df["target"].to_list()
dict_edges = list(set(dict_edges))
dict_edges = [(x, 'Orange') if x == 'VDV IS' else (x, "DodgerBlue") for x in dict_edges]
dict_edges = dict(dict_edges)


# Create a NetworkX graph from the DataFrame
G_nx = nx.from_pandas_edgelist(df, source='source', target='target')

# print_degree_dist(G_nx)

n_nodes = len(G_nx.nodes())
node_distance_coef = n_nodes * 10
node_size_coef = node_distance_coef

# Generate a layout (e.g., Fruchterman Reingold layout)
pos = nx.spring_layout(G_nx)

# Create a PyVis Network object
G_pyvis = Network()

# Add nodes and edges from the NetworkX graph object to the PyVis Network object
for node in G_nx.nodes():

    G_pyvis.add_node(node,
                     x=pos[node][0] * node_distance_coef,
                     y=pos[node][1] * node_distance_coef,
                     color=dict_edges[node])  # Multiply by 1000 to scale the positions

for edge in G_nx.edges():
    G_pyvis.add_edge(edge[0], edge[1])

for node in G_pyvis.nodes:
    node['size'] = math.log((G_nx.degree(node['id'])) ** 10) + 1
    # print(node["size"], '\t', G_nx.degree(node['id']))
    node['label'] = str(node['id'])  # Set labels to node IDs

print(np.min([n["size"] for n in G_pyvis.nodes]))
print(np.max([n["size"] for n in G_pyvis.nodes]))
print("Flatten")

list_pos = np.concatenate(list(pos.values())).ravel().tolist()
# G_pyvis.add_node(name="VDV IS",
#                  id="VDV IS",
#                  n_id=n_nodes,
#                  x=np.median(np.array(list_pos) * node_distance_coef),
#                  y=np.median(np.array(list_pos) * node_distance_coef),
#                  color="",
#                  size=10000,
#                  )

# Configure edges to be straight and have the same width
for edge in G_pyvis.edges:
    # source=edge['from']
    # target=edge['to']
    # edge['length']=G.degree(source) + G.degree(target)

    edge['smooth'] = False  # Straight edges
    # edge['width'] = 2       # Set the desired width

G_pyvis.toggle_physics(False)

# Show the PyVis Network object
G_pyvis.show("interactive_graph.html", notebook=False, )
exit()
