import os
# import bcrypt
from math import log, exp, sqrt
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
from IPython.display import display, HTML

file_loc = r"data\Node_edge_data.xlsx"
col_selection = {"source_col": ["Įstaiga"],
                 "target_cols": ["Informacinės sistemos sutrumpintas pavadinimas",
                                 "Informacinės sistemos pilnas pavadinimas"]}

df = ut.network_data(file_loc, col_selection)

df_edges = df.get_edges()
df_nodes = df.get_nodes(df_edges)

dict_nodes = dict(zip(df_nodes["node_name"], df_nodes["color"]))

dict_edges = df_edges["source"].tolist() + df_edges["target"].to_list()
dict_edges = list(set(dict_edges))
dict_edges = [(x, '#F6803D') if dict_nodes[x] == 'institution' else (x, "#11879D") for x in dict_edges]
dict_edges = dict(dict_edges)

# Creating a NetworkX graph from the DataFrame
G_nx = nx.from_pandas_edgelist(df_edges, source='source', target='target')

n_nodes = len(G_nx.nodes())
node_distance_coef = n_nodes * 9
node_size_coef = node_distance_coef

# Generating a layout
pos = nx.spring_layout(G_nx, k=1.4 * (1 / sqrt(n_nodes)), scale=1.5)

# Creating a PyVis Network object
G_pyvis = Network(height='1500px', width='1500px', select_menu=True, cdn_resources="remote")

# Adding nodes and edges from the NetworkX graph object to the PyVis Network object
for node in G_nx.nodes():
    G_pyvis.add_node(node,
                     x=pos[node][0] * node_distance_coef, y=pos[node][1] * node_distance_coef,
                     # Multiply by 1000 to scale the positions
                     color=dict_edges[node],
                     labelHighlightBold=True,
                     scaling={"label": {"min": 8, "max": 20}}
                     )

for edge in G_nx.edges():
    G_pyvis.add_edge(edge[0], edge[1], color="#F6803D")


"""
The following coeficients used for sigmoidal function to define the size of the node. Institutions with small count of
information systems are made tiny compared to those that have a lot of connections.

# Source: https://www.researchgate.net/figure/A-Properties-of-a-sigmoid-function-Although-many-functional-forms-exist-each-shares_fig1_257462730
# Tested on: https://www.desmos.com/calculator

"""

# Add nodes and edges from the NetworkX graph object to the PyVis Network object
A = 100
B = 1.8
C = 0.17
D = 0.8
Z = 30

# ut.print_min_max_node_size(G_nx, post_transformation=False)

for node in G_pyvis.nodes:

    n_IS = G_nx.degree(node['id'])  # amount of information systems (IS) that institution is connected with
    node['size'] = A * exp(-exp(B - C * n_IS / D)) + Z

    node['label'] = str(node['id'])  # Set labels to node IDs

    node_type = dict_nodes[node['id']]  # marking the nodes type: Institution or (IS)

    if dict_nodes[node['id']] == 'institution':

        node['title'] = f"Type: {node_type}\n" \
                        f"Name: {node['id']}\n" \
                        f"IS count: {n_IS}"
    else:
        node['title'] = f"Type: {node_type}\n" \
                        f"Name: {node['id']}\n"

    node['id'] = f"{node_type} | {node['id']}"

ut.print_min_max_node_size(G_pyvis, post_transformation=True)

list_pos = np.concatenate(list(pos.values())).ravel().tolist()

# Defining edges
for edge in G_pyvis.edges:
    # defining edge id according to source and target nodes ("node_type | node")
    source_node_type = dict_nodes[edge['from']]
    target_node_type = dict_nodes[edge['to']]

    # defining edges
    new_from = f"{source_node_type} | {edge['from']}"
    new_to = f"{target_node_type} | {edge['to']}"

    edge["from"] = new_from
    edge["to"] = new_to

    edge['smooth'] = False  # Straight edges

G_pyvis.toggle_physics(False)

html = G_pyvis.generate_html()

with open("RRF_institutions_n_information_systems.html", mode='w+', encoding='utf-8') as fp:
    fp.write(html)

