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
col_selection_initial = ["Įstaiga", "Informacinės sistemos sutrumpintas pavadinimas", "Informacinės sistemos pilnas pavadinimas"]
col_selection = ["Įstaiga", "Informacinės sistemos sutrumpintas pavadinimas"]

df = ut.network_data(file_loc)

df_edges = df.get_edges()
df_nodes = df.get_nodes(df_edges)

dict_nodes = dict(zip(df_nodes["node_name"], df_nodes["color"]))

dict_edges = df_edges["source"].tolist() + df_edges["target"].to_list()
dict_edges = list(set(dict_edges))
dict_edges = [(x, '#F6803D') if dict_nodes[x] == 'istaiga' else (x, "#11879D") for x in dict_edges]
dict_edges = dict(dict_edges)

# Create a NetworkX graph from the DataFrame
G_nx = nx.from_pandas_edgelist(df_edges, source='source', target='target')

n_nodes = len(G_nx.nodes())
node_distance_coef = n_nodes * 9
node_size_coef = node_distance_coef

# Generate a layout (e.g., Fruchterman Reingold layout)
pos = nx.spring_layout(G_nx, k=1.4 * (1 / sqrt(n_nodes)), scale=1.5)

# Create a PyVis Network object
G_pyvis = Network(height='1500px', width='1500px', select_menu=True, cdn_resources="remote")

# Add nodes and edges from the NetworkX graph object to the PyVis Network object
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

# Add nodes and edges from the NetworkX graph object to the PyVis Network object

"""
Žemiau nurodyti koeficientai yra naudojami matematinėje funkcijojoje, kuri transformuoja 
node dydį taip, kad išryškinti skirtumus tarm mažiausiai ryšių turinčių node ir daugiausiai

Ją išsibandyti galima čia: https://www.desmos.com/calculator
Šaltinis: https://www.researchgate.net/figure/A-Properties-of-a-sigmoid-function-Although-many-functional-forms-exist-each-shares_fig1_257462730

"""

A = 100  # kelia funcijos max reikšmę išlaikant slope
B = 1.8
C = 0.17
D = 0.8
Z = 30  # minimalus dydis

ut.print_min_max_node_size(G_nx, post_transformation=False)

for node in G_pyvis.nodes:

    x = G_nx.degree(node['id'])  # n_informacines_sistemos - ryšių su informacinėmis sistemomis kiekis
    node['size'] = A * exp(-exp(B - C * x / D)) + Z
    node['label'] = str(node['id'])  # Set labels to node IDs

    tipas = dict_nodes[node['id']]

    if dict_nodes[node['id']] == 'istaiga':

        node['title'] = f"Tipas: {tipas}\n" \
                        f"Pavadinimas: {node['id']}\n" \
                        f"IS kiekis: {x}"
    else:
        node['title'] = f"Tipas: {tipas}\n" \
                        f"Pavadinimas: {node['id']}\n"

    node['id'] = f"{dict_nodes[node['id']]} | {node['id']}"

ut.print_min_max_node_size(G_pyvis, post_transformation=True)

list_pos = np.concatenate(list(pos.values())).ravel().tolist()

# Configure edges to be straight and have the same width
for edge in G_pyvis.edges:
    # Pervadinu id, pagal kuriuos braižomi edge - pridedu node tipą, kaip tai padariau ankstesniame for loop'e
    new_from = f"{dict_nodes[edge['from']]} | {edge['from']}"
    new_to = f"{dict_nodes[edge['to']]} | {edge['to']}"
    edge["from"] = new_from
    edge["to"] = new_to

    edge['smooth'] = False  # Straight edges

G_pyvis.toggle_physics(False)

# G_pyvis.show_buttons(filter_=["nodes"])

html = G_pyvis.generate_html()
with open("outputs/RRF įstaigos ir IS_2.html", mode='w', encoding='utf-8') as fp:
    fp.write(html)
display(HTML(html))
# Show the PyVis Network object
# G_pyvis.show("interactive_graph.html", notebook=False, )
