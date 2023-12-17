from matplotlib import pyplot as plt
import pandas as pd
import numpy as np



class network_data:
    def __init__(self, abs_path_to_file, col_selection):
        self.abs_path_to_file = abs_path_to_file
        self.col_selection=col_selection



    def get_edges(self):
        """
        From the xlsx file in data folder, this function takes info about institutions and their information systems,
        cleans it, transforms it into edge table
        :return: pd.Dataframe, edge table
        """

        # data import
        df_edges = pd.read_excel(self.abs_path_to_file,
                                 )
        # column selection
        cols_source_target = self.col_selection["source_col"] + self.col_selection["target_cols"]
        df_edges = df_edges.loc[:, cols_source_target]

        # cleaning values from white space characters
        df_edges = df_edges.replace(to_replace=r"^(\s|\\n)|(\s|\\n)$|", value="", regex=True)

        # Cleaning strings that are shorter than 2 symbols
        for column in df_edges.columns:
            df_edges[column] = df_edges[column].apply(lambda x: np.nan if len(str(x)) < 2 else x)

        # Getting single target column by Coalescing who columns - name of informatin system, full name of information system
        target_cols = self.col_selection["target_cols"]
        df_edges.loc[:, "target"] = df_edges.loc[:, target_cols].bfill(
            axis=1).iloc[:, 0]

        df_edges = df_edges.loc[:, ["Įstaiga", "target"]]
        df_edges.columns = ["source", "target"]

        # additional cleaning
        df_edges = df_edges.loc[df_edges["source"] != df_edges['target'], :]
        df_edges = df_edges.drop_duplicates()
        df_edges = df_edges.dropna(how='any', axis=0)

        return df_edges

    def get_nodes(self, df_edges):
        """
        Getting nodes table from edge table and defining the nodes

        :param df_edges:
        :return: pd.Dataframe, node table
        """
        df_nodes = df_edges.loc[:, ["source", "target"]]
        df_nodes['color'] = ""

        # Creating node_table:
            # For source and target columns:
                # select "source" or "target" column,
                # rename it into node_name,
                # assigning value for color "institution" or "information system"

        df_nodes_source = df_nodes.loc[:, ["source", "color"]]
        df_nodes_source.rename(columns={"source": "node_name"}, inplace=True)
        df_nodes_source["color"] = 'institution'

        df_nodes_target = df_nodes.loc[:, ["target", "color"]]
        df_nodes_target.rename(columns={"target": "node_name"}, inplace=True)
        df_nodes_target["color"] = 'information_system'

        # Concating both node tables and dropping the duplicates
        df_nodes_concated = pd.concat([
            df_nodes_source, df_nodes_target
        ])
        df_nodes_concated = df_nodes_concated.drop_duplicates(subset='node_name')

        return df_nodes_concated

def print_min_max_node_size(G_opject, post_transformation):
    if post_transformation==False:
        print(
            "Min/max size BEFORE transformation"
        )
        min_size=min([x[1] for x in G_opject.degree(G_opject.nodes)])
        max_size=max([x[1] for x in G_opject.degree(G_opject.nodes)])
        print(min_size)
        print(max_size)
    elif post_transformation==True:

        print("Min/max size BEFORE transformation")
        min_size = min([x["size"] for x in G_opject.nodes])
        max_size = max([x["size"] for x in G_opject.nodes])
        print(min_size)
        print(max_size)
    else:
        print("Wrong post_tansformation value - must be bool")

def print_degree_dist(G_nx):
    list_of_nodes=G_nx.nodes()
    list_of_degrees=[G_nx.degree(x) for x in list_of_nodes]
    node_degrees=dict(zip(list_of_nodes, list_of_degrees))
    node_degrees={k:v for (k,v) in node_degrees.items() if v >9 and v<=20}
    print(node_degrees)

    filtered_nodes=node_degrees.keys()
    print(' '.join(filtered_nodes))
    print(len(filtered_nodes))

    plt.hist(node_degrees.values(),bins=30)
    plt.show()
