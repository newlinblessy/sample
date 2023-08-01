#!/usr/bin/env python
# coding: utf-8

# # SMA 

# In[1]:


pip install networkx


# In[6]:


import pandas as pd
import networkx as nx
import numpy as np


# In[5]:


pip install scikit-network


# In[7]:


from IPython.display import SVG 
from sknetwork.visualization import svg_graph 
from sknetwork.data import Bunch 
from sknetwork.ranking import PageRank 


# In[8]:


def draw_graph(G, show_names=False, node_size=1, font_size=10, edge_width=0.5): 

    adjacency = nx.to_scipy_sparse_matrix(G, nodelist=None, dtype=None, weight='weight', format='csr') 

    names = np.array(list(G.nodes())) 

    graph = Bunch() 
    graph.adjacency = adjacency 
    graph.names = np.array(names) 

    pagerank = PageRank() 

    scores = pagerank.fit_transform(adjacency) 

    if show_names: 
        
        image = svg_graph(graph.adjacency, font_size=font_size, node_size=node_size, names=graph.names, width=700, height=500, scores=scores, edge_width=edge_width) 

    else: 

        image = svg_graph(graph.adjacency, node_size=node_size, width=700, height=500, scores = scores, edge_width=edge_width) 

    return SVG(image)


# In[9]:


G = nx.les_miserables_graph()
df = nx.to_pandas_edgelist(G)[['source', 'target']] # cut the weights, for visualization
G = nx.from_pandas_edgelist(df)


# In[10]:


draw_graph(G, node_size=3)


# In[11]:


draw_graph(G, node_size=3, show_names=True)


# In[12]:


sorted(G.nodes)[0:10]


# In[13]:


subgraph_nodes = ['Dahlia', 'Favourite', 'Listolier', 'Fameuil', 'Zephine', 'Blacheville']

G_sub = G.subgraph(subgraph_nodes)

draw_graph(G_sub, node_size=10, font_size=12, edge_width=2, show_names=True)


# In[ ]:




