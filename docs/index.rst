.. StableGNN documentation master file, created by
   sphinx-quickstart on Sat Dec 10 16:13:09 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to StableGNN's documentation!
=====================================


StableGNN consists of three modules:

**Graph**: reading input data and learning graph structure
**Model**: predicting over nodes for disassortative graphs with high extrapolating ability
**Explain**: explanation of models results

**Graph** consists of

- y - list of labels of all nodes in Graphs; dimension is (1,num_nodes)
- num_nodes - number of nodes in your graph
- x - attributes of dimension (num_nodes,d)
- d - dimension of attributes
- edge_index - edge list: (2,m) where m is the number of edges

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   stable_gnn


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
