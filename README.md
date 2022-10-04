# StableGNN
This is a component for autonomous learning of explainable graph neural networks.

### Library Highlights
It consists of three modules:
* Graph: reading input data and learning graph structure
* Model: predicting over nodes for disassortative graphs with high extrapolating ability 
* Explain: explanation of models results

Graph consists of 
* y - list of labels of all nodes in Graphs; dimension is (1,num_nodes)
* num_nodes - number of nodes in your graph
* x - attributes of dimension (num_nodes,d)
* d - dimension of attributes
* edge_index - edge list: (2,m) where m is the number of edges



### Quick Tour for New Users
First of all you need to save your raw data into folder 
```
DataValidation/dataset_name/raw
```
The data folder must contain three files: 

* **dataset_name_edges.txt** consists of two columns of nodes, each row of this table is a pair of vertices connected by an edge.
* **dataset_name_labels.txt** is a column of numbers, meaning labels of nodes. The size of this column is the size of input graph.
* **dataset_name_edge_attrs.txt** TODO


_To load data, run:_
```python
from StableGNN.Graph import Graph
import torch_geometric.transforms as T
root = 'DataValidation/'
name = dataset_name
adjust = True # flag to adjust Graph or not
data = Graph(name, root=root + str(dataset_name), transform=T.NormalizeFeatures(), adjust_flag=adjust)[0]
```
 
_To predict labels, run the model:_
TODO

_To build explanations of trained model, run:_
TODO


### Architecture Overview
**Adjusting** 

**Predicting**

**Explaining**
### Installation

