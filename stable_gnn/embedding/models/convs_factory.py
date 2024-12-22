import torch
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, MessagePassing

class ConvolutionsFactory:
    """Factory responsible for creation list of specified convolution layers"""

    def build_convolutions(
        self, conv: str, num_features: int, hidden_layer: int, out_layer: int, num_layers: int, heads: int
    ) -> torch.nn.ModuleList:
        """Produce convolutions module list based on the parameters

        :param hidden_layer: (int): The size of hidden layer (default:64)
        :param out_layer: (int): The size of output layer (default:128)
        :param num_layers: (int): Number of layers in the model (default:2)
        :param heads: (int): Number of heads in GAT conv (default:1)
        :param num_features: (int): dimensionality of features
        :param conv: (str): Either 'GCN', 'GAT' or 'SAGE' convolution
        """
        if conv == "GCN":
            return self._build_gcn(
                num_features=num_features, hidden_layer=hidden_layer, out_layer=out_layer, num_layers=num_layers
            )

        elif conv == "SAGE":
            return self._build_sage(
                num_features=num_features, hidden_layer=hidden_layer, out_layer=out_layer, num_layers=num_layers
            )

        elif conv == "GAT":
            return self._build_gat(
                num_features=num_features,
                hidden_layer=hidden_layer,
                out_layer=out_layer,
                num_layers=num_layers,
                heads=heads,
            )

        elif conv == "Custom":
            return self._build_custom(
                num_features=num_features, hidden_layer=hidden_layer, out_layer=out_layer, num_layers=num_layers
            )

        raise ValueError(f"Convolution is not supported: {conv}")

    @staticmethod
    def _build_gcn(num_features: int, hidden_layer: int, out_layer: int, num_layers: int) -> torch.nn.ModuleList:
        convs = torch.nn.ModuleList()
        if num_layers == 1:
            convs.append(GCNConv(num_features, out_layer))
        else:
            convs.append(GCNConv(num_features, hidden_layer))
            for i in range(1, num_layers - 1):
                convs.append(GCNConv(hidden_layer, hidden_layer))
            convs.append(GCNConv(hidden_layer, out_layer))
        return convs

    @staticmethod
    def _build_gat(
        num_features: int, hidden_layer: int, out_layer: int, num_layers: int, heads: int
    ) -> torch.nn.ModuleList:
        convs = torch.nn.ModuleList()
        if num_layers == 1:
            convs.append(GATConv(num_features, out_layer, heads=heads))
        else:
            convs.append(GATConv(num_features, hidden_layer, heads=heads))
            for i in range(1, num_layers - 1):
                convs.append(
                    GATConv(
                        heads * hidden_layer,
                        hidden_layer,
                        heads=heads,
                    )
                )
            convs.append(GATConv(heads * hidden_layer, out_layer, heads=heads))
        return convs

    @staticmethod
    def _build_sage(num_features: int, hidden_layer: int, out_layer: int, num_layers: int) -> torch.nn.ModuleList:
        convs = torch.nn.ModuleList()
        if num_layers == 1:
            convs.append(SAGEConv(num_features, out_layer))
        else:
            convs.append(SAGEConv(num_features, hidden_layer))
            for i in range(1, num_layers - 1):
                convs.append(SAGEConv(hidden_layer, hidden_layer))
            convs.append(SAGEConv(hidden_layer, out_layer))
        return convs

    @staticmethod
    def _build_custom(num_features: int, hidden_layer: int, out_layer: int, num_layers: int) -> torch.nn.ModuleList:
        """Build custom convolutions for handling GH-graph specific requirements"""
        convs = torch.nn.ModuleList()
        if num_layers == 1:
            convs.append(CustomConv(num_features, out_layer))
        else:
            convs.append(CustomConv(num_features, hidden_layer))
            for i in range(1, num_layers - 1):
                convs.append(CustomConv(hidden_layer, hidden_layer))
            convs.append(CustomConv(hidden_layer, out_layer))
        return convs


class CustomConv(MessagePassing):
    """Custom convolution layer for GH-graphs supporting edge type weights and fuzzy weights"""

    def __init__(self, in_channels: int, out_channels: int):
        super(CustomConv, self).__init__(aggr="add")  # Aggregation method: 'add'
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.fuzzy_activation = torch.nn.Sigmoid()

    def forward(self, x: Tensor, edge_index: EdgeIndex, edge_weight: Tensor = None) -> Tensor:
        """Forward pass of CustomConv

        :param x: (Tensor): Node features
        :param edge_index: (EdgeIndex): Edge indices
        :param edge_weight: (Tensor): Edge weights (optional)
        :return: (Tensor): Updated node features
        """
        if edge_weight is not None:
            edge_weight = self.fuzzy_activation(edge_weight)  # Apply fuzzy weight transformation
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j: Tensor, edge_weight: Tensor = None) -> Tensor:
        """Message passing logic

        :param x_j: (Tensor): Neighbor node features
        :param edge_weight: (Tensor): Edge weights
        :return: (Tensor): Weighted features
        """
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out: Tensor) -> Tensor:
        """Update node features

        :param aggr_out: (Tensor): Aggregated messages
        :return: (Tensor): Updated node features
        """
        return self.lin(aggr_out)
