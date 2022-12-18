import torch
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


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
