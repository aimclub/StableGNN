
from torch_geometric.nn.conv import MessagePassing


class ModelName(MessagePassing):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

    def message(self):
        pass

    def VirtualVertex(self):
        pass
    def Extrapolate(self):
        pass
    def SelfSupervised(self):
        pass



    #    return (f'{self.__class__.__name__}({self.in_channels}, '
     #           f'{self.out_channels}, heads={self.heads}, '
      #          f'type={self.attention_type})')