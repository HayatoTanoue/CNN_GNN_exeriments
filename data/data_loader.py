import torch
import networkx as nx
from torch_geometric.data.data import Data
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from glob import glob
from tqdm import tqdm
from torch_geometric.utils import from_networkx


def set_transform(resize):
    """画像前処理設定"""
    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]
    )
    return transform


# for cnn loader
def cnn_data_loader_cv(name, sort, resize):
    if name in ["MUTAG", "DD", "COLLAB", "REDDIT-BINARY"]:
        """ベンチマークデータセット"""
        path = f"graph_data/TUDataset/{name}/{sort}/"
        num_class = len(glob(f"graph_data/TUDataset/{name}/{sort}/*"))
    else:
        """合成データセット"""
        path = f"graph_data/{name}/{sort}/pics"
        num_class = 4

    transforms = set_transform(resize)
    dataset = ImageFolder(path, transforms)

    return dataset, num_class


########################################################################
# for gnn loader
def gnn_data_loader_cv(name):
    path = f"graph_data/{name}/graph_tensor"
    data_list = []
    for i in range(20000):
        data = Data(
            x=torch.load(f"{path}/x/{i}.pt"),
            y=torch.load(f"{path}/y/{i}.pt"),
            edge_index=torch.load(f"{path}/edge/{i}.pt"),
        )
        data_list.append(data)

    return data_list, 4


# adjlist -> data (torch geometric)
def data_from_adjlist(path, label):
    """adjlist to torch geometric data"""
    # load adjlist
    G = nx.read_adjlist(path)
    # graph to torch geometric data
    data = from_networkx(G)

    # set node attribute
    data.x = torch.tensor(
        [[d] for d in list(dict(G.degree()).values())],
        dtype=torch.float,
    )

    # set graph label
    data.y = torch.tensor([label])
    return data
