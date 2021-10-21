import torch
import networkx as nx
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from glob import glob
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
    path = f"graph_data/{name}/graph"
    num_class = len(glob(path) + "/*")
    data_list = []
    for i, file in enumerate(sorted(glob(path))):
        # adjlist のパスを取得
        paths = glob(file + "/*.adjlist")
        data_list += [data_from_adjlist(p, i) for p in paths]
    return data_list, num_class


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
