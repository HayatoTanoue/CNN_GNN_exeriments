import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from glob import glob


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


# for gnn loader
