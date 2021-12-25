from PIL import Image

import os
import torch
import numpy as np
import networkx as nx

from scipy.stats import skew, kurtosis


def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))


def create_labels(c_org, c_dim=5, dataset="CelebA", selected_attrs=None, device="cuda"):
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.

    def label2onehot(labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    if dataset == "CelebA":
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"]:
                hair_color_indices.append(i)

    c_trg_list = []
    for i in range(c_dim):
        if dataset == "CelebA":
            c_trg = c_org.clone()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = c_trg[:, i] == 0  # Reverse attribute value.
        elif dataset == "RaFD":
            c_trg = label2onehot(torch.ones(c_org.size(0)) * i, c_dim)

        c_trg_list.append(c_trg.to(device))
    return c_trg_list


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def torch_to_pil(real, label, G):
    # 画像変換 (numpyにキャスト)
    if torch.cuda.is_available():
        fake = denorm(G(real, label)).cpu().detach().numpy().copy()
    else:
        fake = denorm(G(real, label)).detach().numpy().copy()
    # (100, 100) → (100, 100, 1)
    fake = np.squeeze(fake).copy()
    # 255で乗算 → uint8 に変換
    fake = (fake.reshape(100, 100) * 255).astype("uint8")
    # image (grey scale)
    pil_img = Image.fromarray(fake, "L")

    return pil_img


def network_info(G):
    """ネットワーク特徴量抽出"""
    degs = [d for d in dict(G.degree()).values()]

    def __average_path_len(G):
        if nx.is_connected(G):
            return nx.average_shortest_path_length(G)
        else:
            return None

    info = {
        "average": np.average(degs),
        "edges": nx.number_of_edges(G),
        "max_degree": max(degs),
        "min_degree": min(degs),
        "Skewness": skew(degs),  # 歪度
        "Kurtosis": kurtosis(degs),  # 尖度
        "average_cluster": nx.average_clustering(G),  # 平均クラスター
        "average_shortest_path": __average_path_len(G),  # 平均最短経路長
    }

    return info


def norm_min_max(array, min_num, max_num):
    """最大値を max_num、最小値を min_num にする正規化

    Parameters
    ----------
    array : numpy array
    min_num : float
        正規化後の最小値
    max_num : float
        正規化後の最大値

    Returns
    -------
    numpy array
    """
    assert max_num > min_num, "must be min_num < max_num"

    array = (array - np.min(array)) / np.max(array) - np.min(array)
    array *= max_num - min_num

    return array + min_num


def load_img_change(image_path, fake_label, G, device):
    """画像を指定ラベルに変換する"""

    # 画像の読み込み & numpy array化
    fake = np.array(Image.open(image_path))

    # 正規化 (最大 1, 最小 -1)
    fake = norm_min_max(fake, -1, 1)

    # numpy array to torch tensor
    fake = torch.tensor([[fake]], dtype=torch.float).to(device)

    """
    # 画像の変換
    if torch.cuda.is_available():
        fake = denorm(G(fake, fake_label.to(device))).cpu().detach().numpy()
    else:
        fake = denorm(G(fake, fake_label)).detach().numpy()
    """
    with torch.no_grad():
        fake = denorm(G(fake, fake_label.to(device))).cpu().detach().numpy()
    # (100, 100) → (100, 100, 1)
    fake = np.squeeze(fake).copy()
    # 255で乗算 → uint8 に変換
    fake = (fake.reshape(500, 500) * 255).astype("uint8")
    # image (grey scale)
    fake_img = Image.fromarray(fake, "L")

    return fake_img


def make_save_dir(parent_dir, network_name, parameters):
    """test 画像・グラフの保存先フォルダの作成

    Parameters
    ----------
    parent_dir : str
        保存先の親ディレクトリ
    network_name : str
        ネットワーク名
    parameters : List (str)
        ネットワークのパラメータ
        ["m_2", "m_10",,,,]

    Examples
    ----------
    >>> make_save_dir("dir/test", "BA", ["m_", "m_10"])
    dir/test/BA
    - 1_1
        - graph
            - m_2
            - m_10
            -
        - image
            - m_2
            - m_10
    - 1_0
    """

    labels = ["0_0", "0_1", "1_0", "1_1"]

    for label in labels:
        for p in parameters:
            save_path = parent_dir + "/" + network_name + "/" + label

            if not os.path.exists(save_path + "/graph/" + p):
                os.makedirs(save_path + "/graph/" + p)
            if not os.path.exists(save_path + "/image/" + p):
                os.makedirs(save_path + "/image/" + p)


def remove_noise(image_path, thres, use_tril=True):
    """test 画像のノイズを除去

    対称行列に変換し、
    要素が閾値を超えている = 1, 超えていない = 0

    Parameters
    ----------
    image_path : str
        test 画像のパス
    thres : float
        閾値
    use_tril : bool, optional
        下三角成分を使用するか、上三角成分か,
        default = 下三角成分を使用
    Returns
    -------
    numpy array
    """

    def _get_symmetric(array, use_tril):
        """対称行列を作成する関数"""
        if use_tril:
            array = np.tril(array)
        else:
            array = np.triu(array)
        return array + array.T - np.diag(array.diagonal())

    matrix = np.array(Image.open(image_path))
    matrix = _get_symmetric(matrix, use_tril)  # 対称行列にする

    # 閾値より大きいピクセル -> 1, 低いピクセル -> 0Image.open(i_path)
    array = np.where(matrix > thres, 1, 0)

    return array
