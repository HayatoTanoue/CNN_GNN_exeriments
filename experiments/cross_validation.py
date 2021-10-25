import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader as DL_torch
from torch_geometric.loader import DataLoader as DL_geometric

from models.set_model import CNN_base, GNN_base, D1D2_base, model_parameter_counter
from experiments.cnn_train_utils import cnn_train_val_1epoch
from experiments.gnn_train_utils import gnn_train_val_1epoch


def train(
    model,
    train_loader,
    valid_loader,
    device,
    optimizer,
    criterion,
    fold,
    trial,
    save_dir,
    config,
):
    """学習"""

    # 学習データの保存
    df = pd.DataFrame(
        columns=["fold", "epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    )

    start = time.time()  # 学習時間計測用
    for epoch in range(config.num_epoch):
        if config.model_name in ["CNN", "BrainNetCNN", "D1D2"]:
            train_loss, train_acc, val_loss, val_acc = cnn_train_val_1epoch(
                model, train_loader, valid_loader, device, optimizer, criterion
            )
        else:
            train_loss, train_acc, val_loss, val_acc = gnn_train_val_1epoch(
                model, train_loader, valid_loader, device, optimizer, criterion
            )

        df = df.append(
            pd.Series(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                name=epoch,
            )
        )

        # 10epoch ごとにモデル保存
        if epoch % 10 == 0:
            if epoch != 0:
                torch.save(
                    model.state_dict(),
                    f"{save_dir}/model_weight/fold{fold}_trial{trial}_epoch{epoch}.pth",
                )

    # save training log
    df.to_csv(f"{save_dir}/train_log/fold{fold}_trial{trial}_log.csv", index=False)
    end = time.time() - start

    print(
        f"{fold} cv , trial {trial} : time = {round(end,2)}, acc = {round(val_acc, 2)} "
    )
    return val_acc


def split_data(dataset, train_idx, valid_idx, config):
    if config.model_name in ["CNN", "BrainCNN", "Deep_D1D2"]:
        # split data
        train_loader = DL_torch(
            Subset(dataset, train_idx),
            shuffle=True,
            batch_size=config.batchsize,
        )
        valid_loader = DL_torch(
            Subset(dataset, valid_idx),
            shuffle=False,
            batch_size=config.batchsize,
        )
    elif config.model_name in ["GIN", "DGCNN", "Deepsets"]:
        train_loader = DL_geometric(
            Subset(dataset, train_idx),
            shuffle=True,
            batch_size=config.batchsize,
        )
        valid_loader = DL_geometric(
            Subset(dataset, valid_idx),
            shuffle=False,
            batch_size=config.batchsize,
        )
    return train_loader, valid_loader


def CV(num_class, dataset, save_dir, config):
    start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 交差検証
    fold = KFold(n_splits=10, shuffle=True, random_state=0)
    # 精度格納用リスト
    valid_accs = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(dataset)):
        # モデルを構築
        if config.model_name in ["Deep_D1D2"]:
            model = D1D2_base(
                config.model_name, config.resize, num_class, config.use_attention
            ).to(device)
        if config.model_name in ["CNN", "BrainCNN"]:
            model = CNN_base(config.model_name, num_class, config.resize).to(device)
        if config.model_name in ["GIN", "DGCNN", "Deepsets"]:
            model = GNN_base(config.model_name, num_class, 1).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.adam_lr)

        # split dataset
        train_loader, valid_loader = split_data(dataset, train_idx, valid_idx, config)

        final_acc = 0
        for trial in range(config.trial):
            # training
            final_acc += train(
                model,
                train_loader,
                valid_loader,
                device,
                optimizer,
                criterion,
                fold_idx,
                trial,
                save_dir,
                config,
            )
        ave_final_acc = final_acc / config.trial
        valid_accs.append(ave_final_acc)

    elapsed_time = round(time.time() - start, 2)
    ave_val_acc = round(np.average(valid_accs), 2)
    print(f"{config.name} : {elapsed_time} [sec], acc = {ave_val_acc}")

    total_param = model_parameter_counter(model)

    return valid_accs, total_param
