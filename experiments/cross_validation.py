import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader

from models.set_model import CNN_base, GNN_base, model_parameter_counter
from experiments.cnn_train_utils import cnn_train_val_1epoch


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
        if config.model_name in ["CNN", "BrainNetCNN"]:
            train_loss, train_acc, val_loss, val_acc = cnn_train_val_1epoch(
                model, train_loader, valid_loader, device, optimizer, criterion
            )
        else:
            train_loss, train_acc, val_loss, val_acc = cnn_train_val_1epoch(
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
            torch.save(
                model.state_dict(),
                f"{save_dir}/model_weight/fold{fold}_trial{trial}_epoch{epoch}.pth",
            )

    # save training log
    df.to_csv(f"{save_dir}/train_log/fold{fold}_trial{trial}_log.csv", index=False)
    end = time.time() - start
    print(f"{fold} cv , trial {trial} : time = {end}, acc = {np.average}")
    return val_acc


def CV(num_class, dataset, save_dir, config):
    start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 交差検証
    fold = KFold(n_splits=10, shuffle=True, random_state=0)
    # 精度格納用リスト
    valid_accs = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(dataset)):
        # モデルを構築
        if config.model_name in ["CNN", "BrainNetCNN"]:
            model = CNN_base(config.model_name, num_class, config.input_size)
        else:
            model = GNN_base(config.model_name, 4, 1)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.adam_lr)

        # split data
        train_loader = DataLoader(
            Subset(dataset, train_idx),
            shuffle=True,
            batch_size=config.batchsize,
        )
        valid_loader = DataLoader(
            Subset(dataset, valid_idx),
            shuffle=False,
            batch_size=config.batchsize,
        )

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
        ave_final_acc = final_acc / 3

    elapsed_time = time.time() - start
    print(
        "{} : {} [sec], acc = {}".format(
            config.name, elapsed_time, np.average(valid_accs)
        )
    )

    total_param = model_parameter_counter(model)

    return valid_accs, total_param
