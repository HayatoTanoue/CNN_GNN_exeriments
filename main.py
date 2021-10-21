from data.data_loader import cnn_data_loader_cv, gnn_data_loader_cv
from models.set_model import CNN_base
from experiments.cross_validation import CV

import argparse
import os


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


def make_savedir(config):
    os.makedirs(
        f"logs/{config.name}/{config.model_name}/{config.sort}_{config.resize}_{config.adam_lr}",
        exist_ok=True,
    )
    save_dir = f"logs/{config.name}/{config.model_name}/{config.sort}_{config.resize}_{config.adam_lr}"

    os.makedirs(save_dir + "/train_log", exist_ok=True)
    os.makedirs(save_dir + "/model_weight", exist_ok=True)

    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--resize", type=int, default=100)
    parser.add_argument("--sort", type=str, default="shuffle")
    parser.add_argument("--name", type=str, default="MUTAG")
    # model setting
    parser.add_argument("--classifier", type=str, default="simple")
    parser.add_argument("--out_feature", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="CNN")

    # train params
    parser.add_argument("--adam_lr", type=float, default=0.01)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--batchsize", type=int, default=32)

    config = parser.parse_args()

    # load dataset
    if config.name in ["GIN", "DGCNN", "Deepsets"]:
        dataset, num_class = gnn_data_loader_cv(config.name)
    else:
        dataset, num_class = cnn_data_loader_cv(config.name, config.sort, config.resize)

    # make save dir
    save_dir = make_savedir(config)
    # train and validation
    CV(num_class, dataset, save_dir, config)
