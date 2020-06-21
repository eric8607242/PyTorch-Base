import time
import logging
import argparse

import torch
import torch.nn as nn

from utils.config import get_config
from utils.util import get_writer, get_logger, set_random_seed, cal_model_efficient
from utils.dataflow import get_transforms, get_dataset, get_dataloader
from utils.optim import get_optimizer, get_lr_scheduler
from utils.trainer import Trainer
from utils.model import Model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="path to the config file", required=True)
    args = parser.parse_args()

    CONFIG = get_config(args.cfg)

    if CONFIG.cuda:
        device = torch.device("cuda" if (torch.cuda.is_available() and CONFIG.ngpu > 0) else "cpu")
    else:
        device = torch.device("cpu")

    set_random_seed(CONFIG.seed)

    get_logger(CONFIG.log_dir)
    writer = get_writer(CONFIG.write_dir)

    train_transform, val_transform, test_transform = get_transforms(CONFIG)
    train_dataset, val_dataset, test_dataset = get_dataset(train_transform, val_transform, test_transform, CONFIG)
    train_loader, val_loader, test_loader = get_dataloader(train_dataset, val_dataset, test_dataset, CONFIG)

    lookup_table = LookUpTable(CONFIG)

    criterion = cross_encropy_with_label_smoothing

    layers_config = lookup_table.decode_arch_param(arch_param)
    model = Model(layers_config, CONFIG.dataset, CONFIG.classes)    

    if CONFIG.model_pretrained is not None:
        model.load_state_dict(torch.load(CONFIG.model_pretrained)["model"])

    model = model.to(device)
    if (device.type == "cuda" and CONFIG.ngpu >= 1):
        model = nn.DataParallel(model, list(range(CONFIG.ngpu)))

    cal_model_efficient(model, CONFIG)

    optimizer = get_optimizer(model, CONFIG.optim_state)
    scheduler = get_lr_scheduler(optimizer, len(train_loader), CONFIG)

    start_time = time.time()
    trainer = Trainer(criterion, optimizer, scheduler, writer, device, CONFIG)
    test_top1_avg = trainer.validate(model, test_loader, 0)
    logging.info("Total training time : {:.2f}".format(time.time() - start_time))
    

