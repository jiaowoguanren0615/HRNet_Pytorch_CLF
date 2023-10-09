import argparse
import os
import pprint
import shutil
import sys

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from my_dataset import MyDataset
import models.cls_hrnet
from config.default import _C as config
from config.default import update_config
from config.models import MODEL_EXTRAS
from core.function import train
from core.function import validate
from utils.modelsummary import get_model_summary
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import Plot_ROC
from utils.split_data import read_split_data



def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='./experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
                        type=str)

    parser.add_argument('--data_root',
                        help='the root path of your dataset',
                        default='/usr/local/Huangshuqi/ImageData/flower_data',
                        type=str)

    parser.add_argument('--num_classes',
                        help='num_classes of your dataset',
                        default=5,
                        type=int)

    parser.add_argument('--resume',
                        help='resume',
                        default=False,
                        type=bool)

    parser.add_argument('--scalar',
                        help='auto mixture precision',
                        default=True,
                        type=bool)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')

    args = parser.parse_args()
    update_config(config, args)

    return args



def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    model = eval('models.' + config.MODEL.NAME + '.get_cls_net')(
        config)

    dump_input = torch.rand(
        (1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model, dump_input))

    # copy model file
    this_dir = os.path.dirname(__file__)
    models_dst_dir = os.path.join(final_output_dir, 'models')
    if os.path.exists(models_dst_dir):
        shutil.rmtree(models_dst_dir)
    shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(config, model)

    best_perf = 0.0
    last_epoch = config.TRAIN.BEGIN_EPOCH

    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
            last_epoch - 1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
            last_epoch - 1
        )

    checkpoint_save_path = os.path.join(final_output_dir, 'checkpoint.pth')

    if args.resume:
        model_state_file = checkpoint_save_path
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            model.load_state_dict(checkpoint['state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            

            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))


    # Data loading code
    train_image_path, train_image_label, val_image_path, val_image_label, class_indices = read_split_data(args.data_root)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        'valid': transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ])
    }

    train_set = MyDataset(train_image_path, train_image_label, data_transforms['train'])
    valid_set = MyDataset(val_image_path, val_image_label, data_transforms['valid'])

    train_loader = DataLoader(
        train_set,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    scalar = torch.cuda.amp.GradScaler() if args.scalar else None

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()
        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict, device, scalar)
        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, model, criterion,
                                  final_output_dir, tb_log_dir, device, writer_dict)

        if perf_indicator > best_perf:
            best_perf = perf_indicator

            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_parameters = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }

            torch.save(save_parameters, checkpoint_save_path)

    writer_dict['writer'].close()

    # net, val_loader, save_name, device
    Plot_ROC(model, valid_loader, checkpoint_save_path, device)

if __name__ == '__main__':
    main()