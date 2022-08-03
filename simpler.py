# import package

# model
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR

# dataset and transformation
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader, DistributedSampler
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP



def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--local rank ', type=int, help='for use of torch.distributed.launch')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=90)
    parser.add_argument('--batch_size', type=int, default=1024)
    # parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--vis_step', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=16)
    # parser.add_argument('--gpu_ids', nargs="+", default=['0'])
    parser.add_argument('--gpu_ids', nargs="+", default=['0', '1'])
    # parser.add_argument('--gpu_ids', nargs="+", default=['0', '1', '2'])
    # parser.add_argument('--gpu_ids', nargs="+", default=['0', '1', '2', '3'])
    parser.add_argument('--world_size', type=int, default=0)
    parser.add_argument('--port', type=int, default=2022)
    parser.add_argument('--root', type=str, default='./cifar')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='./save')
    parser.add_argument('--save_file_name', type=str, default='vgg_cifar')
    # usage : --gpu_ids 0, 1, 2, 3
    return parser


def main():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args=parser.parse_args()
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:23456',
                            world_size=args.world_size,
                            rank=args.local_rank)
    local_rank=args.local_rank

    path2data = '/data'
    # if not os.path.exists(path2data):
    #     os.mkdir(path2data)

    # load dataset
    train_ds = datasets.STL10(path2data, split='train', download=True, transform=transforms.ToTensor())
    val_ds = datasets.STL10(path2data, split='test', download=True, transform=transforms.ToTensor())

    # To normalize the dataset, calculate the mean and std
    train_meanRGB = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in train_ds]
    train_stdRGB = [np.std(x.numpy(), axis=(1, 2)) for x, _ in train_ds]

    train_meanR = np.mean([m[0] for m in train_meanRGB])
    train_meanG = np.mean([m[1] for m in train_meanRGB])
    train_meanB = np.mean([m[2] for m in train_meanRGB])
    train_stdR = np.mean([s[0] for s in train_stdRGB])
    train_stdG = np.mean([s[1] for s in train_stdRGB])
    train_stdB = np.mean([s[2] for s in train_stdRGB])

    train_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize([train_meanR, train_meanG, train_meanB], [train_stdR, train_stdG, train_stdB]),
        transforms.RandomHorizontalFlip(),
    ])

    val_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize([train_meanR, train_meanG, train_meanB], [train_stdR, train_stdG, train_stdB]),
    ])

    # apply transforamtion
    train_ds.transform = train_transformation
    val_ds.transform = val_transformation

    train_sampler = DistributedSampler(train_ds)

    # create DataLoader
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=(train_sampler is None), sampler=train_sampler)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=True)

    device = torch.device("cuda:{}".format(local_rank))
    model = resnet50()
    model = model.to(device)
    model = DDP(module=model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss().to(local_gpu_id)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer=optimizer,
                       step_size=30,
                       gamma=0.1)

    if args.start_epoch != 0:

        checkpoint = torch.load(os.path.join(args.save_path, args.save_file_name) + '.{}.pth.tar'
                                .format(args.start_epoch - 1),
                                map_location=torch.device('cuda:{}'.format(local_gpu_id)))
        model.load_state_dict(checkpoint['model_state_dict'])  # load model state dict
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # load optim state dict
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # load sched state dict
        if args.rank == 0:
            print('\nLoaded checkpoint from epoch %d.\n' % (int(args.start_epoch) - 1))

    for epoch in range(args.start_epoch, args.epoch):
        tic = time.time()
        model.train()
        train_sampler.set_epoch(epoch)
        for i, (images, labels) in enumerate(train_dl):
            images = images.to(local_gpu_id)
            labels = labels.to(local_gpu_id)
            outputs = model(images)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            for param_group in optimizer.param_groups:
                lr = param_group['lr']

            toc = time.time()

            if (i % args.vis_step == 0 or i == len(train_dl) - 1) and args.rank == 0:
                print('Epoch [{0}/{1}], Iter [{2}/{3}], Loss: {4:.4f}, LR: {5:.5f}, Time: {6:.2f}'.format(epoch,
                                                                                                          args.epoch, i,
                                                                                                          len(train_dl),
                                                                                                          loss.item(),
                                                                                                          lr,
                                                                                                          toc - tic))

        if args.rank == 0:
            # if not os.path.exists(args.save_path):
            #     os.mkdir(args.save_path)

            checkpoint = {'epoch': epoch,
                          'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'scheduler_state_dict': scheduler.state_dict()}

            torch.save(checkpoint, os.path.join(args.save_path, args.save_file_name + '.{}.pth.tar'.format(epoch)))
            print("save pth.tar {} epoch!".format(epoch))

        if args.rank == 0:
            model.eval()

            val_avg_loss = 0
            correct_top1 = 0
            correct_top5 = 0
            total = 0

            with torch.no_grad():
                for i, (images, labels) in enumerate(val_dl):
                    images = images.to(args.rank)  # [100, 3, 224, 224]
                    labels = labels.to(args.rank)  # [100]
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_avg_loss += loss.item()
                    # ------------------------------------------------------------------------------
                    # rank 1
                    _, pred = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct_top1 += (pred == labels).sum().item()

                    # ------------------------------------------------------------------------------
                    # rank 5
                    _, rank5 = outputs.topk(5, 1, True, True)
                    rank5 = rank5.t()
                    correct5 = rank5.eq(labels.view(1, -1).expand_as(rank5))

                    # ------------------------------------------------------------------------------
                    for k in range(5):  # 0, 1, 2, 3, 4, 5
                        correct_k = correct5[:k + 1].reshape(-1).float().sum(0, keepdim=True)
                    correct_top5 += correct_k.item()

            accuracy_top1 = correct_top1 / total
            accuracy_top5 = correct_top5 / total

            val_avg_loss = val_avg_loss / len(val_dl)  # make mean loss

            print("top-1 percentage :  {0:0.3f}%".format(accuracy_top1 * 100))
            print("top-5 percentage :  {0:0.3f}%".format(accuracy_top5 * 100))

    return ()


def init_process(rank, args):
    args.rank = rank
    local_gpu_id = int(args.gpu_ids[args.rank])
    torch.cuda.set_device(local_gpu_id)
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:23456',
                            world_size=args.world_size,
                            rank=args.rank)
    torch.distributed.barrier()
    print(args)
    return local_gpu_id


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", default=0, type=int)
    # args = parser.parse_args()
    # args.world_size = len(args.gpu_ids)
    # args.num_workers = len(args.gpu_ids)
    # mp.spawn(main_worker, args=(args,), nprocs=args.world_size, join=True)
    main()