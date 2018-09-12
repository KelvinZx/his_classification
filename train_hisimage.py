from config import Config
from imgclsdataset import GeneralDataset
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
from torch.optim import SGD
from torch.autograd import Variable
import time
import resnet
import numpy as np
import shutil
from torchvision.datasets import ImageFolder
from torchvision import transforms
import PIL
from imgaug import augmenters as iaa
import imgaug as ia
from tqdm import tqdm

MAIN_DIR = os.getcwd()
DATA_DIR = os.path.join(MAIN_DIR, 'breaKHis_patient_binary')

best_val_acc = 0
best_test_acc = 0


def adjust_learing_rate(opt, epoch):
    lr = Config.lr * (0.1 ** epoch//50) #reduce 10 percent every 50 epoch
    for param_group in opt.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.path.tar'):
    torch.save(state, os.path.join(MAIN_DIR, filename))
    if is_best:
        shutil.copyfile(filename, 'model_best.path.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    total = 0
    correct = 0
    with torch.no_grad():
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    percent_acc = 100 * correct/total
    return percent_acc


def train_epoch(data_loader, model, criterion, optimizer, epoch, print_freq=50):
    losses = AverageMeter()
    percent_acc = AverageMeter()
    model.train()
    time_now = time.time()

    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.cuda()
        target = target.cuda()
        output = model(data)
        loss = criterion(output, target)
        losses.update(loss.item(), data.size(0))

        acc = accuracy(output, target)
        percent_acc.update(acc, data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time_end = time.time() - time_now
        if batch_idx % print_freq == 0:
            print('Training Round: {}, Time: {}'.format(batch_idx, np.round(time_end, 2)))
            print('Training Loss: val:{} avg:{} Acc: val:{} avg:{}'.format(losses.val, losses.avg,
                                                                  percent_acc.val, percent_acc.avg))
    return losses, percent_acc


def validate(val_loader, model, criterion, print_freq=50):
    model.eval()
    losses = AverageMeter()
    percent_acc = AverageMeter()
    with torch.no_grad():
        time_now = time.time()
        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.cuda()
            target = target.cuda()

            output = model(data)
            loss = criterion(output, target)
            losses.update(loss.item(), data.size(0))

            acc = accuracy(output, target)
            percent_acc.update(acc, data.size(0))

            time_end = time.time() - time_now
            if batch_idx % print_freq == 0:
                print('Validation Round: {}, Time: {}'.format(batch_idx, np.round(time_end, 2)))
                print('Validation Loss: val:{} avg:{} Acc: val:{} avg:{}'.format(losses.val, losses.avg,
                                                                      percent_acc.val, percent_acc.avg))
    return losses, percent_acc


class ImageTransform:
    """
    Use imgaug library to do image augmentation.
    """
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Fliplr(0.5),
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


def main():
    cudnn.benchmark = True
    batch_size = Config.gpu_count * Config.image_per_gpu
    EPOCHS = Config.epoch
    lr = Config.lr
    workers = Config.workers
    global best_val_acc, best_test_acc
    Config.distributed = Config.gpu_count > 4 # TODO!
    model = resnet.resnet18(num_class=Config.out_class)
    #if Config.gpu is not None:
    model = model.cuda()
    if Config.gpu_count > 1:
        model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')
    test_dir = os.path.join(DATA_DIR, 'test')

    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize((256,256)),
        #ImageTransform(),
        #lambda x: PIL.Image.fromarray(x),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
         #                    std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(ImageFolder(root=train_dir, transform=TRANSFORM_IMG),
                              batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=workers)
    val_loader = DataLoader(ImageFolder(root=val_dir, transform=TRANSFORM_IMG),
                            batch_size=batch_size, shuffle=True, pin_memory=True,
                            num_workers=workers)
    test_loader = DataLoader(ImageFolder(root=test_dir, transform=TRANSFORM_IMG),
                             batch_size=batch_size, shuffle=True, pin_memory=True,
                             num_workers=workers)

    for epoch in range(EPOCHS):
        adjust_learing_rate(optimizer, epoch)
        train_losses, train_acc = train_epoch(train_loader, model, criterion, optimizer, epoch)
        val_losses, val_acc = validate(val_loader, model, criterion)
        is_best = val_acc.avg > best_val_acc
        print('>>>>>>>>>>>>>>>>>>>>>>')
        print('train loss: {}, train acc: {}, valid loss: {}, valid acc: {}'.format(train_losses.avg, train_acc.avg,
                                                                                    val_losses.avg, val_acc.avg))
        print('>>>>>>>>>>>>>>>>>>>>>>')
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_val_acc': best_val_acc,
                         'optimizer': optimizer.state_dict(),}, is_best)
    _, test_acc = validate(test_loader, model, criterion)
    print('Test accuracy: {}'.format(test_acc))


if __name__ == '__main__':
    main()
