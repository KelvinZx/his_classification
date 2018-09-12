import argparse


class Config(object):
    """
    Manually setting configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_count", default=1, type=int)
    parser.add_argument("--image_per_gpu", type=int, default=64)
    parser.add_argument("--epoch", default=300, type=int)
    parser.add_argument("--backbone", default='xception', type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--out_class", default=8, type=int)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--resume", default='', type=str, metavar='PATH', help='path the latest ckpt')
    parser.add_argument("--workers", default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument("--gpu", default=None, type=str,
                        help='set gpu if no data parallel are using(default: None)')
    parser.add_argument("class_num", default=2, type=int)
    args = parser.parse_args()
    class_num = args.num_class
    resume = args.resume
    momentum = args.momentum
    backbone = args.backbone
    out_class = args.out_class
    image_per_gpu = args.image_per_gpu
    epoch = args.epoch
    lr = args.lr
    gpu_count = args.gpu_count
    workers = args.workers
    gpu = args.gpu