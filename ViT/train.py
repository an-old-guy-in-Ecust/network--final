import argparse
import logging
import random
import time

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import torch.nn as nn
from tqdm import tqdm
from util.misc import *
from util.eval import *
from util.augment import *
from vit_pytorch import SimpleViT

# 设置随机种子
manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# 设置设备（CPU或GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义超参数
def check_p(value: float):
    if value < 0 or value > 1:
        raise argparse.ArgumentTypeError("value must be 0-1")
    return value


parser = argparse.ArgumentParser(description='PyTorch CIFAR-100 Classifier')
parser.add_argument('--batch_size', type=int, default=128,
                    help="Every train dataset size.")
parser.add_argument('--lr', type=float, default=0.0001,
                    help="starting lr")
parser.add_argument('--epochs', type=int, default=200, help="Train loop")
parser.add_argument('--image_size', type=int,
                    default=1024, help='the size of image')
parser.add_argument('--patch_size', type=int,
                    default=16, help='the size of patch')
parser.add_argument('--num_heads', type=int, default=12,
                    help='heads for multiHeadAttention')
parser.add_argument('--num_layers', type=int, default=4,
                    help='the number of TransformerBlock')
parser.add_argument('--num_classes', type=int, default=100,
                    help='the number of classes')
parser.add_argument('--dropout_rate', type=int,
                    default=0.5, help='dropout rate')
parser.add_argument('--hidden_dim', type=int, default=768,
                    help='patch embedding dim')
parser.add_argument('--phase', type=str, default='train',
                    help="train or eval? Default:`train`")
parser.add_argument('--model_path', type=str,
                    default="./checkpoints/CIFAR100_baseline_epoch_201.pth", help="load model path.")
parser.add_argument('--augment', type=str, default='baseline',
                    help="augment method? choices: baseline, mixup, cutout, cutmix Default:`baseline`",
                    choices=['baseline', 'mixup', 'cutout', 'cutmix'])
parser.add_argument('--alpha', type=int, default=1.0,
                    help='alpha for mixup or cutmix')
parser.add_argument('--p', type=check_p, default=0.5,
                    help="probability of cutting out or cutmix")
parser.add_argument('--maskout_size', type=int, default=8,
                    help="maskout_size for cutout")
args = parser.parse_args()
# 准备数据集
train_dataloader, test_dataloader = load_dataset(args.batch_size)
# 实例化模型
model = SimpleViT(image_size=args.image_size, patch_size=args.patch_size, num_classes=args.num_classes,
                  dim=args.hidden_dim,
                  heads=args.num_heads, depth=args.num_layers, mlp_dim=4 * args.hidden_dim).to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# log地址
logging.basicConfig(filename='./log/' + args.augment + 'logging.log', filemode='a+',
                    format="%(message)s", level=logging.INFO)
writer = SummaryWriter('./' + args.augment + '_log')


def train(model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for batch_idx, data in enumerate(train_dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        # get the inputs; data is a list of [inputs, labels]
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        if args.augment == 'mixup':
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                           args.alpha)
            inputs, targets_a, targets_b = map(Variable, (inputs,
                                                          targets_a, targets_b))
            # compute output
            output = model(inputs)
            loss = mixup_criterion(
                criterion, output, targets_a, targets_b, lam)
            prec1_a, prec5_a = accuracy(output, targets_a, topk=(1, 5))
            prec1_b, prec5_b = accuracy(output, targets_b, topk=(1, 5))
            prec1, prec5 = lam * prec1_a + \
                (1 - lam) * prec1_b, lam * prec5_a + (1 - lam) * prec5_b
        elif args.augment == 'baseline':
            output = model(inputs)
            loss = criterion(output, targets)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, targets, topk=(1, 5))
        elif args.augment == 'cutout':
            CutOut = cutout(args.maskout_size, args.p, False)
            inputs = CutOut(inputs)
            output = model(inputs)
            loss = criterion(output, targets)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, targets, topk=(1, 5))
        elif args.augment == 'cutmix':
            inputs, targets_a, targets_b, lam = cutmix(
                args.alpha, args.p, inputs, targets)
            output = model(inputs)
            loss = criterion(output, targets_a) * lam + \
                criterion(output, targets_b) * (1. - lam)
            prec1, prec5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # compute gradients in a backward pass
        optimizer.zero_grad()
        loss.backward()

        # Call step of optimizer to update model params
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    logging.info(f"Epoch [{epoch + 1}] [{batch_idx}/{len(train_dataloader)}]\t"
                 f"Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                 f"Loss {losses.avg:.4f}\t")
    if epoch % 50 == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"./checkpoints/CIFAR100_{args.augment}_epoch_{epoch + 1}.pth")
    return losses.avg


def test(model):
    # switch to evaluate mode
    model.eval()
    # init value
    total = 0.
    correct = 0.
    with torch.no_grad():
        loss_fucntion = nn.CrossEntropyLoss()
        loss = 0
        for _, data in enumerate(test_dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss += loss_fucntion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    loss = loss / total
    return loss, accuracy


def train_acc(model):
    # switch to evaluate mode
    model.eval()
    # init value
    total = 0.
    correct = 0.
    with torch.no_grad():
        for _, data in enumerate(train_dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def run():
    best_prec1 = 0.
    avg_loss = []
    prec = []
    for epoch in tqdm(range(args.epochs)):
        # train for one epoch
        logging.info(f"Begin Training Epoch {epoch + 1}")
        loss = train(model, criterion, optimizer, epoch)
        avg_loss.append(loss)
        train_acc1 = train_acc(model)
        # evaluate on validation set
        logging.info(f"Begin Validation @ Epoch {epoch + 1}")
        test_loss, prec1 = test(model)
        prec.append(prec1)
        # remember best prec@1 and save checkpoint if desired
        best_prec1 = max(prec1, best_prec1)

        logging.info("Epoch Summary: ")
        logging.info(f"\tEpoch test Accuracy: {prec1}")
        logging.info(f"\tBest test Accuracy: {best_prec1}")

        logging.info(f"training acc: {train_acc1}")
        logging.info(f"test_loss: {test_loss}")
        writer.add_scalar('ViT-loss', loss, epoch)
        writer.add_scalar('Vit-test_loss', test_loss, epoch)
        writer.add_scalar('Vit-train_acc', train_acc1, epoch)
        writer.add_scalar('Vit-prec', prec1, epoch)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"./checkpoints/CIFAR100_{args.augment}_epoch_{args.epochs + 1}.pth")
    logging.info(avg_loss)
    logging.info(prec)


if __name__ == '__main__':
    if args.phase == "train":
        run()
    elif args.phase == "eval":
        if args.model_path != "":
            print("Loading model...")
            checkpoint = torch.load(
                args.model_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Loading model successful!")
            acc = test(model)
            print(
                f"Accuracy of the network on the 10000 test images: {acc:.2f}%.")
        else:
            print(
                "WARNING: You want use eval pattern, so you should add --model_path MODEL_PATH")
    else:
        print(args)
