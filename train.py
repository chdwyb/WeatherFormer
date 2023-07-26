import sys
import time
import argparse
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from utils import torchPSNR, CharbonnierLoss
from WeatherFormer import WeatherFormer
from datasets import *
import torch.nn.functional as F


def train(args):

    cudnn.benchmark = True
    best_psnr = 0
    best_epoch = 0

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)

    criterion_char = CharbonnierLoss()
    if args.cuda:
        criterion_char = criterion_char.cuda()

    myNet = WeatherFormer()
    if args.cuda:
        myNet = myNet.cuda()

    # optimizer
    optimizer = optim.Adam(myNet.parameters(), lr=args.lr)
    # schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=1e-7)

    # training dataset
    path_train_input, path_train_target = args.train_data + 'input/', args.train_data + 'target/'
    datasetTrain = MyTrainDataSet(path_train_input, path_train_target, patch_size=args.patch_size)
    trainLoader = DataLoader(dataset=datasetTrain, batch_size=args.batch_size, shuffle=True,
                             drop_last=True, num_workers=4, pin_memory=True)

    # validation dataset
    path_val_input, path_val_target = args.val_data + 'input/', args.val_data + 'target/'
    datasetValue = MyValueDataSet(path_val_input, path_val_target, patch_size=args.patch_size)
    valueLoader = DataLoader(dataset=datasetValue, batch_size=args.batch_size, shuffle=True,
                             drop_last=True, num_workers=4, pin_memory=True)

    # begin training
    print('-------------------------------------------------------------------------------------------------------')
    if os.path.exists(args.resume_state):
        if args.cuda:  # CUDA
            myNet.load_state_dict(torch.load(args.resume_state))
        else:  # CPU
            myNet.load_state_dict(torch.load(args.resume_state, map_location=torch.device('cpu')))

    for epoch in range(args.epoch):
        myNet.train()
        iters = tqdm(trainLoader, file=sys.stdout)
        epochLoss = 0
        timeStart = time.time()
        for index, (x, y) in enumerate(iters, 0):

            myNet.zero_grad()
            optimizer.zero_grad()

            if args.cuda:
                input_train, target = Variable(x).cuda(), Variable(y).cuda()
            else:
                input_train, target = Variable(x), Variable(y)

            output_train, degradation_features = myNet(input_train)
            l1_loss = F.l1_loss(output_train, target)

            _, degradation_free_features = myNet(target)
            char_loss = criterion_char(degradation_features, degradation_free_features)

            loss = l1_loss + 0.04 * char_loss

            loss.backward()
            optimizer.step()
            epochLoss += loss.item()

            iters.set_description('Training !!!  Epoch %d / %d,  Batch Loss %.6f' % (epoch+1, args.epoch, loss.item()))

        if epoch % args.val_frequency == 0:
            myNet.eval()
            psnr_val_rgb = []
            for index, (x, y) in enumerate(valueLoader, 0):
                input_, target_value = (x.cuda(), y.cuda()) if args.cuda else (x, y)
                with torch.no_grad():
                    output_value, _ = myNet(input_)
                for output_value, target_value in zip(output_value, target_value):
                    psnr_val_rgb.append(torchPSNR(output_value, target_value))

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

            if psnr_val_rgb >= best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save(myNet.state_dict(), args.save_state)
        scheduler.step(epoch)
        timeEnd = time.time()
        print("------------------------------------------------------------")
        print("Epoch:  {}  Finished,  Time:  {:.4f} s,  Loss:  {:.6f}, current psnr:  {:.3f}, best psnr:  {:.3f}.".format(epoch+1, timeEnd-timeStart, epochLoss, psnr_val_rgb, best_psnr))
        print('-------------------------------------------------------------------------------------------------------')
    print("Training Process Finished ! Best Epoch : {} , Best PSNR : {:.2f}".format(best_epoch, best_psnr))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--train_data', type=str, default='./allweather/')
    parser.add_argument('--val_data', type=str, default='./raindrop_a/')
    parser.add_argument('--resume_state', type=str, default='./model_pre.pth')
    parser.add_argument('--save_state', type=str, default='./model_best.pth')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--val_frequency', type=int, default=3)
    args = parser.parse_args()

    train(args)