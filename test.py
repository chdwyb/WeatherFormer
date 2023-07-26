import sys
import time
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from WeatherFormer import WeatherFormer
from datasets import *
from utils import pad, unpad


def test(args):

    myNet = WeatherFormer()
    if args.cuda:
        myNet = myNet.cuda()

    # testing dataset
    datasetTest = MyTestDataSet(args.input_dir)
    testLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, drop_last=False,
                            num_workers=4, pin_memory=True)

    print('--------------------------------------------------------------')
    if args.cuda:
        myNet.load_state_dict(torch.load(args.resume_state))
    else:
        myNet.load_state_dict(torch.load(args.resume_state, map_location=torch.device('cpu')))
    myNet.eval()

    with torch.no_grad():
        timeStart = time.time()
        for index, (x, name) in enumerate(tqdm(testLoader, desc='Testing !!! ', file=sys.stdout), 0):
            torch.cuda.empty_cache()

            input_test = x.cuda() if args.cuda else x

            input_test, pad_size = pad(input_test, factor=16)
            output_test, _ = myNet(input_test)
            output_test = unpad(output_test, pad_size)

            save_image(output_test, args.result_dir + name[0])
        timeEnd = time.time()
        print('---------------------------------------------------------')
        print("Testing Process Finished !!! Time: {:.4f} s".format(timeEnd - timeStart))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./Test1/input/')
    parser.add_argument('--result_dir', type=str, default='./Test1/result/')
    parser.add_argument('--resume_state', type=str, default='./model_best.pth')
    parser.add_argument('--cuda', type=bool, default=True)
    args = parser.parse_args()

    test(args)
