from models import utils
import argparse
import os
import torchvision
from managers.classifier_manager import AutoEncoderManager
from datasets import dataset
import torch
import torch.nn as nn
from models.autoencoders.cifar.vgg_model import SegNet_E, SegNet_D, Normalize, DeNormalize

from threadpoolctl import threadpool_limits
_thread_limit = threadpool_limits(limits=8)
print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


class CifarAutoencoderManager(AutoEncoderManager):
    def __init__(self, model_name="vgg16", exp_name="cifar-all-nopretrained", i=-1):
        checkpoint_dir = utils.concat_home_dir(os.path.join('autoencoders', 'cifar10', model_name, exp_name, 'checkpoints'))

        if model_name == "resnet18":
            self.model = torchvision.models.resnet18()
        if model_name == "vgg16":
            parser = argparse.ArgumentParser(description='vgg16 based Encoder-Decoder CIFAR')
            parser.add_argument('--latent-dim', default=128, type=int, help='latent dimension')
            args = parser.parse_args()

            self.model_E = SegNet_E(args)
            self.model_D = SegNet_D(args)

        self.model_E = nn.Sequential(Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), self.model_E)
        self.model_D = nn.Sequential(self.model_D, DeNormalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

        self.model_E = nn.DataParallel(self.model_E)
        self.model_D = nn.DataParallel(self.model_D)

        self.model_E = self.model_E.cuda()
        self.model_D = self.model_D.cuda()

        dir = [int(i.split('_')[-1][:-3]) for i in os.listdir(checkpoint_dir)]
        if len(dir) > 0:
            print(dir)
            start_epoch = sorted(dir)[i]
            check_pt = torch.load(os.path.join(checkpoint_dir, 'epoch_{}.pt'.format(start_epoch)))
            self.model_E.load_state_dict(check_pt['model_E_state_dict'])
            self.model_D.load_state_dict(check_pt['model_D_state_dict'])

        self.model = nn.Sequential(self.model_E, self.model_D)
        self.model = self.model.eval()
        self.model = self.model.cuda()

    def encode(self, image):
        code = self.model_E(image)
        if len(code.shape) < 2:
            code = code.unsqueeze(0)
        return code

    def decode(self, code):
        with torch.no_grad():
            if len(code.shape) < 2:
                code = code.unsqueeze(0)
        return self.model_D(code)

    def test_model(self):
        test_loader = dataset.CIFARData.getTestSetIterator(batch_size=200)
        super(CifarAutoencoderManager, self).test(test_loader)


def load_model(model_name="vgg16", exp_name=None):
    if exp_name is not None:
        return CifarAutoencoderManager(model_name=model_name, exp_name=exp_name)

if __name__ == '__main__':
    manager = CifarAutoencoderManager()
    manager.test_model()