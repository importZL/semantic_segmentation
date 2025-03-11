import torch
import lpips


class PerceptualLoss(torch.nn.Module):
    def __init__(self, model='net-lin', net='alex', use_gpu=True):
        super(PerceptualLoss, self).__init__()
        self.model = model
        self.net = net
        self.use_gpu = use_gpu
        if self.net == 'alex':
            self.loss_fn = lpips.LPIPS(net='alex', verbose=False)
        elif self.net == 'vgg':
            self.loss_fn = lpips.LPIPS(net='vgg', verbose=False)
        if self.use_gpu:
            self.loss_fn.cuda()

    def forward(self, x, y):
        if self.model == 'net-lin':
            if self.net == 'alex':
                return self.loss_fn.forward(x, y)
            elif self.net == 'vgg':
                return self.loss_fn.forward(x, y)
        elif self.model == 'l2':
            return torch.mean(torch.pow(x - y, 2))
        elif self.model == 'l1':
            return torch.mean(torch.abs(x - y))
        else:
            raise NotImplementedError('Loss model [%s] not implemented!' % self.model)