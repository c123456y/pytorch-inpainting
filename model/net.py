import torch
import torch.nn as nn
import torch.nn.functional as F
from base.basemodel import BaseModel
from base.basenet import BaseNet
from model.loss import Perceptual, Style, advloss, focal_loss
from model.layer import init_weights, ConfidenceDrivenMaskLayer, SpectralNorm, GatedConv2d
from model.convblock import DUC, MiBlock
from model.selayer import FE, ConvDown, Mix, ConvUp, SELayer
import numpy as np


# generative convolutional neural net

class InpaintGenerator(BaseNet):
    def __init__(self, in_channels, cnum=32):
        super(InpaintGenerator, self).__init__()
        ch = cnum
        self.se = SELayer(512, 16)
        self.fe = FE(512)
        self.fuse = ConvDown(1536, 512, 1, 1)
        self.mix1 = Mix(m=-1)
        self.mix2 = Mix(m=-0.6)
        self.down_256 = ConvDown(64, 128, 4, 2, padding=1, layers=3)
        self.down_128 = ConvDown(128, 256, 4, 2, padding=1, layers=2)
        self.down_64 = ConvDown(256, 256, 4, 2, padding=1)
        self.up_256 = ConvUp(512, 64, 1, 1)
        self.up_128 = ConvUp(512, 128, 1, 1)
        self.up_64 = ConvUp(512, 256, 1, 1)

        # network structure
        self.encoder1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            GatedConv2d(in_channels, ch * 2, kernel_size=7, stride=1, padding=0, dilation=1, batch_norm=True,
                        activation=nn.ReLU(inplace=True)),
        )
        self.encoder2 = nn.Sequential(
            GatedConv2d(ch * 2, ch * 4, kernel_size=4, stride=2, padding=1, dilation=1, batch_norm=True,
                        activation=nn.ReLU(inplace=True)),
        )
        self.encoder3 = nn.Sequential(
            GatedConv2d(ch * 4, ch * 8, kernel_size=4, stride=2, padding=1, dilation=1, batch_norm=True,
                        activation=nn.ReLU(inplace=True))
        )
        self.middle = nn.Sequential(
            MiBlock(ch * 8),
            MiBlock(ch * 8),
            MiBlock(ch * 8),
            MiBlock(ch * 8),
            MiBlock(ch * 8),
            MiBlock(ch * 8),
            MiBlock(ch * 8),
            MiBlock(ch * 8)
        )
        self.decoder1 = nn.Sequential(
            DUC(ch * 8, ch * 4 * 4),
            nn.ReLU(True)
        )
        self.decoder2 = nn.Sequential(
            DUC(ch * 4, ch * 2 * 4),
            nn.ReLU(True)
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(ch * 4, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, mask):
        x_1 = self.encoder1(x)
        x_2 = self.encoder2(x_1)
        x_3 = self.encoder3(x_2)

        x_f1 = self.down_256(x_1)
        x_f2 = self.down_128(x_2)
        x_f3 = self.down_64(x_3)
        x_cat = torch.cat([x_f1, x_f2, x_f3], 1)
        x_cat_fuse = self.fuse(x_cat)
        x_fe = self.se(x_cat_fuse)
        x_fed3 = self.up_64(x_fe, (64, 64)) + x_3
        x_fed2 = self.up_128(x_fe, (128, 128)) + x_2
        x_fed1 = self.up_256(x_fe, (256, 256)) + x_1

        x_middle = self.middle(x_3)

        x_out_mix = self.mix1(x_fed3, x_middle)
        x_up1 = self.decoder1(x_out_mix)
        x_up1_mix = self.mix2(x_fed2, x_up1)
        x_up2 = self.decoder2(x_up1_mix)
        x_up2 = torch.cat([x_fed1, x_up2], 1)
        x_out = self.decoder3(x_up2)

        x_out = torch.tanh(x_out)
        return x_out


# return one dimensional output indicating the probability of realness or fakeness

class Discriminator(BaseNet):
    def __init__(self, in_channels, cnum=32):
        super(Discriminator, self).__init__()
        ch = cnum
        self.conv = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, ch*2, kernel_size=4, padding=1, stride=2)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(ch * 2, ch * 4, kernel_size=4, padding=1, stride=2)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(ch * 4, ch * 8, kernel_size=4, padding=1, stride=2)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(ch * 8, ch * 16, kernel_size=4, padding=1, stride=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch * 16, 1, 4, stride=1, padding=1)
        )

    def forward(self, x):
        feat = self.conv(x)
        return feat


from util.utils import generate_mask


class InpaintingModel(BaseModel):
    def __init__(self, in_channels, act=F.elu, norm=None, opt=None):
        super(InpaintingModel, self).__init__()
        self.opt = opt
        self.init(opt)

        self.confidence_mask_layer = ConfidenceDrivenMaskLayer()

        self.net = InpaintGenerator(in_channels, cnum=opt.g_cnum).cuda()
        init_weights(self.net)
        self.model_names = ['']
        if self.opt.phase == 'test':
            return

        self.netD = None
        self.img_size = opt.img_size
        self.gamma = opt.gamma
        self.BCEloss = nn.BCEWithLogitsLoss().cuda()
        self.G_loss_bce = None
        self.D_loss_bce = None
        self.loss_bce = 5
        self.zeros = torch.zeros((opt.batch_size, 1)).cuda()
        self.ones = torch.ones((opt.batch_size, 1)).cuda()

        self.optimizer_G = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        self.optimizer_D = None

        self.recloss = nn.L1Loss()
        self.Perceptualloss = None
        self.Styleloss = None
        self.rec_loss_func = None

        self.lambda_percep = opt.lambda_adv
        self.lambda_style = opt.lambda_rec

        self.lambda_adv = opt.lambda_adv
        self.lambda_rec = opt.lambda_rec
        self.lambda_ae = opt.lambda_ae
        self.lambda_gp = opt.lambda_gp
        self.lambda_mrf = opt.lambda_mrf
        self.adv_weight = opt.adv_weight

        self.lambda_l1 = opt.lambda_l1
        self.lambda_style = opt.lambda_style
        self.lambda_content = opt.lambda_content

        self.G_loss_l1 = None
        self.G_loss_style = None
        self.G_loss_perceptual = None

        self.G_loss = None
        self.G_loss_reconstruction = None
        self.G_loss_adv = None
        self.G_loss_ae = None
        self.D_loss = None
        self.D_hingeloss = None
        self.GAN_loss = None
        self.l1loss = None

        self.gt = None
        self.mask, self.mask_01 = None, None
        self.rect = None
        self.im_in, self.gin = None, None

        self.completed = None
        self.completed_logit = None
        self.gt_logit = None

        self.pred = None

        if self.opt.pretrain_network is False:

            self.netD = Discriminator(3, cnum=opt.d_cnum).cuda()
            init_weights(self.netD)
            self.optimizer_D = torch.optim.Adam(filter(lambda x: x.requires_grad, self.netD.parameters()), lr=opt.lr,
                                                betas=(0.5, 0.999))

            self.perceptualloss = Perceptual()
            self.styleloss = Style()
            self.adv_loss = advloss()

    def initVariables(self):
        self.gt = self.input['gt']
        mask, rect = generate_mask(self.opt.mask_type, self.opt.img_shapes, self.opt.mask_shapes)
        self.mask_01 = torch.from_numpy(mask).cuda().repeat([self.opt.batch_size, 1, 1, 1])
        self.mask = self.confidence_mask_layer(self.mask_01)
        if self.opt.mask_type == 'rect':
            self.rect = [rect[0, 0], rect[0, 1], rect[0, 2], rect[0, 3]]
            self.gt_local = self.gt[:, :, self.rect[0]:self.rect[0] + self.rect[1],
                            self.rect[2]:self.rect[2] + self.rect[3]]
        else:
            self.gt_local = self.gt
        self.im_in = self.gt * (1 - self.mask_01)
        self.gin = torch.cat((self.im_in, self.mask_01), 1)

    def forward_G(self):

        # reconstruction losses
        self.G_loss_l1 = self.recloss(self.completed * self.mask, self.gt.detach() * self.mask)
        self.G_loss_l1 = self.G_loss_l1 / torch.mean(self.mask_01)

        self.G_loss_perceptual = self.perceptualloss(self.completed * self.mask, self.gt.detach() * self.mask)
        self.G_loss_perceptual = self.G_loss_perceptual / torch.mean(self.mask_01)

        self.G_loss_style = self.styleloss(self.completed * self.mask, self.gt.detach() * self.mask)
        self.G_loss_style = self.G_loss_style / torch.mean(self.mask_01)

        self.G_loss = self.lambda_l1 * self.G_loss_l1 + self.lambda_style * self.G_loss_style + self.lambda_percep * self.G_loss_perceptual

        # discriminator
        loss_fn = torch.nn.MSELoss(reduction='mean')
        lossadv = loss_fn(self.completed * self.mask, self.gt.detach() * self.mask)
        self.G_loss = self.G_loss + self.adv_weight * lossadv

        self.D_loss, self.G_loss_adv = self.adv_loss(self.netD, self.completed, self.gt, self.mask)

        self.G_loss = self.G_loss + self.adv_weight * self.G_loss_adv


    def forward_D(self):
        self.completed_logit = self.netD(self.completed.detach())
        self.gt_logit = self.netD(self.gt)

        # hinge loss
        self.D_hingeloss = nn.ReLU()(1.0 - self.gt_logit).mean() + nn.ReLU()(1.0 + self.completed_logit).mean()
        loss_fn = torch.nn.MSELoss(reduction='mean')
        self.D_loss_adv = loss_fn(self.completed * self.mask, self.gt.detach() * self.mask)

        self.D_loss, self.G_loss_adv = self.adv_loss(self.netD, self.completed, self.gt, self.mask)
        hr = torch.sum(self.mask, dim=[1, 2, 3]) / (self.img_size * self.img_size)
        dis_seg_loss = focal_loss(self.netD, self.completed, self.mask, hr, self.gamma)

        self.D_loss = self.D_loss + self.D_hingeloss + self.D_loss_adv + dis_seg_loss

    def Dra(self, x1, x2):
        return x1 - torch.mean(x2)

    def backward_G(self):
        self.G_loss.backward(self.G_loss.clone().detach())

    def backward_D(self):
        self.D_loss.backward(retain_graph=True)

    def optimize_parameters(self):
        self.initVariables()

        self.pred = self.net(self.gin, self.mask_01)
        self.completed = self.pred * self.mask_01 + self.gt * (1 - self.mask_01)

        if self.opt.pretrain_network is False:
            for i in range(self.opt.D_max_iters):
                self.optimizer_D.zero_grad()
                self.optimizer_G.zero_grad()
                self.forward_D()
                self.backward_D()
                self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.forward_G()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_losses(self):
        l = {'G_loss': self.G_loss.item(),
             'G_loss_l1': self.G_loss_l1.item(),
             'G_loss_style': self.G_loss_style.item(),
             'G_loss_adv': self.G_loss_style.item(),
             'G_loss_perceptual': self.G_loss_perceptual.item()}
        if self.opt.pretrain_network is False:
            l.update({
                      'D_loss': self.D_loss.item(),
                      'G_loss_l1': self.G_loss_l1.item(),
                      'G_loss_style': self.G_loss_style.item(),
                      'G_loss_adv': self.G_loss_style.item(),
                      'G_loss_perceptual': self.G_loss_perceptual.item()})
        return l

    def get_current_visuals(self):
        return {'input': self.im_in.cpu().detach().numpy(), 'gt': self.gt.cpu().detach().numpy(),
                'completed': self.completed.cpu().detach().numpy()}

    def get_current_visuals_tensor(self):
        return {'input': self.im_in.cpu().detach(), 'gt': self.gt.cpu().detach(),
                'completed': self.completed.cpu().detach()}

    def evaluate(self, im_in, mask):
        im_in = torch.from_numpy(im_in).type(torch.FloatTensor).cuda() / 127.5 - 1
        mask = torch.from_numpy(mask).type(torch.FloatTensor).cuda()
        im_in = im_in * (1-mask)
        xin = torch.cat((im_in, mask), 1)

        ret = self.net(xin, mask) * mask + im_in * (1-mask)
        ret = (ret.cpu().detach().numpy() + 1) * 127.5

        return ret.astype(np.uint8)

