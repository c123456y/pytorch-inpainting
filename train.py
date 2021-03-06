import os
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from data.data import InpaintingDataset, ToTensor
from model.net import InpaintingModel
from options.train_options import TrainOptions
from util.utils import getLatest
import torch


def main():
    config = TrainOptions().parse()

    print('loading data..')
    dataset = InpaintingDataset('', '', transform=transforms.Compose([
        ToTensor()
    ]))
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, drop_last=True)
    print('data loaded..')

    print('configuring model..')
    ourModel = InpaintingModel(in_channels=4, opt=config)
    ourModel.print_networks()
    if config.load_model_dir != '':
        print('Loading pretrained model from {}'.format(config.load_model_dir))
        ourModel.load_networks(getLatest(os.path.join(config.load_model_dir, '*.pth')))
        print('Loading done.')
    ourModel = torch.nn.DataParallel(ourModel).cuda()
    print('model setting up..')
    print('training initializing..')
    writer = SummaryWriter(log_dir=config.model_folder)
    cnt = 0
    for epoch in range(config.epochs):
        for i, data in enumerate(dataloader):
            gt = data['gt'].cuda()
            # normalize to values between -1 and 1
            gt = gt / 127.5 - 1

            data_in = {'gt': gt}
            ourModel.setInput(data_in)
            ourModel.optimize_parameters()

            if (i+1) % config.viz_steps == 0:
                ret_loss = ourModel.get_current_losses()
                if config.pretrain_network is False:
                    print(
                        '[%d, %5d] G_loss: %.4f (l1: %.4f, style: %.4f, perceptual: %.4f, adv: %.4f), D_loss: %.4f'
                         % (epoch + 1, i + 1, ret_loss['G_loss'], ret_loss['G_loss_l1'], ret_loss['G_loss_style'],
                           ret_loss['G_loss_perceptual'], ret_loss['G_loss_adv'], ret_loss['D_loss']))

                    writer.add_scalar('D_loss', ret_loss['D_loss'], cnt)
                    writer.add_scalar('G_per_loss', ret_loss['G_loss_perceptual'], cnt)
                else:
                    print(
                        '[%d, %5d] G_loss: %.4f (l1: %.4f, style: %.4f, perceptual: %.4f, adv: %.4f), D_loss: %.4f'
                        % (epoch + 1, i + 1, ret_loss['G_loss'], ret_loss['G_loss_l1'], ret_loss['G_loss_style'],
                           ret_loss['G_loss_perceptual'], ret_loss['G_loss_adv'], ret_loss['D_loss']))

                writer.add_scalar('G_loss', ret_loss['G_loss'], cnt)
                writer.add_scalar('l1_loss', ret_loss['G_loss_l1'], cnt)
                writer.add_scalar('style_loss', ret_loss['G_loss_style'], cnt)
                writer.add_scalar('adv_loss', ret_loss['G_loss_adv'], cnt)

                images = ourModel.get_current_visuals_tensor()
                im_completed = vutils.make_grid(images['completed'], normalize=True, scale_each=True)
                im_input = vutils.make_grid(images['input'], normalize=True, scale_each=True)
                im_gt = vutils.make_grid(images['gt'], normalize=True, scale_each=True)
                writer.add_image('gt', im_gt, cnt)
                writer.add_image('input', im_input, cnt)
                writer.add_image('completed', im_completed, cnt)
                if (i+1) % config.train_spe == 0:
                    print('saving model ..')
                    ourModel.save_networks(epoch+1)
            cnt += 1
        ourModel.save_networks(epoch+1)
    writer.close()


if __name__ == '__main__':
    main()


