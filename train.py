import argparse
import datetime
import json
import os
from os.path import join
from utils import find_model
import torch.utils.data as data

import torch
import torchvision.utils as vutils
from attgan import AttGAN
from data import check_attribute_conflict
from helpers import Progressbar, add_scalar_dict
from tensorboardX import SummaryWriter

output_file_path = './output'
attrs_default = [
        'Bangs', 'Mouth_Slightly_Open',
        'Eyeglasses', 'Bushy_Eyebrows'
    ]
    

def parse(args=None):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to learn')
    parser.add_argument('--data', dest='data', type=str, choices=['CelebA', 'CelebA-HQ'], default='CelebA')
    parser.add_argument('--data_path', dest='data_path', type=str, default='/home/psdz/data/img_align_celeba')
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='/home/psdz/data/list_attr_celeba.txt')
    parser.add_argument('--image_list_path', dest='image_list_path', type=str, default='data/image_list.txt')
    
    parser.add_argument('--img_size', dest='img_size', type=int, default=64)
    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1)
    parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=0)
    parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
    parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
    parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64)
    parser.add_argument('--dis_fc_dim', dest='dis_fc_dim', type=int, default=1024)
    parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=4)
    parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=4)
    parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5)
    parser.add_argument('--enc_norm', dest='enc_norm', type=str, default='batchnorm')
    parser.add_argument('--dec_norm', dest='dec_norm', type=str, default='batchnorm')
    parser.add_argument('--dis_norm', dest='dis_norm', type=str, default='instancenorm')
    parser.add_argument('--dis_fc_norm', dest='dis_fc_norm', type=str, default='none')
    parser.add_argument('--enc_acti', dest='enc_acti', type=str, default='lrelu')
    parser.add_argument('--dec_acti', dest='dec_acti', type=str, default='relu')
    parser.add_argument('--dis_acti', dest='dis_acti', type=str, default='lrelu')
    parser.add_argument('--dis_fc_acti', dest='dis_fc_acti', type=str, default='relu')
    parser.add_argument('--lambda_1', dest='lambda_1', type=float, default=100.0)
    parser.add_argument('--lambda_2', dest='lambda_2', type=float, default=10.0)
    parser.add_argument('--lambda_3', dest='lambda_3', type=float, default=1.0)
    parser.add_argument('--lambda_gp', dest='lambda_gp', type=float, default=10.0)
    
    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--epochs', dest='epochs', type=int, default=200, help='# of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0)
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--n_d', dest='n_d', type=int, default=5, help='# of d updates per g update')
    
    parser.add_argument('--b_distribution', dest='b_distribution', default='none', choices=['none', 'uniform', 'truncated_normal'])
    parser.add_argument('--thres_int', dest='thres_int', type=float, default=0.5)
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--n_samples', dest='n_samples', type=int, default=16, help='# of sample images')
    
    parser.add_argument('--save_interval', dest='save_interval', type=int, default=1000)
    parser.add_argument('--sample_interval', dest='sample_interval', type=int, default=1000)
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true')
    parser.add_argument('--experiment_name', dest='experiment_name', default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    
    return parser.parse_args(args)

args = parse()
print(args)
#with open(join('output', args.experiment_name, 'setting.txt'), 'r') as f:
    #args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))
args.lr_base = args.lr
args.n_attrs = len(args.attrs)
args.betas = (args.beta1, args.beta2)

os.makedirs(join('output', args.experiment_name), exist_ok=True)
os.makedirs(join('output', args.experiment_name, 'checkpoint'), exist_ok=True)
os.makedirs(join('output', args.experiment_name, 'sample_training'), exist_ok=True)
with open(join('output', args.experiment_name, 'setting.txt'), 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

if args.data == 'CelebA':
    from data import CelebA
    train_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'train', args.attrs)
    valid_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'valid', args.attrs)
if args.data == 'CelebA-HQ':
    from data import CelebA_HQ
    train_dataset = CelebA_HQ(args.data_path, args.attr_path, args.image_list_path, args.img_size, 'train', args.attrs)
    valid_dataset = CelebA_HQ(args.data_path, args.attr_path, args.image_list_path, args.img_size, 'valid', args.attrs)
train_dataloader = data.DataLoader(
    train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
    shuffle=True, drop_last=True
)
valid_dataloader = data.DataLoader(
    valid_dataset, batch_size=args.n_samples, num_workers=args.num_workers,
    shuffle=False, drop_last=False
)
print('Training images:', len(train_dataset), '/', 'Validating images:', len(valid_dataset))

attgan = AttGAN(args)
#attgan.load(find_model(join('output', args.experiment_name, 'checkpoint'), 'latest'))
progressbar = Progressbar()
writer = SummaryWriter(join('output', args.experiment_name, 'summary'))

it = 0
it_per_epoch = len(train_dataset) // args.batch_size
for epoch in range(args.epochs):
    # train with base lr in the first 100 epochs
    # and half the lr in the last 100 epochs
    lr = args.lr_base / (10 ** (epoch // 100))
    attgan.set_lr(lr)
    writer.add_scalar('LR/learning_rate', lr, it+1)
    for img_a, att_a in progressbar(train_dataloader):
        attgan.train()
        
        img_a = img_a.cuda() if args.gpu else img_a
        att_a = att_a.cuda() if args.gpu else att_a
        idx = torch.randperm(len(att_a))
        att_b = att_a[idx].contiguous()
        img_b = img_a[idx]
        
        att_a = att_a.type(torch.float)
        att_b = att_b.type(torch.float)
        
        att_a_ = (att_a * 2 - 1) * args.thres_int
        att_b_ = (att_b * 2 - 1) * args.thres_int
        if (it+1) % (3+1) != 0:
            errD = attgan.trainD(img_a, img_b, att_a, att_a_, att_b, att_b_)
            add_scalar_dict(writer, errD, it+1, 'D')
        else:
            errG = attgan.trainG(img_a, img_b, att_a, att_a_, att_b, att_b_)
            add_scalar_dict(writer, errG, it+1, 'G')
            progressbar.say(epoch=epoch, iter=it+1, d_loss=errD['d_loss'], g_loss=errG['g_loss'], gr_loss=errG['gr_loss'])
        
        if (it+1) % args.save_interval == 0:
            # To save storage space, I only checkpoint the weights of G.
            # If you'd like to keep weights of G, D, optim_G, optim_D,
            # please use save() instead of saveG().
            attgan.saveG(os.path.join(
                'output', args.experiment_name, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
            ))
            # attgan.save(os.path.join(
            #     'output', args.experiment_name, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
            # ))
        if (it+1) % 500 == 0:
            attgan.eval()
            with torch.no_grad():
                _, zs_a = attgan.G(img_a, mode='enc')
                _, zs_b = attgan.G(img_b, mode='enc')
                h1_a, h2_a = torch.split(zs_a, 256, dim=1)
                h1_b, h2_b = torch.split(zs_b, 256, dim=1)
                z_b, gen2_b = attgan.classify(zs_b, att_b)
                z_a, gen2_a = attgan.classify(zs_a, att_a)
        
                h_a1b2 = torch.cat([h1_a, gen2_b], dim=1)
                h_b1a2 = torch.cat([h1_b, gen2_a], dim=1)
                img_fake_a = attgan.G(h_a1b2, mode='dec')
                img_fake_b = attgan.G(h_b1a2, mode='dec')
                viz_images = torch.stack([img_a,img_b,img_fake_a], dim=1)
                viz_images = viz_images.view(-1, *list(img_a.size())[1:])
                vutils.save_image(viz_images,
                                  '%s/niter_%03d.png' % (output_file_path, it),
                                  nrow=3 * 4,
                                  normalize=True)
        it += 1
