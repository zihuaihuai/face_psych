import numpy as np
import os
from util.image_pool import ImagePool
from .Basic_Model import Basic_Model
from . import networks
import jittor as jt

class IS_Model(Basic_Model):
    def name(self):
        return 'IS_Model'
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True)

        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):
            return [l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, d_real, d_fake), flags) if f]

        return loss_filter

    def initialize(self, opt):
        self.opt = opt
        Basic_Model.initialize(self, opt)
        self.save_dir = os.path.join(opt.Save_Dir, opt.name)
        # BaseModel.initialize(self, opt)
        input_nc = opt.input_nc
        self.gpu_ids = opt.gpu_ids
        self.Tensor = self.gpu_ids
        ##### define networks
        # Generator network
        self.part = {'bg': (0, 0, 512),
                     'eye1': (108, 156, 128),
                     'eye2': (255, 156, 128),
                     'nose': (182, 232, 160),
                     'mouth': (169, 301, 192)}

        self.Decoder_Part = {}
        self.Encoder_Part = {}
        for key in self.part.keys():
            self.Decoder_Part[key] = networks.define_feature_decoder(model=key,
                                                                output_nc = 32, norm=opt.norm,
                                                                latent_dim = opt.latant_dim)

            self.Encoder_Part[key] = networks.define_part_encoder(model=key,
                                                        input_nc=1,
                                                        norm='instance',
                                                        latent_dim= 512)
            if not key == 'nose':
                self.load_network(typeof='AE', network=self.Encoder_Part[key], network_label='encoder_' + key, epoch_label='latest')
                self.load_network(typeof='IS', network=self.Decoder_Part[key], network_label='DE_' + key, epoch_label='latest')
        opt.ngf = 56
        self.netG = networks.define_G(opt.num_inter_channels, opt.output_nc, opt.ngf,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.norm)
        # self.load_network(typeof='IS', network=self.netG, network_label='G', epoch_label=opt.which_epoch)
        #netD_input_nc = opt.output_nc
        netD_input_nc = 35
        self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid=False,
                                      num_D = opt.num_D, getIntermFeat = not opt.no_ganFeat_loss)
    # set loss functions and optimizers
        self.fake_pool = ImagePool(opt.pool_size)
        self.old_lr = opt.lr

        # define loss functions
        self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)

        self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
        self.criterionFeat = jt.nn.L1Loss()
        if not opt.no_vgg_loss:
            self.criterionVGG = networks.VGGLoss(self.gpu_ids)

        # Names so we can breakout loss
        self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake')
        self.fdcoder_params = {}
        for key in self.part.keys():
            self.fdcoder_params = list(self.Decoder_Part[key].parameters())
        self.G_params = self.netG.parameters()
        self.optimizer_G = jt.nn.Adam(self.G_params, lr=opt.lr, betas=(opt.beta1, 0.999))
        self.D_params = self.netD.parameters()
        self.optimizer_D = jt.nn.Adam(self.D_params, lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optim_Decoder = {}
        self.optim_Decoder_params = {}
        for key in self.Decoder_Part.keys():
            self.optim_Decoder_params[key] = self.Decoder_Part[key].parameters()
            self.optim_Decoder[key] = jt.nn.Adam(self.optim_Decoder_params[key], lr=opt.lr, betas=(opt.beta1, 0.999))

    def discriminate(self, generate_label, test_image, use_pool=False):
        test_image = test_image / test_image.max()
        # print('test_image:\n')
        # print('mean:\n{}\n'.format(jt.mean(test_image)), 'sum:\n{}\n'.format(test_image.sum()), 'std:\n{}\n'.format(jt.std(test_image)))
        generate_label = generate_label / generate_label.max()
        # print('generate_label\n')
        # print('mean:\n{}\n'.format(jt.mean(generate_label)), 'sum:\n{}\n'.format(generate_label.sum()),
        #       'std:\n{}\n'.format(jt.std(generate_label)))
        input_concat = jt.contrib.concat((generate_label, test_image), dim=1)
        # print('input_concat\n')
        # print('mean:\n{}\n'.format(jt.mean(input_concat)), 'sum:\n{}\n'.format(input_concat.sum()), 'std:\n{}\n'.format(jt.std(input_concat)))
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.execute(fake_query)
        else:
            return self.netD.execute(input_concat)

    def encode_input(self, eye1, eye2, mouth, nose, bg):
        nose = self.Encoder_Part['nose'](nose)
        eye1 = self.Encoder_Part['eye1'](eye1)
        eye2 = self.Encoder_Part['eye2'](eye2)
        mouth = self.Encoder_Part['mouth'](mouth)
        bg = self.Encoder_Part['bg'](bg)
        return eye1, eye2, mouth, nose, bg

    def execute(self, eye1, eye2, mouth, nose, bg, real_image, infer = False):

        eye1, eye2, mouth, nose, bg = self.encode_input(eye1, eye2, mouth, nose, bg)

        eye1_r_feature = self.Decoder_Part['eye1'](eye1)
        eye2_r_feature = self.Decoder_Part['eye2'](eye2)
        nose_r_feature = self.Decoder_Part['nose'](nose)
        mouth_r_feature = self.Decoder_Part['mouth'](mouth)
        bg_r_feature = self.Decoder_Part['bg'](bg)
        bg_r_feature[:, :, 301:301 + 192, 169:169 + 192] = mouth_r_feature
        bg_r_feature[:, :, 232:232 + 160 - 36, 182:182 + 160] = nose_r_feature[:, :, :-36, :]
        bg_r_feature[:, :, 156:156 + 128, 108:108 + 128] = eye1_r_feature
        bg_r_feature[:, :, 156:156 + 128, 255:255 + 128] = eye2_r_feature

        input_concat = bg_r_feature

        fake_image_raw = self.netG.execute(input_concat)
        fake_image_vgg = fake_image_raw
        if infer:
            fakes = fake_image_raw[0, :, :, :].detach().numpy() # TODO：change the numpy cost too much time.
            fakes = (np.transpose(fakes, (1, 2, 0)) + 1) / 2.0 * 255.0
            fakes = np.clip(fakes, 0, 255)
            fake_image = fakes.astype(np.uint8)

        pred_fake_pool = self.discriminate(input_concat, fake_image_raw, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)
        # print('loss_D_fake:')
        # GAN feature matching loss
        pred_real = self.discriminate(input_concat, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)
        # print('loss_D_real:')
        # print(loss_D_real)
        # GAN loss (Fake     Loss)
        pred_fake = self.discriminate(input_concat, fake_image_raw, use_pool=False)
        # pred_fake = self.netD.execute(jt.contrib.concat((input_concat, fake_image), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        # print('pred_fake:')
        # print('mean:\n{}\n'.format(jt.mean(pred_fake)), 'sum:\n{}\n'.format(pred_fake.sum()),
        #       'std:\n{}\n'.format(jt.std(pred_fake)))
        # print('losses are as followed:\n',loss_D_fake.numpy(), loss_D_real.numpy(), loss_G_GAN.numpy(), loss_G_GAN_Feat, loss_G_VGG)
        # self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        # GAN L1 matching loss
        # VGG feature matching loss
        feat_weights = 4.0 / (self.opt.n_layers_D + 1)
        loss_G_GAN_Feat = 0
        D_weights = 1.0 / self.opt.num_D

        for i in range(self.opt.num_D):
            for j in range(len(pred_fake[i]) - 1):
                # print(pred_fake[i][j],pred_real[i][j])
                loss_G_GAN_Feat += D_weights * feat_weights * \
                                   self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
        loss_G_VGG = 0
        # print(self.opt.no_vgg_loss) False
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image_vgg, real_image) * self.opt.lambda_feat

        # print('losses are as followed:\n',loss_D_fake.numpy(), loss_D_real.numpy(), loss_G_GAN.numpy(), loss_G_GAN_Feat, loss_G_VGG)
        # self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        # Only return the fake_B image if necessary to save BW
        return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake),
                None if not infer else fake_image]

    # def load_network(self, network, network_label, epoch_label, save_dir='', save_path=''):
    #     save_filename = '%s_net_%s.pkl' % (epoch_label, network_label)
    #     if not save_dir:
    #         save_dir = self.save_dir
    #     save_path = os.path.join(save_dir, save_filename)
    #     print("load_path",save_path)
    #     if not os.path.isfile(save_path):
    #         print('%s not exists yet!' % save_path)
    #     else:
    #         network.load(save_path)

    def save(self, which_epoch):
        for key in self.Decoder_Part.keys():
            self.save_network(self.Decoder_Part[key], key, which_epoch, self.gpu_ids)
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        # if self.gen_features:
        #     self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        params = list(self.netG.parameters())
        self.optimizer_G = jt.nn.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        params = list(self.netD.parameters())
        self.optimizer_D = jt.nn.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr



