# this is model for fix netP and mask only left eye, right eye, nose and skin
# original loss
import jittor as jt
from . import networks
from .Basic_Model import Basic_Model
import numpy as np

class CE_Model(Basic_Model):
    def name(self):
        return 'CE_Model'

    def init_loss_filter(self, use_mseloss):
        flags = (use_mseloss, False)
        def loss_filter(loss):
            loss = (loss, 'no use')
            return [l for (l, f) in zip(loss, flags) if f]
        return loss_filter

    def initialize(self, opt, feature):
        # self.isTrain == True
        self.name = 'CE_Model'
        Basic_Model.initialize(self, opt)
        # if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
        #     torch.backends.cudnn.benchmark = True
        self.isTrain = True
        input_nc = 1

        self.net_encoder = networks.define_part_encoder(model=feature,
                                                        input_nc=1,
                                                        norm='instance',
                                                        latent_dim=512)
        self.net_decoder = networks.define_part_decoder(model=feature,
                                                        norm='instance',
                                                        output_nc=1,
                                                        latent_dim=512)
        self.load_network(typeof='AE', network=self.net_encoder, network_label='encoder_' + feature, epoch_label='latest')
        self.load_network(typeof='AE', network=self.net_decoder, network_label='decoder_' + feature + '_image', epoch_label='latest')

        self.criterion = networks.MSELoss()

        params_encoder = self.net_encoder.parameters()
        params_decoder = self.net_decoder.parameters()

        self.encoder_optimizer = jt.nn.Adam(params_encoder, lr=opt.lr, betas=(opt.beta1, 0.999))
        self.decoder_optimizer = jt.nn.Adam(params_decoder, lr=opt.lr, betas=(opt.beta1, 0.999))
        self.feature_vector = None
        self.use_mseloss = True
        self.loss_filter = self.init_loss_filter(use_mseloss = self.use_mseloss)
        self.loss_names = self.loss_filter('Mse_Loss')

    def execute(self, feature, input_part):
        # input2vector
        feature_vector = self.net_encoder(input_part)
        self.feature_vector = feature_vector
        fake_part = self.net_decoder(feature_vector)
        print(jt.reshape(self.feature_vector,(1,512)))
        loss = self.criterion(fake_part, input_part.detach()) * 10
        loss = loss.reshape(1)
        return fake_part, self.loss_filter(loss)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        self.optimizer_G = jt.nn.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        # 0.002,25
        lr = self.old_lr - lrd
        for param_group in self.encoder_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.decoder_optimizer.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            ## store_false
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def save(self, which_epoch, feature):
        self.save_network(self.net_encoder, feature, which_epoch, self.gpu_ids)
        self.save_network(self.net_decoder, feature, which_epoch, self.gpu_ids)