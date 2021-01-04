# import os
# import jittor as jt
# class Basic_Model(jt.nn.Module):
#     def name(self):
#         return 'BaseModel'
#
#     def initialize(self, opt):
#         self.opt = opt
#         self.gpu_ids = opt.gpu_ids
#         self.isTrain = opt.isTrain
#         self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
#
#     def set_input(self, input):
#         self.input = input
#
#     def forward(self):
#         pass
#
#     # used in test time, no backprop
#     def test(self):
#         pass
#
#     def get_image_paths(self):
#         pass
#
#     def optimize_parameters(self):
#         pass
#
#     def get_current_visuals(self):
#         return self.input
#
#     def get_current_errors(self):
#         return {}
#
#     def save(self, label):
#         pass
#
#     # helper saving function that can be used by subclasses
#     def save_network(self, network, network_label, epoch_label, gpu_ids):
#         save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
#         save_path = os.path.join(self.save_dir, save_filename)
#         jt.nn.Module.save(network.cpu().state_dict(), save_path)
#         # torch.save(network.cpu().state_dict(), save_path)
#         if len(gpu_ids) and jt.compiler.has_cuda:
#             network.cuda()
#
#
#     # helper loading function that can be used by subclasses
#     def load_network(self, network, network_label, epoch_label, save_dir='', save_path=''):
#         save_filename = '%s_net_%s.pkl' % (epoch_label, network_label)
#         if not save_dir:
#             save_dir = self.save_dir
#         save_path = os.path.join(save_dir, save_filename)
#         print("load_path", save_path)
#         if not os.path.isfile(save_path):
#             print('%s not exists yet!' % save_path)
#         else:
#             network.load(save_path)
#
#     def update_learning_rate(self):
#         pass

import os
import jittor as jt


class Basic_Model(jt.nn.Module):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.save_dir = opt.Save_Dir
        self.AE_save_dir = opt.AE_dir
        self.Conbine_save_dir = opt.Conbine_dir

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        jt.nn.Module.save(network, save_path)
        # torch.save(network.cpu().state_dict(), save_path)

        # 'EncoderGenerator_Res "object
        # has
        # no
        # attribute
        # 'cuda'
        # if len(gpu_ids) and jt.compiler.has_cuda:
        #     network.cuda()

    # helper loading function that can be used by subclasses
    def load_network(self, typeof, network, network_label, epoch_label, save_dir='', save_path=''):
        save_filename = '%s_net_%s.pkl' % (epoch_label, network_label)

        if not save_dir:
            if typeof == 'AE':
                save_dir = self.AE_save_dir
            else:
                save_dir = self.Conbine_save_dir
        save_path = os.path.join(save_dir, save_filename)
        print("load_path", save_path)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
        else:
            network.load(save_path)

    def update_learning_rate(self):
        pass
