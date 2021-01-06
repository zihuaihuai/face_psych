import jittor as jt
jt.flags.use_cuda = 1


import numpy as np
from util.Visualizer import Visualizer
import os
from PIL import Image
import time
from models.CE_Model import CE_Model
from collections import OrderedDict
from options.train_options import TrainOptions
import util.util as util
from jittor.dataset import Dataset
from jittor import transform
def make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images
class AlignedDataset(Dataset):
    def initialize(self,root,is_sketch,part_sketch):
        super().__init__(num_workers=0)
        self.part = {'bg': (0, 0, 512),
                     'eye1': (108, 156, 128),
                     'eye2': (255, 156, 128),
                     'nose': (182, 232, 160),
                     'mouth': (169, 301, 192)}
        self.part_sketch = part_sketch
        self.root = root
        if is_sketch:
            self.dirname = root + '/Sketches'
            self.file_paths = sorted(make_dataset(self.dirname))
        else:
            self.dirname = root + '/Images'
            self.file_paths = sorted(make_dataset(self.dirname))
        

    def __getitem__(self, index):
        A_path = self.file_paths[index]
        A = Image.open(A_path)
        new_w = 512
        new_h = 512
        A = A.resize((new_w, new_h), Image.NEAREST)
        A_tensor = transform.to_tensor(A) * 255.0
        if self.part_sketch !='bg':
            loc_p = self.part[self.part_sketch]
            A_tensor = A_tensor[0, loc_p[1]:loc_p[1] + loc_p[2], loc_p[0]:loc_p[0] + loc_p[2]]
        else:
            for key_p in self.part.keys():
                if key_p != 'bg':
                    loc_p = self.part[key_p]
                    A_tensor[0, loc_p[1]:loc_p[1] + loc_p[2], loc_p[0]:loc_p[0] + loc_p[2]] = 255
        A_tensor = (A_tensor - 127.5) / 127.5
        A_tensor = np.expand_dims(A_tensor, axis=0)
        A_tensor = A_tensor.astype('float32')
        A_tensor = transform.to_tensor(A_tensor)
        return A_tensor
    def __len__(self):
        return len(self.file_paths)

opt = TrainOptions().parse()
eye1_dataset = AlignedDataset()
eye1_dataset.initialize('./Asian_Face', True, 'eye1')
eye1_dataset.set_attrs(batch_size=1, shuffle=True)
eye2_dataset = AlignedDataset()
eye2_dataset.initialize('./Asian_Face', True, 'eye2')
eye2_dataset.set_attrs(batch_size=1, shuffle=True)
mouth_dataset = AlignedDataset()
mouth_dataset.initialize('./Asian_Face', True, 'mouth')
mouth_dataset.set_attrs(batch_size=1, shuffle=True)
nose_dataset = AlignedDataset()
nose_dataset.initialize('./Asian_Face', True, 'nose')
nose_dataset.set_attrs(batch_size=1, shuffle = True)
bg_dataset = AlignedDataset()
bg_dataset.initialize('./Asian_Face', True, 'bg')
bg_dataset.set_attrs(batch_size=1, shuffle = True)
dataset_list = [eye1_dataset, eye2_dataset, mouth_dataset, nose_dataset, bg_dataset]

feature_list = ['eye1', 'eye2', 'mouth', 'nose', 'bg']

for sequence in range(0, len(feature_list)):
    dataset = dataset_list[sequence]
    feature = feature_list[sequence]
    iter_path = os.path.join('./checkpoints/CE_iter')
    continue_train = False
    if continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
    else:
        start_epoch, epoch_iter = 1, 0

    dataset_size = len(bg_dataset)
    batchSize = 1
    print('feature:\n', feature)
    print('#training images = %d' % dataset_size)

    model = CE_Model(opt, feature)

    visualizer = Visualizer(opt)
    model.initialize(opt,feature)
    encoder_optimizer, decoder_optimizer = model.encoder_optimizer, model.decoder_optimizer
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    display_freq = 100
    display_delta = total_steps % display_freq
    print_freq = 10
    print_delta = total_steps % print_freq
    save_latest_freq = 1000
    save_delta = total_steps % save_latest_freq
    niter = 100
    niter_decay = 100
    label_nc = 35
    feature_vector = []
    for epoch in range(start_epoch, 100 + 100 + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size
        # for i, data in enumerate(dataset, start=epoch_iter):
        for i, data in enumerate(dataset):
            if total_steps % print_freq == print_delta:
                iter_start_time = time.time()
            total_steps += batchSize
            epoch_iter += batchSize
            save_fake = True
            # save_fake = total_steps % display_freq == display_delta
            ############## Forward Pass ######################
            # save vector


            generated, losses = model(feature, data)
            temp_feature_vector = model.feature_vector
            feature_vector.append(temp_feature_vector)
            losses = [jt.core.ops.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.loss_names, losses))

            losses = loss_dict['Mse_Loss']
            # print(i, losses)

            # encoder_optimizer.zero_grad()
            # decoder_optimizer.zero_grad()

            # loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            # loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)
            # losses.backward()
            # zero_grad and backward are included in step
            # encoder_optimizer.backward()
            # decoder_optimizer.backward()
            encoder_optimizer.step(losses)
            decoder_optimizer.step(losses)
            ############## Display results and errors ##########
            ### print out errors
            if total_steps % print_freq == print_delta:
                errors = {k: v.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / print_freq
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)
                # call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
            ### display output images
            # if save_fake:
            #     visuals = OrderedDict([('input_label', util.tensor2label(data, opt.label_nc)),
            #                            ('synthesized_image', util.tensor2im(generated)),
            #                            ('real_image', util.tensor2im(data))])
            #     visualizer.display_current_results(visuals, epoch, total_steps)


            if total_steps % save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save(str(epoch)+'_'+str(total_steps) , feature)
                # np.savetxt(iter_path, feature_vector, delimiter=',', fmt='%d')
            if epoch_iter >= dataset_size:
                break

            # end of epoch


        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, niter + niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            # model.save('latest', feature)
            model.save('Epoch_' + str(feature), feature)
            # np.savetxt(iter_path, (epoch + 1, 0)[0], delimiter=',', fmt='%d')

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            model.update_learning_rate()
    feature_vector = np.array(feature_vector)
    # np.savetxt(iter_path + str(feature) + '.txt', feature_vector, delimiter=',', fmt='%d')