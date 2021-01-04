import numpy as np
import cv2
from PIL import Image
import time
from util.Visualizer import Visualizer
import jittor as jt
from collections import OrderedDict
from options.train_options import TrainOptions
import util.util as util
from models.IS_Model import IS_Model
from jittor.dataset import Dataset
import jittor_core
import os
jt.flags.use_cuda = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
print(jittor_core.get_device_count())
def make_dataset(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images
class AlignedDataset(Dataset):
    def initialize(self, root):
        self.part = {'bg': (0, 0, 512),
                     'eye1': (108, 156, 128),
                     'eye2': (255, 156, 128),
                     'nose': (182, 232, 160),
                     'mouth': (169, 301, 192)}
        self.root = root
        self.image_dirname = root + '/Images'
        self.sketch_dirname = root + '/Sketches'
        self.sketch_file_paths = sorted(make_dataset(self.sketch_dirname))
        self.image_file_paths = sorted(make_dataset(self.image_dirname))

    def __getitem__(self, index):
        Tensor = {}
        new_w = 512
        new_h = 512

        Image_Path = self.image_file_paths[index]
        Image_Self = Image.open(Image_Path)
        Image_Tensor = Image_Self.resize((new_w, new_h), Image.NEAREST)
        Image_Tensor = jt.transform.to_tensor(Image_Tensor)*255.0

        Sketch_Path = self.sketch_file_paths[index]
        Sketch = Image.open(Sketch_Path)
        Sketch_Tensor = Sketch.resize((new_w, new_h), Image.NEAREST)
        Sketch_Tensor = jt.transform.to_tensor(Sketch_Tensor)*255.0
        # Temp_Tensor = Temp_Tensor.astype('float32')
        # Temp_Tensor = (Temp_Tensor - 127.5) / 127.5
        for key in self.part.keys():
            if key != 'bg':
                loc_p = self.part[key]
                Tensor[key] = Sketch_Tensor[0, loc_p[1]:loc_p[1] + loc_p[2], loc_p[0]:loc_p[0] + loc_p[2]]
                Tensor[key] = (Tensor[key]- 127.5) / 127.5
                Tensor[key] = np.expand_dims(Tensor[key], axis=0)
                Tensor[key] = Tensor[key].astype('float32')
                Tensor[key] = jt.transform.to_tensor(jt.array(Tensor[key]))
            else:
                temp = Sketch_Tensor[0]
                for key_p in self.part.keys():
                    if key_p != 'bg':
                        loc_p = self.part[key_p]
                        temp[loc_p[1]:loc_p[1] + loc_p[2], loc_p[0]:loc_p[0] + loc_p[2]] = 255
                # Tensor[key] = jt.transform.to_tensor(jt.array(temp))
                Tensor[key] = temp
                Tensor[key] = (Tensor[key]- 127.5) / 127.5
                Tensor[key] = np.expand_dims(Tensor[key], axis=0)
                Tensor[key] = Tensor[key].astype('float32')
                Tensor[key] = jt.transform.to_tensor(jt.array(Tensor[key]))

        Tensor['image'] = Image_Tensor/255
        return Tensor['eye1'], Tensor['eye2'], Tensor['nose'], Tensor['mouth'], Tensor['bg'], Tensor['image']
    def __len__(self):
        return len(self.sketch_file_paths)

opt = TrainOptions().parse()
all_dataset = AlignedDataset()
all_dataset.initialize('./Asian_Face')
all_dataset = all_dataset.set_attrs(batch_size=1, shuffle=True)
dataset_size = len(all_dataset)
feature_list = ['eye1', 'eye2', 'mouth', 'nose', 'bg', 'G']

iter_path = os.path.join('./checkpoints_dir/iter.txt')
continue_train = False
if continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0
batchSize = 1
print('#training images = %d' % dataset_size)
model = IS_Model(opt)
model.initialize(opt)
visualizer = Visualizer(opt)
optimizer_G, optimizer_D, optimizer_Decoder = model.optimizer_G, model.optimizer_D, model.optim_Decoder
total_steps = (start_epoch - 1) * dataset_size + epoch_iter
display_freq = 100
display_delta = total_steps % display_freq
print_freq = 20
print_delta = total_steps % print_freq
save_latest_freq = 1000
save_delta = total_steps % save_latest_freq
niter = 100
niter_decay = 100
label_nc = 35
fm_freq = 5

model.train()
for epoch in range(start_epoch, 100 + 100 + 1):
    epoch_start_time = time.time()
    print('Start of epoch %d / %d' % (epoch, niter + niter_decay))
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, (eye1, eye2, nose, mouth, bg, image) in enumerate(all_dataset):
        if total_steps % print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += batchSize
        epoch_iter += batchSize
        save_fake = total_steps % display_freq == display_delta
        ############## Forward Pass ######################

        losses, generated = model(eye1, eye2, mouth, nose, bg, image, save_fake)
        losses = [jittor_core.Var.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(model.loss_names, losses))

        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)

        ############## Back Propagation ######################
        # loss_G.sync()
        optimizer_G.step(loss_G)
        # loss_D.sync()
        optimizer_D.step(loss_D)
        if i % fm_freq == 0:
            for key in optimizer_Decoder.keys():
                optimizer_Decoder[key].step(loss_G)
        ############## Display results and errors ##########
        ### print out errors
        if total_steps % print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
            # call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
        ### display output images
            if save_fake:
                visuals = OrderedDict([('synthesized_image', cv2.cvtColor(generated, cv2.COLOR_BGR2RGB)),
                                       ('real_image', util.tensor2im(image))])
                cv2.imwrite('trainimage/synthesized_image_e' + str(epoch) + '_i'+str(i) + '.jpg', visuals["synthesized_image"])
                cv2.imwrite('trainimage/real_image_e' + str(epoch) + '_i' + str(i) + '.jpg', visuals["real_image"])
                visualizer.display_current_results(visuals, epoch, total_steps)
            if total_steps % save_latest_freq == save_delta:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save(epoch)
                np.savetxt(iter_path, (epoch, epoch_iter)[0], delimiter=',', fmt='%d')
            if epoch_iter >= dataset_size:
                break
        # end of epoch
        iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, niter + niter_decay, time.time() - epoch_start_time))
    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.save(epoch)
        np.savetxt(iter_path, (epoch + 1, 0)[0], delimiter=',', fmt='%d')

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.update_learning_rate()