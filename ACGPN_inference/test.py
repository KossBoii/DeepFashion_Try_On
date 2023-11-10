import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
import os
import numpy as np
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import cv2
from torchmetrics.functional.image import structural_similarity_index_measure as SSIM
from torchmetrics.functional.image import peak_signal_noise_ratio as PSNR


writer = SummaryWriter('runs/G1G2')
SIZE=320
NC=14
def generate_label_plain(inputs):
    size = inputs.size()
    pred_batch = []
    for input in inputs:
        input = input.view(1, NC, 256,192)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)
    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_batch = pred_batch.view(size[0], 1, 256,192)
    return label_batch
def generate_label_color(inputs):
    label_batch = []
    for i in range(len(inputs)):
        label_batch.append(util.tensor2label(inputs[i], opt.label_nc))
    label_batch = np.array(label_batch)
    label_batch = label_batch * 2 - 1
    input_label = torch.from_numpy(label_batch)
    return input_label
def complete_compose(img,mask,label):
    label=label.cpu().numpy()
    M_f=label>0
    M_f=M_f.astype(np.int)
    M_f=torch.FloatTensor(M_f).cuda()
    masked_img=img*(1-mask)
    M_c=(1-mask.cuda())*M_f
    M_c=M_c+torch.zeros(img.shape).cuda()##broadcasting
    return masked_img,M_c,M_f
def compose(label,mask,color_mask,edge,color,noise):
    masked_label=label*(1-mask)
    masked_edge=mask*edge
    masked_color_strokes=mask*(1-color_mask)*color
    masked_noise=mask*noise
    return masked_label,masked_edge,masked_color_strokes,masked_noise
def changearm(old_label):
    label=old_label
    arm1=torch.FloatTensor((data['label'].cpu().numpy()==11).astype(np.int))
    arm2=torch.FloatTensor((data['label'].cpu().numpy()==13).astype(np.int))
    noise=torch.FloatTensor((data['label'].cpu().numpy()==7).astype(np.int))
    label=label*(1-arm1)+arm1*4
    label=label*(1-arm2)+arm2*4
    label=label*(1-noise)+noise*4
    return label

os.makedirs('sample',exist_ok=True)
opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('# Inference images = %d' % dataset_size)

model = create_model(opt)
step = 0
ssim_total = 0
psnr_total = 0

for i, data in enumerate(dataset):
    t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float))
    mask_clothes = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int))
    mask_fore = torch.FloatTensor((data['label'].cpu().numpy() > 0).astype(np.int))
    img_fore = data['image'] * mask_fore
    all_clothes_label = changearm(data['label'])

    ############## Forward Pass ######################
    (
      fake_image, 
      real_image
    ) = model(
      data['label'],        # label,  
      data['edge'],   # pre_clothes_mask,  
      img_fore,             # img_fore,  
      mask_clothes,         # clothes_mask,  
      data['color'],  # clothes,  
      all_clothes_label,    # all_clothes_label,  
      data['image'],        # real_image,  
      data['pose'],         # pose,  
      data['image'],        # grid,  
      mask_fore             # mask_fore 
    )
    
    del t_mask, mask_clothes, mask_fore, img_fore, all_clothes_label
    ssim_total += SSIM(fake_image, real_image)
    psnr_total += PSNR(fake_image, real_image)
    del fake_image, real_image

print(f'AVG SSIM score: {ssim_total/len(dataset)}')
print(f'AVG PSNR score: {psnr_total/len(dataset)}')