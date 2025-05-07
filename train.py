import argparse
import time
import datetime
import os
import shutil
import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torchvision.transforms as transforms

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.loss import get_segmentation_loss
from core.utils.distributed import *
from core.utils.logger import setup_logger
from core.utils.lr_scheduler import WarmupPolyLR
from core.utils.score import SegmentationMetric

from torchvision.transforms.functional import to_pil_image
from PIL import Image




specific_image_indices = [10,20,30,40,50]

save_predictions_folder = '/home/swz45/Documents/awesome-semantic-segmentation-pytorch/awesome-semantic-segmentation-pytorch-master/test_miao'

if not os.path.exists(save_predictions_folder):
   os.mkdir(save_predictions_folder)

augmentations = [
        transforms.RandomRotation(degrees=30),  # Rotate by up to ±30 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness & contrast
        transforms.ElasticTransform(alpha=50.0)  # Elastic deformation
    ]






class_colors = {
    "0": (0, 0, 0),
    "1": (255, 0, 0),
    "2": (0, 255, 0),
    "3": (0, 0, 255),
    "4": (255, 255, 0),
    #"5": (128, 0, 128),
    "5": (0, 255, 255),
    "6": (255, 165, 0),
    #"8": (255, 192, 203)
}

'''
class_colors = {
    0: (0, 0, 0),         
    1: (0, 255, 0),       
    2: (255, 255, 0),     
    3: (255, 0, 0), 
}
'''




def apply_class_colors(mask, class_colors):
    """
    Convert a single-channel mask with class indices into a color image.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, color in class_colors.items():
        color_mask[mask == int(class_idx)] = color

    return color_mask



def getweight_balance():

    
    
    habitat_classes = {
          0: 'back',
          #1: 'Bedrock',
          1: 'Shadow',
          2: 'Fine',
          #4: 'Bank',
          #5: 'Woody',
          3: 'Boulder',
          4: 'Rocky_Fine',
          #6: 'Manmade'
    }
    
    
    '''
    habitat_classes = {
          0: 'back',
          1: 'Shadow',
          2: 'Bank',
          3: 'Otherclass'
    }
    '''
    
    '''
    file_list_path = '/home/swz45/Documents/awesome-semantic-segmentation-pytorch/awesome-semantic-segmentation-pytorch-master/datasets/ShadowBank/spliting/train.txt' 
    with open(file_list_path, 'r') as file:
         file_names = file.read().splitlines()
    '''
    
    pixel_counts = {class_id: 0 for class_id in habitat_classes.keys()}
   
    #msk_dir = '/home/swz45/Documents/Shiqi/river-project/awesome-semantic-segmentation-pytorch/datasets/Lamine_river/cropped_data/combine_msk'
    
    for file_name in os.listdir('/home/swz45/Documents/Shadow_correction/4x4_Cropped/full_resolution/augmented_masks'):
         
        #parts = file_name.split('_')  # e.g., ["one", "two", "three", "four"]

        # Combine everything except the last element
        #combined = '_'.join(parts[:-1]) + '_binarymask.png'

        file_name1 = '/home/swz45/Documents/Shadow_correction/4x4_Cropped/full_resolution/augmented_masks/' + file_name
        mask = cv2.imread(file_name1, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # Only count if class_id >= 1
            for class_id in habitat_classes.keys():
                if class_id >= 1:
                   pixel_counts[class_id] += np.sum(mask == class_id)
       
    counts = [pixel_counts[class_id] for class_id in habitat_classes.keys()]
    print(counts)
    
    total_sum = sum(counts)
    
    print(counts / total_sum)
    
    weights = 1/(counts / total_sum)
    
    print(weights)
    
    weights_list = weights / sum(weights)
    
    
    return list(weights)
    
early_step = 10
    

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='fcn',
                        choices=['fcn32s', 'fcn16s', 'fcn8s', 'fcn', 'psp', 'deeplabv3', 
                            'deeplabv3_plus', 'danet', 'denseaspp', 'bisenet', 'encnet', 
                            'dunet', 'icnet', 'enet', 'ocnet', 'psanet', 'cgnet', 'espnet', 
                            'lednet', 'dfanet','swnet'],
                        help='model name (default: fcn32s)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['vgg16', 'resnet18', 'resnet50', 'resnet101', 'resnet152', 
                            'densenet121', 'densenet161', 'densenet169', 'densenet201'],
                        help='backbone name (default: vgg16)')
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        choices=['pascal_voc', 'pascal_aug', 'ade20k', 'citys', 'sbu', 'substrate'],
                        help='dataset name (default: pascal_voc)')
    parser.add_argument('--base-size', type=int, default=520,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
    parser.add_argument('--workers', '-j', type=int, default=0,
                        metavar='N', help='dataloader threads')
    # training hyper params
    parser.add_argument('--jpu', action='store_true', default=False,
                        help='JPU')
    parser.add_argument('--use-ohem', type=bool, default=False,
                        help='OHEM Loss for cityscapes dataset')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--warmup-iters', type=int, default=0,
                        help='warmup iters')
    parser.add_argument('--warmup-factor', type=float, default=1.0 / 3,
                        help='lr = warmup_factor * lr')
    parser.add_argument('--warmup-method', type=str, default='linear',
                        help='method of warmup')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='~/.torch/models',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='../runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'pascal_aug': 80,
            'pascal_voc': 50,
            'pcontext': 80,
            'ade20k': 160,
            'citys': 120,
            'sbu': 160,
        }
        args.epochs = epoches[args.dataset.lower()]
    if args.lr is None:
        lrs = {
            'coco': 0.004,
            'pascal_aug': 0.001,
            'pascal_voc': 0.0001,
            'pcontext': 0.001,
            'ade20k': 0.01,
            'citys': 0.01,
            'sbu': 0.001,
            'substrate': 0.0001
        }
        args.lr = lrs[args.dataset.lower()] / 8 * args.batch_size
    return args


class Trainer(object):
    def modify_model_for_grayscale(self):
         # Modify the first convolutional layer of the pretrained model to accept 1-channel input
        
        
        first_conv_layer = self.model.pretrained[0]  # Accessing the first Conv2d directly
        if isinstance(first_conv_layer, nn.Conv2d) and first_conv_layer.in_channels == 3:
            new_conv_layer = nn.Conv2d(1, first_conv_layer.out_channels, kernel_size=first_conv_layer.kernel_size,
                                       stride=first_conv_layer.stride, padding=first_conv_layer.padding,
                                       dilation=first_conv_layer.dilation, groups=first_conv_layer.groups,
                                       bias=first_conv_layer.bias is not None)
            # Copy weights from the original layer, summing across the input channels
            new_conv_layer.weight.data = torch.mean(first_conv_layer.weight.data, dim=1, keepdim=True)
            
            with torch.no_grad():
                 new_conv_layer.weight.data = torch.mean(first_conv_layer.weight.data, dim=1, keepdim=True)
                 if first_conv_layer.bias is not None:
                    new_conv_layer.bias.data = first_conv_layer.bias.data.clone()
            
            
            new_conv_layer = new_conv_layer.to(self.device)
            self.model.pretrained[0] = new_conv_layer
        
    
    def adjust_learning_rate(self, optimizer, epoch):
        """
        Adjusts the learning rate according to the schedule:
        - 0.001 from 0 to 30 epochs
        - 0.0005 from 30 to 60 epochs
        - 0.00025 from 60 to 80 epochs
        """
        if epoch < 30:
           lr = 0.0005
        elif 30 <= epoch < 60:
           lr = 0.0005
        elif 60 <= epoch <= 80:
           lr = 0.0005
        else:
           lr = 0  # Optionally set to zero for epochs beyond 80

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr    






    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            #augmentations,
            transforms.ToTensor(),
            #transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
            transforms.Normalize([.485], [.229]),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
        train_dataset = get_segmentation_dataset(args.dataset, split='train', mode='train', **data_kwargs)
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='val', **data_kwargs)
        args.iters_per_epoch = len(train_dataset) // (args.num_gpus * args.batch_size)
        args.max_iters = args.epochs * args.iters_per_epoch

        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iters)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, 2)


        

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                            aux=args.aux, jpu=args.jpu, norm_layer=BatchNorm2d).to(self.device)
                                            
                                            
        #self.modify_model_for_grayscale()
        print(self.model)

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # create criterion
        #new_weights = getweight_balance()
        #print(new_weights)
        #class_weights = torch.tensor(new_weights, dtype=torch.float32)
        #print(class_weights)
        #self.criterion = get_segmentation_loss(args.model, use_ohem=args.use_ohem, aux=args.aux,
        #                                       aux_weight=class_weights, ignore_index=0).to(self.device)
        #self.criterion = nn.CrossEntropyLoss(ignore_index=0).to(self.device)
        #class_weights = torch.tensor([1.0, 0.04])





        '''
        shadow_class=2,  # Your shadow class index
        alpha=0.25,      # Weight for shadow class
        gamma=2.0,       # Moderate focus on hard examples
        aux=True,
        aux_weight=0.2,
        ignore_index=0 


        '''

        #self.criterion = get_segmentation_loss(args.model, use_ohem=args.use_ohem, aux=args.aux,
        #                                       aux_weight=args.aux_weight, ignore_index=0).to(self.device)
        


        self.criterion = get_segmentation_loss(args.model, use_ohem=args.use_ohem, recall_alpha=0.4, aux=False, aux_weight=0.4, ignore_index=0).to(self.device)  # Background class

        #recall_alpha=0.4

        # optimizer, for model just includes pretrained, head and auxlayer
        params_list = list()
        if hasattr(self.model, 'pretrained'):
            params_list.append({'params': self.model.pretrained.parameters(), 'lr': args.lr})
        if hasattr(self.model, 'exclusive'):
            for module in self.model.exclusive:
                params_list.append({'params': getattr(self.model, module).parameters(), 'lr': args.lr * 10})
        self.optimizer = torch.optim.SGD(params_list,
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        # lr scheduling
        self.lr_scheduler = WarmupPolyLR(self.optimizer,
                                         max_iters=args.max_iters,
                                         power=0.9,
                                         warmup_factor=args.warmup_factor,
                                         warmup_iters=args.warmup_iters,
                                         warmup_method=args.warmup_method)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                             output_device=args.local_rank)

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)

        self.best_pred = 0.0

    def train(self):
        save_to_disk = get_rank() == 0
        epochs, max_iters = self.args.epochs, self.args.max_iters
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.args.iters_per_epoch
        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        start_time = time.time()
        logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))
        
        miou_records = []
        accuracy_records = []
        valid_accuracy_records = []
        valid_miou_records = []
        train_acc = []
        train_miou = []
        
        
        

        self.model.train()
        for iteration, (images, targets, _) in enumerate(self.train_loader):
            try:
              iteration = iteration + 1
              self.lr_scheduler.step()

              images = images.to(self.device)
              targets = targets.to(self.device)

              try:
                
                outputs = self.model(images)
                self.metric.update(outputs[0], targets)
                pixAcc, mIoU, singleacc = self.metric.get()

                train_acc.append(pixAcc)
                train_miou.append(mIoU)

                loss_dict = self.criterion(outputs, targets)
                losses = sum(loss for loss in loss_dict.values())

                # Reduce losses over all GPUs for logging purposes
                loss_dict_reduced = reduce_loss_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
              except Exception as e:
                logger.error(f"Error during forward pass or loss computation: {e}")
                continue

              try:
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
              except Exception as e:
                logger.error(f"Error during backward pass or optimizer step: {e}")
                continue

              eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
              eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

              if iteration % log_per_iters == 0 and save_to_disk:
                 logger.info(
                   "Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                    iteration, max_iters, self.optimizer.param_groups[0]['lr'], losses_reduced.item(),
                    str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))

              if iteration % save_per_iters == 0 and save_to_disk:
                 try:
                   save_checkpoint(self.model, self.args, is_best=False)
                 except Exception as e:
                   logger.error(f"Error saving checkpoint: {e}")

              if not self.args.skip_val and iteration % val_per_iters == 0:
                 try:
                   valid_accuracy, valid_miou = self.validation()
                   valid_accuracy_records.append(valid_accuracy)
                   valid_miou_records.append(valid_miou)
                   miou_records.append(train_miou[-1])
                   accuracy_records.append(train_acc[-1])
                   #train_miou = 0
                   #train_acc = 0
                   self.model.train()
                 except Exception as e:
                   logger.error(f"Error during validation: {e}")
                   continue

                 current_epoch = iteration // self.args.iters_per_epoch + 1
                 #self.adjust_learning_rate(self.optimizer, current_epoch)
              '''
              if iteration % self.args.iters_per_epoch == 0 and current_epoch % 2 == 0:
                 try:
                  # Create folder for the current epoch
                   epoch_folder = os.path.join(save_predictions_folder, f"epoch_{current_epoch}")
                   os.makedirs(epoch_folder, exist_ok=True)

                   accuracy_record1 = []  # List to record accuracies for specific images

                   for idx in specific_image_indices:
                      if idx < len(self.train_loader.dataset):
                        try:
                            # Get the specific image and ground truth
                            image, target, _ = self.train_loader.dataset[idx]

                            print(image.shape)

                            # Save the input image
                            input_image_path = os.path.join(epoch_folder, f"image_{idx}_input.png")
                            #to_pil_image(image.squeeze().cpu()).save(input_image_path)
                            to_pil_image(image.squeeze()).save(input_image_path)

                            # Apply color to the ground truth
                            ground_truth_colored = apply_class_colors(target.cpu().numpy(), class_colors)
                            ground_truth_path = os.path.join(epoch_folder, f"image_{idx}_ground_truth.png")
                            Image.fromarray(ground_truth_colored).save(ground_truth_path)

                            # Run the model to get predictions
                            image = image.unsqueeze(0).to(self.device)  # Add batch dimension
                            output = self.model(image)

                            # Convert the output prediction to a color image
                            prediction = output[0].argmax(dim=1).squeeze().cpu().numpy()
                            target_np = target.cpu().numpy()
                            prediction[target_np == 0] = 0

                            prediction_colored = apply_class_colors(prediction, class_colors)
                            prediction_path = os.path.join(epoch_folder, f"image_{idx}_prediction.png")
                            Image.fromarray(prediction_colored).save(prediction_path)

                            # Calculate and record accuracy for the specific image
                            self.metric.update(output[0], target.unsqueeze(0).to(self.device))
                            pixAcc, _, singleacc = self.metric.get()
                            accuracy_record1.append((idx, singleacc))
                        except Exception as e:
                            logger.error(f"Error processing specific image {idx}: {e}")

                # Save the accuracy records to a file
                   accuracy_file = os.path.join(epoch_folder, "accuracy_records.txt")
                   with open(accuracy_file, "w") as f:
                       f.write("Image Index\tAccuracy\n")
                       for record in accuracy_record1:
                           f.write(f"{record[0]}\t{record[1]:.4f}\n")
                 except Exception as e:
                   logger.error(f"Error during epoch folder creation or processing: {e}")
            '''
            except Exception as e:
              logger.error(f"Unexpected error in iteration {iteration}: {e}")
            
            

        save_checkpoint(self.model, self.args, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))
        plt.figure(figsize=(6, 6))

        # Plot for Mean Intersection over Union (mIOU)
    	#plt.subplot(1, 2, 1)
        plt.plot(range(len(miou_records)), miou_records, label='Training mIOU', marker='o', linestyle='-', color='blue')
        plt.plot(range(len(valid_miou_records)), valid_miou_records, label='Validation mIOU', marker='x', linestyle='--', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('mIOU')
        plt.title('mIOU over Epochs')
        plt.legend()
        plt.tight_layout()
        plt.savefig('/home/swz45/Documents/awesome-semantic-segmentation-pytorch/awesome-semantic-segmentation-pytorch-master/matrix_linegraph/deeplab_1x1_nobankwoody_miou.png', dpi=300)
        plt.close()
        
        plt.figure(figsize=(6, 6))
        plt.plot(range(len(accuracy_records)), accuracy_records, label='Training Accuracy', marker='o', linestyle='-', color='green')
        plt.plot(range(len(valid_accuracy_records)), valid_accuracy_records, label='Validation Accuracy', marker='x', linestyle='--', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Pixel Accuracy over Epochs')
        plt.legend()
        plt.tight_layout()
        plt.savefig('/home/swz45/Documents/awesome-semantic-segmentation-pytorch/awesome-semantic-segmentation-pytorch-master/matrix_linegraph/deeplab_accuracy_nobankwoody.png', dpi=300)
        plt.close()
        
        
        
        
    def validation(self):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False
        self.metric.reset()
        
        valid_pix_accuracy = []
        valid_miou = []
        
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()

        counting = 0
        for i, (image, target, filename) in enumerate(self.val_loader):
            counting += 1
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)
            self.metric.update(outputs[0], target)
            pixAcc, mIoU, singleacc = self.metric.get()
            valid_pix_accuracy.append(pixAcc) 
            valid_miou.append(mIoU) 
        
            #logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))


        logger.info("Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(valid_pix_accuracy[-1], valid_miou[-1]))

        #new_pred = (pixAcc + mIoU) / 2
        if valid_pix_accuracy[-1] > self.best_pred:
            is_best = True
            self.best_pred = valid_pix_accuracy[-1]
        save_checkpoint(self.model, self.args, is_best)
        synchronize()
        return valid_pix_accuracy[-1], valid_miou[-1]

def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}_{}_weight_bal_deeplab_accuracy_1024x1024_nobankwoody.pth'.format(args.model, args.backbone, args.dataset)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = '{}_{}_{}_weight_bal_best_deeplab_accuracy_1x1_1024x1024_nobankwoody.pth'.format(args.model, args.backbone, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()

    # reference maskrcnn-benchmark
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    print('number of gpus:')
    print(args,num_gpus)
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    args.lr = args.lr * num_gpus

    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(), filename='{}_{}_{}_deeplab_accuracy_1x1_1024x1024_nobankwoody.txt'.format(
        args.model, args.backbone, args.dataset))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
