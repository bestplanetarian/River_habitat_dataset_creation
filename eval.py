from __future__ import print_function

import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from torchvision import transforms
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.score import SegmentationMetric, pixelAccuracy
from core.utils.visualize import get_color_pallete
from core.utils.logger import setup_logger
from core.utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler

from train import parse_args
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from PIL import Image

from scipy.ndimage import gaussian_filter

def gaussian_weights(patch_size, sigma=0.125):
    """Generate 2D Gaussian weights for blending."""
    # Sigma is relative to patch_size (e.g., sigma=0.125 * 1024 = 128)
    weights = np.ones((patch_size, patch_size))
    weights = gaussian_filter(weights, sigma=sigma * patch_size)
    return torch.from_numpy(weights).float().to("cuda")


class Evaluator(object):
    def modify_model_for_grayscale(self):
         # Modify the first convolutional layer of the pretrained model to accept 1-channel input
        first_conv_layer = self.model.pretrained[0]  # Accessing the first Conv2d directly
        if isinstance(first_conv_layer, nn.Conv2d) and first_conv_layer.in_channels == 3:
            new_conv_layer = nn.Conv2d(1, first_conv_layer.out_channels, kernel_size=first_conv_layer.kernel_size,
                                       stride=first_conv_layer.stride, padding=first_conv_layer.padding,
                                       dilation=first_conv_layer.dilation, groups=first_conv_layer.groups,
                                       bias=first_conv_layer.bias is not None)
            # Copy weights from the original layer, summing across the input channels
            with torch.no_grad():
                new_conv_layer.weight.data = torch.mean(first_conv_layer.weight.data, dim=1, keepdim=True)
                if first_conv_layer.bias is not None:
                    new_conv_layer.bias.data = first_conv_layer.bias.data.clone()
            
            new_conv_layer = new_conv_layer.to(self.device)
            self.model.pretrained[0] = new_conv_layer


    def update_confusion_matrix(self, cm, pred, target, num_classes=5):
        classes = ['Boulder', 'Shadow', 'Fine', 'Rocky Fine']
        #lasses = ['Others', 'Shadow']
        #classes = ['Boulder', 'Shadow', 'Fine', 'Rocky_Fine']
        # Flatten the arrays
        pred = pred.flatten()
        target = target.flatten()
        
        print(f"Range of pred: {pred.min()} to {pred.max()}")
        print(f"Range of target: {target.min()} to {target.max()}")
        
        
    
        # Exclude the 'back' class (assume 'back' is class 0)
        mask = target != 0
    
        pred = pred[mask]
        target = target[mask]
    
        # Adjust the class indices
        pred = pred - 1
        target = target -1 
    
        # Compute the confusion matrix
        cm += confusion_matrix(target, pred, labels=np.arange(num_classes - 1))
    
        
    
        return cm



    def __init__(self, args):
        self.args = args
        self.device = self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485], [.229]),
        ])

        # dataset and dataloader
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='testval', transform=input_transform)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                            aux=args.aux, pretrained=True, pretrained_base=False,
                                            local_rank=args.local_rank,
                                            norm_layer=BatchNorm2d).to(self.device)
        print(self.model)
        
        #self.modify_model_for_grayscale()
        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank])
        self.model.to("cuda")
        #output_device=args.local_rank
        
        
        
        
        
        self.metric = SegmentationMetric(val_dataset.num_class)

    def eval(self):
        mean_pixel_accuracy = 0
        label_mean_pixel_accuracy = 0
        print(self.model)
        self.metric.reset()
        self.model.eval()
        
        classes = ['Boulder', 'shadow', 'Fine', 'Rockyfine']
        
        
        acc = []
        
        cm = np.zeros((4, 4), dtype=np.int64)
        
        all_preds = []
        all_targets = []
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        
        

        file_line = []

        for i, (image, target, filename) in enumerate(self.val_loader):
        #Move data to CPU
            image = image.to("cuda")
            target = target.to("cuda")
    
            print(image.shape)
            print(target.shape)

      


            '''
            
            _, _, height, width = image.shape
    
            
    
            patch_size = 1024
            overlap = 512
            #h_patches = (height + patch_size - 1) // patch_size
            #w_patches = (width + patch_size - 1) // patch_size


            stride = patch_size - overlap  # stride = 512 for 50% overlap
            h_patches = (height - overlap + stride - 1) // stride
            w_patches = (width - overlap + stride - 1) // stride

            # Initialize output and weight accumulator
            full_output = torch.zeros((1, 7, height, width)).to("cuda")
            weight_mask = torch.zeros((1, 1, height, width)).to("cuda")





            #full_output = torch.zeros((1, 7, height, width)).to("cuda")  # Initialize on GPU
            #weighted_mask = weight_mask = torch.zeros((1, 1, height, width)).to("cuda")

            gauss_weights = gaussian_weights(patch_size)

            for h_idx in range(h_patches):
                for w_idx in range(w_patches):
            # Calculate crop coordinates
                    h_start = h_idx * stride
                    w_start = w_idx * stride
                    h_end = min(h_start + patch_size, height)
                    w_end = min(w_start + patch_size, width)

            # Skip patches smaller than `min_patch_size`
                    if (h_end - h_start) < 32 or (w_end - w_start) < 32:
                         continue

                    patch = image[:, :, h_start:h_end, w_start:w_end]
            
                    with torch.no_grad():
                         patch_output = model(patch)
                         if isinstance(patch_output, tuple):
                            patch_output = patch_output[0]

            # Apply blending weights
                    current_weights = gauss_weights[:h_end-h_start, :w_end-w_start].unsqueeze(0).unsqueeze(0)
                    full_output[:, :, h_start:h_end, w_start:w_end] += patch_output * current_weights
                    weight_mask[:, :, h_start:h_end, w_start:w_end] += current_weights

    # Normalize by the weight mask to blend smoothly
            full_output = full_output / weight_mask.clamp(min=1e-6)  # Avoid division by zero
        
            
            # Place the patch output in the correct position
            # full_output[:, :, h_start:h_end, w_start:w_end] = patch_output

    # Get the final output
            outputs = full_output
            

             # Ensure target has the correct shape (remove channel dimension if present)
            #if len(target.shape) == 4:
            #   target = target.squeeze(1)  # Remove channel dimension for class indices

            
            

    # Calculate metrics
            self.metric.update(outputs, target)
            pixAcc, mIoU, singpixacc = self.metric.get()

            acc.append(pixAcc)

            image_name = filename[0] if isinstance(filename, (list, tuple)) else filename
            file_line.append([filename, singpixacc])
            mean_pixel_accuracy += singpixacc

    # Get predictions
            pred = torch.argmax(outputs, 1).cpu().data.numpy()  # [1, H, W]
            pred = pred.squeeze(0)  # [H, W]

    # Prepare target for numpy operations
            target = target.long().cpu().data.numpy()  # [1, H, W] or [H, W]
            if len(target.shape) == 3:
               target = target.squeeze(0)  # [H, W]

            single_acc, _, _ = pixelAccuracy(pred, target)

            logger.info(
              "Sample: {:d}, Image: {}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
               i + 1, image_name, single_acc * 100, mIoU * 100
            ))

            cm = self.update_confusion_matrix(cm, pred, target, 3)
            '''
            

            
            with torch.no_grad():
                 outputs = model(image)  # Move image to GPU only during inference

            outputs = outputs[0].to("cuda")  # Move output back to CPU
            #print(outputs.shape)

            self.metric.update(outputs, target)
            pixAcc, mIoU, singpixacc = self.metric.get()
    
            acc.append(pixAcc)
    
            image_name = filename[0] if isinstance(filename, (list, tuple)) else filename
            file_line.append([filename, singpixacc])
            mean_pixel_accuracy += singpixacc
    
            pred = torch.argmax(outputs, 1).cpu().data.numpy()  # Ensure `pred` is on CPU
            pred = pred.squeeze(0)
    
            target = target.long().cpu().data.numpy()  # Ensure `target` is on CPU
    
            single_acc, _, _ = pixelAccuracy(pred, target)

            logger.info(
                "Sample: {:d}, Image: {}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                 i + 1, image_name, single_acc * 100, mIoU * 100
            ))


            #pixel_labeled = np.sum(target > 0)
            #print(f'Number of labeled pixels: {pixel_labeled}')
    
            cm = self.update_confusion_matrix(cm, pred, target, 5)
            
            
            
            #mask, gt, raw, gt_msk = get_color_pallete(pred, target, self.args.dataset)
            #mask.save(os.path.join(outdir, os.path.splitext(filename[0])[0] + '.png'))
            #gt.save(os.path.join(outdir, os.path.splitext(filename[0])[0] + '_mask.png'))
            #raw.save(os.path.join(outdir, os.path.splitext(filename[0])[0] + '_raw.png'))
            #gt_msk.save(os.path.join(outdir, os.path.splitext(filename[0])[0] + '_gt_msk.png'))
            
            #pred_path = '/home/swz45/Documents/awesome-semantic-segmentation-pytorch/awesome-semantic-segmentation-pytorch-master/runs/pred_pic_Lamine/fcn32s_vgg16_substrate/' + os.path.splitext(filename[0])[0] + '_raw.png'
            
            #gt_path = '/home/swz45/Documents/awesome-semantic-segmentation-pytorch/awesome-semantic-segmentation-pytorch-master/runs/pred_pic_Lamine/fcn32s_vgg16_substrate/' + os.path.splitext(filename[0])[0] + '_gt_msk.png'
            
            #pred_class = Image.open(pred_path)
            #gt_class = Image.open(gt_path) 
            '''   
            pred_class1 = np.array(raw)
            gt_class2 = np.array(gt_msk)
            
            
            count_same_pixel = 0
            total_second_pixel = 0
            
    
  
    
   
            
        # Iterate through each row
            for row in range(raw.height):
                for col in range(raw.width):
                    pixel1 = pred_class1[row, col]
                    pixel2 = gt_class2[row, col]
                    
        
        	    # Ignore black areas in the second image list(pixel2) in color_pixel
                    if pixel2 > 0:
                    #if list(pixel2) in color_pixel:
                       total_second_pixel += 1
                       if pixel1 == pixel2:
                          count_same_pixel += 1
            
            percentage_correct = (count_same_pixel / total_second_pixel) * 100
            
            print('Percentage of correct pixel based on label is :',  percentage_correct)
            
            count_same_pixel = 0
            total_second_pixel = 0              
            
            label_mean_pixel_accuracy += percentage_correct   
            '''    
                
        print('The mean_pixel_accuracy for test_set is: ')
        overall_pixel_accuracy = np.diag(cm).sum()/ cm.sum()
        print(f"overall_pixel_accuracy on confusion_matrix is the {overall_pixel_accuracy:.4f}")
        print(f"overall_pixel_accuracy on metric is the {acc[-1]:.4f}")

        
        with open("deeplabv3_5x5_1024x1024_test.txt", "w") as file:
             for line in file_line:
                 file.write(f"{line[0]}: validation pixAcc = {line[1] * 100:.3f}\n")
        
        
        
        #print(all_preds)
        #all_preds = np.concatenate(all_preds, axis=0)
        #all_targets = np.concatenate(all_targets)

        # Compute the confusion matrix
        #cm = compute_confusion_matrix(all_preds, all_targets)
        
        correct_predictions = np.diag(cm)  # True Positives for each class
        total_predictions = cm.sum(axis=1)  # Ground Truth Pixels for each class
        predicted_totals = cm.sum(axis=0)  # Total Predicted Pixels for each class

        for i, class_name in enumerate(classes):
            if total_predictions[i] > 0:
               recall = correct_predictions[i] / total_predictions[i]  # Recall
            else:
               recall = 0.0
    
            if predicted_totals[i] > 0:
               precision = correct_predictions[i] / predicted_totals[i]  # Precision
            else:
               precision = 0.0

            if precision + recall > 0:
               f1_score = 2 * (precision * recall) / (precision + recall)  # F1-score
            else:
               f1_score = 0.0
    
            print(f"  Class: {class_name}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1_score:.4f}")
        

        # Plot the confusion matrix
        '''
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        plt.figure(figsize=(10, 10)) 
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig('/home/swz45/Documents/awesome-semantic-segmentation-pytorch/awesome-semantic-segmentation-pytorch-master/matrix_linegraph/fcn32_shadow.png')
        '''


        '''  
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Normalize to percentages
        cm_percentage = np.round(cm_percentage, decimals=2)  
        print(cm_percentage)
        '''
        
        cm_percentage = cm.astype(np.float32) / cm.sum() * 100.0
        print("\nConfusion Matrix as Percent of Total:")
        print(cm_percentage)

  


        disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=classes)
        fig, ax = plt.subplots(figsize=(10, 10))  # Create figure and axis explicitly
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)  # Ensure colorbar is enabled

        # Safely access the colorbar
        if ax.collections:  # Check if collections exist
           colorbar = ax.collections[0].colorbar
           colorbar.set_label('Percentage (%)')  # Update colorbar label
        else:
           print("Warning: No collections found in the axis. The colorbar may not be generated.")

        # Safely access the colorbar after the plot is created
        # colorbar = ax.collections[0].colorbar
        # colorbar.set_label('Percentage (%)')
        
        plt.savefig('/home/swz45/Documents/awesome-semantic-segmentation-pytorch/awesome-semantic-segmentation-pytorch-master/matrix_linegraph/deeplab_1024x1024_nowoodybank.png')
        plt.show()
        
        synchronize()


if __name__ == '__main__':
    args = parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    print(args.distributed)
    print(args.model)
    print(args.no_cuda)
    print(args.model)
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

    # TODO: optim code
    args.save_pred = True
    if args.save_pred:
        outdir = '../runs/pred_deeplab5x5_1024x1024_shadowonly_nowoodybank/{}_{}_{}'.format(args.model, args.backbone, args.dataset)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    logger = setup_logger("semantic_segmentation_deeplab5x5_1024x1024", args.log_dir, get_rank(),
                          filename='{}_{}_{}_log_deeplab5x5_1024x1024_dataset_nowoodybank.txt'.format(args.model, args.backbone, args.dataset), mode='a+')

    evaluator = Evaluator(args)
    evaluator.eval()
    torch.cuda.empty_cache()
