from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import glob
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
import torch.nn.functional as F 
import time
from torch.utils.data import DataLoader, Dataset
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from externals.utils import set_seed, make_dirs, cfg_init
from externals.dataloading import read_image_mask, get_train_valid_dataset, get_transforms, CustomDataset
from externals.models import Unet3d_point_regressor, point_regressor_simple, timm3d_efficientnet
from externals.metrics import AverageMeter, calc_fbeta, fbeta_numpy
from externals.training_procedures import get_scheduler, scheduler_step
# from externals.postprocessing import post_process
from torch.optim.swa_utils import AveragedModel, SWALR
import wandb
import timm
import ast
import h5py
import segmentation_models_pytorch as smp
from monai.networks.nets.unetr import UNETR
from scipy.ndimage.filters import gaussian_filter

dl = smp.losses.DiceLoss(mode="binary", ignore_index=-1, smooth=0)
bce = smp.losses.SoftBCEWithLogitsLoss(ignore_index=-1, smooth_factor=0, reduction="none")

from volumentations import *
l1_loss = torch.nn.L1Loss()
mse_loss = torch.nn.MSELoss(reduction="none")

def criterion(y_preds, y_true):
    ignore_idxs = (y_true == -1)
    y_preds = y_preds[~ignore_idxs]
    y_true = y_true[~ignore_idxs]
    loss = mse_loss(y_true, y_preds)
    return loss 

# def criterion(predictions, labels):
#     # Mask for valid (labeled) points
#     valid_labels = (labels != -1).all(dim=-1, keepdim=True)  # Check both coordinates are not -1

#     # Expand dimensions for broadcasting
#     predictions_exp = predictions.unsqueeze(2)  # [B, D, 1, T, 2]
#     labels_exp = labels.unsqueeze(3)             # [B, D, T, 1, 2]

#     # Compute squared distances for all pairs
#     distances = (predictions_exp - labels_exp) ** 2
#     distances = distances.sum(dim=-1)  # Sum over the last dimension (x and y)

#     # Apply mask to the distances
#     max_distance = distances.max() + 1
#     distances = torch.where(valid_labels, distances, max_distance)

#     # Find the minimum distance for each label point to any prediction
#     min_distances_labels = distances.min(dim=3).values  # Minimum distances from each label to closest prediction

#     # Find the minimum distance for each prediction point to any label
#     min_distances_preds = distances.min(dim=2).values  # Minimum distances from each prediction to closest label

#     # Calculate the mean of the valid minimum distances for both directions
#     valid_indices = valid_labels.squeeze(-1)
#     loss_labels = min_distances_labels[valid_indices].mean()
#     loss_preds = min_distances_preds[valid_indices].mean()

#     # Final symmetric loss
#     loss = loss_labels + (loss_preds/10)

#     return loss

def euclidean_distance(predicted, actual):
    counts = (actual != -1).sum(dim = -1)/2
    # print(counts)
    # Calculate the squared differences along the last dimension (x and y coordinates)
    squared_diff = (predicted - actual) ** 2

    # Sum the squared differences along the last dimension
    sum_squared_diff = torch.sum(squared_diff, dim=-1)

    # Take the square root to get the Euclidean distance
    euclidean_dist = torch.sqrt(sum_squared_diff)

    return euclidean_dist, counts

class CFG:
    is_multiclass = False
    # edit these so they match your local data path
    comp_name = 'vesuvius_unroll_3d'
    comp_dir_path = './input'
    comp_folder_name = 'vesuvius-challenge-ink-detection'
    comp_dataset_path = f'{comp_dir_path}{comp_folder_name}/'
    # ========================
    
    exp_name = 'full_forecast_segment_only'
    # starting_checkpoint = "/home/ryanc/kaggle/working/outputs/vesuvius_3d/pretrained/pretrained_model.pth"
    starting_checkpoint = "None"
    # starting_checkpoint = f"working/outputs/vesuvius_unroll_3d/full_forecast/vesuvius_unroll_3d-models/full_forecast_3d_unet_last.pth"
    # ============== pred target =============
    target_size = 1
    # ============== model cfg =============
    model_name = '3d_unet'
    # ============== training cfg =============
    size = 256
    in_chans = 1

    train_batch_size = 12
    valid_batch_size = train_batch_size
    use_amp = True

    history = 100
    forecast_length = 100
    scheduler = 'GradualWarmupSchedulerV2'
    epochs = 100
    valid_id = "856"
    # adamW warmup
    warmup_factor = 1
    lr = 1e-4 / warmup_factor
    # ============== fixed =============
    min_lr = 1e-6
    weight_decay = 1e-5
    max_grad_norm = 10
    num_workers = 4
    seed = int(time.time())
    # ============== set dataset path =============
    print('set dataset path')

    outputs_path = f'working/outputs/{comp_name}/{exp_name}/'

    submission_dir = outputs_path + 'submissions/'
    submission_path = submission_dir + f'submission_{exp_name}.csv'

    model_dir = outputs_path + \
        f'{comp_name}-models/'

    figures_dir = outputs_path + 'figures/'

    log_dir = outputs_path + 'logs/'
    log_path = log_dir + f'{exp_name}.txt'


cfg_init(CFG)

from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation as R
def rotate_volume(volume, points, all_mid, history, p1, p2):
    # Calculate the angle of rotation
    center = np.array(volume.shape)//2
    dx, dy = p2 - p1
    angle = np.arctan2(dy, dx)
    #1.57079633 90 degrees in radian
    rotation_matrix = R.from_euler('z', (-angle)+1.57079633, degrees=False).as_matrix()
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = rotation_matrix
    affine_matrix[:3, 3] = center - np.dot(rotation_matrix, center)
    
    volume = affine_transform(volume, affine_matrix[:3, :3], offset=affine_matrix[:3, 3], order=1, mode='constant', cval=0.0)
    
    # Rotate the points
    rotated_points = np.empty_like(points)
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            rotated_points[i, j] = np.dot(points[i, j], rotation_matrix.T)
    
    # Rotate the points
    rotated_all_mid = np.empty_like(all_mid)
    for i in range(all_mid.shape[0]):
        for j in range(all_mid.shape[1]):
            rotated_all_mid[i, j] = np.dot(all_mid[i, j], rotation_matrix.T)
            
    # Rotate the points
    rotated_history = np.empty_like(history)
    for i in range(history.shape[0]):
        for j in range(history.shape[1]):
            rotated_history[i, j] = np.dot(history[i, j], rotation_matrix.T)
    
    return volume, rotated_points, rotated_all_mid, rotated_history

def get_augmentation():
    return Compose([
    ], p=1.0)

aug = get_augmentation()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = point_regressor_simple(CFG)
model = Unet3d_point_regressor(CFG)
class CustomDataset(Dataset):
    def __init__(self, volume_path, cfg, labels=None, transform=None, mode="test", size=1000, coords = None, check_counts = True):
        self.volumes = volume_path
        self.cfg = cfg
        self.labels = labels
        self.transform = transform
        self.mode = mode
        self.size = size
        self.coords = coords
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        invalid_volume = True
        reroll_volume = False
        while invalid_volume:
            #pick a segment to draw from
            coord_pick = np.random.randint(0, len(self.coords))
            selected_coords = self.coords[coord_pick].copy()
            #pick a random layer from the segment
            z_idx = np.random.randint(self.cfg.size//2, selected_coords.shape[0]-self.cfg.size//2)
            #pick a random xy starting point
            xy_idx = np.random.randint(self.cfg.size//2, selected_coords.shape[1]-self.cfg.size//2)
            #grab the y, x, z for that layer and xy starting point
            mid = selected_coords[z_idx, xy_idx].copy()
            last_point = selected_coords[z_idx, xy_idx-1].copy()
            #grab offsets for all of the layers we're predicting
            all_mid = selected_coords[z_idx-(self.cfg.size//2):(z_idx+self.cfg.size//2), xy_idx:xy_idx+1].copy()
            #grab all the past points
            history = selected_coords[z_idx-(self.cfg.size//2):z_idx+(self.cfg.size//2), xy_idx-self.cfg.history:xy_idx].copy()
            #grab all the future points we want to predict
            forecast = selected_coords[z_idx-(self.cfg.size//2):z_idx+(self.cfg.size//2), xy_idx-self.cfg.history:xy_idx+self.cfg.forecast_length].copy()
  
            #use all the offsets so on every layer we are predicting the movement from last known point instead of position
            history = (history - mid[None, None, :]).copy()
            forecast = (forecast- mid[None, None, :]).copy()
            all_mid = (all_mid - mid[None, None, :]).copy()
            with h5py.File(self.volumes, 'r') as f:
                image = f["scan_volume"][mid[1]-(self.cfg.size//2):(mid[1]+(self.cfg.size//2)),
                                        mid[0]-(self.cfg.size//2):(mid[0]+(self.cfg.size//2)),
                                        mid[2]-(self.cfg.size//2):(mid[2]+(self.cfg.size//2))]/255.
            image, forecast, all_mid, history = rotate_volume(image, forecast, all_mid, history, np.array([self.cfg.size//2, self.cfg.size//2]), (last_point[:2]-mid[:2] + np.array([self.cfg.size//2,self.cfg.size//2])))
            drawn_forecast = forecast.copy() + self.cfg.size//2
            mask = (drawn_forecast[..., 0] < 0) | (drawn_forecast[..., 0] > self.cfg.size-1) | (drawn_forecast[..., 1] < 0) | (drawn_forecast[..., 1] > self.cfg.size-1)
            drawn_forecast[mask] = [-1, -1, -1]
            realigned_forecast = np.zeros_like(drawn_forecast) - 1
            label_volume = np.zeros_like(image).astype(np.uint8)
            # for depth, row in enumerate(history[self.cfg.size//2:(self.cfg.size//2)+1, -2:]):
            for depth, row in enumerate(history[:, -2:]):
                for xy_idx, point in enumerate(row):
                    point = point + self.cfg.size//2
                    if (point < 0).any() or (point > self.cfg.size-1).any() :
                        continue
                    if image[point[1], point[0], depth] == 0:
                        continue
                    # image[point[1], point[0], depth] = 1
            for depth, row in enumerate(drawn_forecast):
                aligned_idx = 0
                pt_list = []
                for xy_idx, point in enumerate(row):
                    # image[point[1], point[0], depth] = 1
                    if ((point < 0).any() or (point > self.cfg.size-1).any()) or (image[point[1], point[0], depth] == 0):
                        continue
                    else:
                        if len(pt_list) > 0:
                            if (np.abs(pt_list[-1][0] - (int(point[0]))) > 10) or (np.abs(pt_list[-1][1] - (int(point[1]))) > 10):
                                reroll_volume = True
                                break
                        pt_list.append((int(point[0]), int(point[1])))
                    realigned_forecast[depth, aligned_idx] = point
                    aligned_idx += 1
                if reroll_volume:
                    break
                labeled_slice = label_volume[:, :, depth].copy()
                labeled_slice = np.dstack([labeled_slice, labeled_slice, labeled_slice])
                labeled_slice = cv2.polylines(labeled_slice, np.array([pt_list]).reshape(1, -1, 2), isClosed=False, color=(1), thickness=4)
                if labeled_slice.max() == 0:
                    # print("blank volume")
                    reroll_volume = True
                    break
                label_volume[:, :, depth] = labeled_slice[:, :, 0]
            
            image, final_label = image.astype(np.float16), realigned_forecast.astype(np.float16)

            if reroll_volume or final_label.max() == 0:
                # print("rerolled", (np.abs(pt_list[-1][0] - (int(point[0])))), (np.abs(pt_list[-1][1] - (int(point[1])))))
                reroll_volume = False
                continue
            else:
                # print("finished")
                invalid_volume = False
        os.makedirs("scratch_imgs", exist_ok=True)
        cv2.imwrite(f"scratch_imgs/{np.random.randint(0,10)}.png", (image[:, :, self.cfg.size//2]*255).astype(np.uint8))
        #all_mid only used for drawing on predictions
        return image[None], final_label[:, :, 0:2], all_mid, label_volume.astype(np.float16)

    
    
def train_fn(train_loader, model, criterion, optimizer, device):
    model.train()
    model.to(device)
    scaler = GradScaler(enabled=CFG.use_amp)
    losses = AverageMeter()
    volume_losses = AverageMeter()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    index = 0
    for step, (images, labels, all_mid, label_volume) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        label_volume = label_volume.to(device)
        batch_size = labels.size(0)

        with torch.autocast(device_type="cuda"):
            y_preds, volume_preds = model(images)
            loss = criterion(y_preds.clone(), labels.clone())
            volume_loss = bce(volume_preds, label_volume)
            # bad_images = volume_loss.mean(-1).mean(-1).mean(-1) > 10
            loss = loss.mean()
            volume_loss = volume_loss.mean()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm)
        pbar.set_description_str(f"point loss: {str(losses.avg)}, volume loss:{str(volume_losses.avg)}")
        losses.update(loss.item(), batch_size)
        volume_losses.update(volume_loss.item(), batch_size)
        scaler.scale(volume_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # os.makedirs("bad_images", exist_ok=True)
        # images = images.detach().cpu().numpy()
        # label_volume = label_volume.detach().cpu().numpy()

        # for image, label in zip(images[bad_images.cpu().numpy()], label_volume[bad_images.cpu().numpy()]):
        #     index += 1
        #     input_out = cv2.VideoWriter(f'./bad_images/{index}_input.avi', fourcc, 15, (image.shape[-1], image.shape[-2]))
        #     output_out = cv2.VideoWriter(f'./bad_images/{index}_output.avi', fourcc, 15, (image.shape[-1], image.shape[-2]))
        #     for idx, preds in enumerate(image):
        #         input_out.write(cv2.cvtColor((image[:, :, idx]*255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        #         output_out.write(cv2.cvtColor((label_volume[:, :, idx]*255).astype(np.uint8), cv2.COLOR_GRAY2BGR))

        #     input_out.release()
        #     output_out.release()

    return losses.avg, volume_losses.avg

def valid_fn(valid_loader, model, criterion, device):
    model.eval()
    error_distances = np.zeros((CFG.size, CFG.history+CFG.forecast_length))
    error_counts = np.zeros((CFG.size, CFG.history+CFG.forecast_length))
    losses = AverageMeter()
    volume_losses = AverageMeter()
    
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    os.makedirs("./extrapolations", exist_ok=True)
    os.makedirs("./pred_volumes", exist_ok=True)
    for step, (images, labels, all_mid, label_volume) in pbar:
        batch_size = labels.size(0)
        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                images = images.to(device)
                labels = labels.to(device)
                label_volume = label_volume.to(device)
                y_preds, volume_preds = model(images)
                batch_size = labels.size(0)
                loss = criterion(y_preds, labels).mean()
                volume_loss = bce(volume_preds, label_volume).mean()
                error_distance, error_count = euclidean_distance(y_preds.clone(), labels.clone())
                error_distances += error_distance.sum(0).detach().cpu().numpy()
                error_counts += error_count.sum(0).detach().cpu().numpy()
        pbar.set_description_str(f"point loss: {str(losses.avg)}, volume loss:{str(volume_losses.avg)}")
        losses.update(loss.mean().item(), batch_size)
        volume_losses.update(volume_loss.item(), batch_size)
        images = images.detach().cpu().numpy()[0][0]
        y_preds = y_preds[0].detach().cpu().numpy()
        labels = labels[0].detach().cpu().numpy()
        all_mid = all_mid[0].detach().cpu().numpy()
        volume_preds = torch.sigmoid(volume_preds[0]).detach().cpu().numpy()
        label_volume = label_volume[0].detach().cpu().numpy()
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(f'./extrapolations/{step}.avi', fourcc, 15, (images.shape[-1], images.shape[-2]))
        for depth, row in enumerate(labels):
            for xy_idx, point in enumerate(row):
                if ((point < 0).any() or (point > CFG.size-1).any()):
                    continue
                images[int(point[1]), int(point[0]), depth] = 1
        for depth, row in enumerate(y_preds):
            for xy_idx, point in enumerate(row):
                # print(point)
                if ((point < 0).any() or (point > CFG.size-1).any()):
                    continue
                if (images[int(point[1]), int(point[0]), depth] == 0):
                    break
                images[int(point[1]), int(point[0]), depth] = 0
            out.write(cv2.cvtColor((images[:, :, depth]*255).astype(np.uint8), cv2.COLOR_GRAY2BGR))


        out.release()
        out = cv2.VideoWriter(f'./pred_volumes/{step}.avi', fourcc, 15, (images.shape[-1], images.shape[-2]))
        for idx, preds in enumerate(volume_preds):
            out.write(cv2.cvtColor((volume_preds[:, :, idx]*255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
            # cv2.imwrite(f"./pred_volumes/{idx}.jpg", (volume_preds[:, :, idx]*255).astype(np.uint8))

        out.release()
        
        out = cv2.VideoWriter(f'./pred_volumes/{step}_labels.avi', fourcc, 15, (images.shape[-1], images.shape[-2]))
        for idx, preds in enumerate(label_volume):
            out.write(cv2.cvtColor((label_volume[:, :, idx]*255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
            # cv2.imwrite(f"./pred_volumes/{idx}.jpg", (volume_preds[:, :, idx]*255).astype(np.uint8))

        out.release()
    error_counts[error_counts == 0] = 1
    error_distances = error_distances/error_counts
    print(error_counts)
    metrics = {}
    extrap_distance = []
    extrap_metrics = []
    layer_num = []
    layer_metrics = []
    for i in range(error_distances.shape[1]):
        extrapolation_error = error_distances[:, i]
        extrapolation_error = extrapolation_error[error_counts[:, i] >= (error_counts[:, 0].mean()*.9)]
        if extrapolation_error.shape[0] > 0:
            extrapolation_error = extrapolation_error.mean()
            print(i, extrapolation_error)
        else:
            last_extrap = i
            break
        extrap_distance.append(i)
        extrap_metrics.append(extrapolation_error)
    for i, error in enumerate(error_distances):
        layer_distance = error[:last_extrap]
        layer_distance = layer_distance.mean()
        print(i, layer_distance)
        layer_num.append(i)
        layer_metrics.append(layer_distance)
    plt.figure(figsize=(10, 6))
    # plt.ylim([0,85])
    plt.plot(extrap_distance, extrap_metrics)
    metrics["extrapolation metrics"] = wandb.Image(plt)
    plt.close()
    plt.figure(figsize=(10, 6))
    # plt.ylim([0,40])
    plt.plot(layer_num, layer_metrics)
    metrics["layer metrics"] = wandb.Image(plt)
    plt.close()
    return losses.avg, volume_losses.avg, metrics
import random
import monai

train_coords = []
validation_coords = []
# os.makedirs("distance_plots", exist_ok=True)
# for coord_file in tqdm(glob.glob("/home/ryanc/kaggle/segment_arrays/*")):
#     coords = np.load(coord_file)
#     plt.hist(np.diff(coords, axis = -2).reshape(-1, 3)[:, 0], bins=15)
#     plt.hist(np.diff(coords, axis = -2).reshape(-1, 3)[:, 1], bins=15)
#     plt.yscale('log')
#     plt.xlabel('Distance')
#     plt.ylabel('Frequency')
#     plt.title('Histogram')
#     plt.savefig(f"distance_plots/{coord_file.split('/')[-1].split('.')[0]}.png")
#     plt.close()
#     if (coords.shape[0] < 1000) or (coords.shape[1] < 1000):
#         continue
#     if "20231005123335" in coord_file:
#         validation_coords.append(coords)
#     else:
#         coords = coords[-(len(coords)//2):]
#         train_coords.append(coords)

for coord_file in tqdm(glob.glob("segment_arrays/*")):
    coords = np.load(coord_file)
    
    x_diff = np.abs(np.diff(coords, axis = -2).reshape(-1, 3)[:, 0])
    y_diff = np.abs(np.diff(coords, axis = -2).reshape(-1, 3)[:, 1])
    if (coords.shape[0] < 1000) or (coords.shape[1] < 1000) or (x_diff.max() > 7) or (y_diff.max() > 7):
        continue
    if "20231005123335" in coord_file:
        validation_coords.append(coords)
    else:
        train_coords.append(coords)
    print(coords.size)
        
print(len(train_coords), len(validation_coords))
training_dataset = CustomDataset(volume_path="volume_compressed/volume_compressed.hdf5", labels=None, cfg=CFG,
                                 transform=None, mode="train", size = 100000, coords=train_coords, check_counts=True)
sampler = torch.utils.data.RandomSampler(training_dataset, replacement=False, num_samples=10000)
train_loader = DataLoader(training_dataset, batch_size=CFG.train_batch_size, shuffle=False, num_workers=32, pin_memory=False, drop_last=True, sampler=sampler)

valid_dataset = CustomDataset(volume_path="volume_compressed/volume_compressed.hdf5", labels=None, cfg=CFG, transform=None, size = 1000, coords=validation_coords)
valid_sampler = torch.utils.data.RandomSampler(valid_dataset, replacement=True, num_samples=1000)
valid_loader = DataLoader(valid_dataset, batch_size=CFG.train_batch_size, shuffle=False, num_workers=32, pin_memory=False, drop_last=True, sampler=valid_sampler)
cfg_pairs = {value:CFG.__dict__[value] for value in dir(CFG) if value[1] != "_"}
model_name = f"{CFG.exp_name}_{CFG.model_name}"

if os.path.exists(CFG.starting_checkpoint):
    print(CFG.starting_checkpoint)
    model.load_state_dict(torch.load(CFG.starting_checkpoint))
    
model = torch.nn.DataParallel(model)
model.to(device)
swa_model = AveragedModel(model)
swa_start = 2

best_counter = 0
best_loss = np.inf
best_score = 0
optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
# swa_scheduler = SWALR(optimizer, swa_lr=0.05)
scheduler = get_scheduler(CFG, optimizer)

wandb.init(
    project="3d_unroll_new",
    name=CFG.exp_name
)
for epoch in range(CFG.epochs):
    # avg_loss, volume_loss = 0, 0
    avg_loss, volume_loss = train_fn(train_loader, model, criterion, optimizer, device)
    
    avg_val_loss, avg_val_volume_loss, metrics = valid_fn(
        valid_loader, model, criterion, device)
    metrics.update({"avg_train_loss":avg_loss, "avg_val_loss":avg_val_loss, "avg_train_volume_loss":volume_loss, "avg_val_volume_loss":avg_val_volume_loss})
    print(metrics)
    wandb.log(metrics)
    if avg_val_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.module.state_dict(),
                CFG.model_dir + f"{model_name}_best.pth")

    scheduler_step(scheduler, avg_val_loss, epoch)
    torch.save(model.module.state_dict(),
            CFG.model_dir + f"{model_name}_last.pth")
