import os.path
from data.unaligned_dataset import UnalignedDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import numpy as np
import torch.nn.functional as F
import torch
from pathlib import Path
from skimage import io as skio
import torchvision.transforms.v2.functional as TF
import math


class GeppettoDataset(UnalignedDataset):
    def __init__(self, opt):
        super().__init__(opt)

        self.dir_paired =  Path(os.path.join(opt.dataroot, "paired_dataset")) # opt.dir_paired)
        self.dir_labeled =  Path(os.path.join(opt.dataroot, "labeled_masks")) #  opt.dir_labeled)

        self.paired_A_paths = list((self.dir_paired / "masks").glob("*"))
        self.paired_B_paths = list((self.dir_paired / "images").glob("*"))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = skio.imread(A_path).astype(np.uint8)
        A_img = np.expand_dims(A_img, (0, 1)) #.transpose(3, 0, 1, 2)

        B_img = skio.imread(B_path).astype(np.uint8)
        B_img = np.expand_dims(B_img, (0, 1)) #.transpose(3, 0, 1, 2)
        # A_img = np.clip(A_img, 0, np.max(B_img)).astype(np.uint8)

        # print("A_img (pre-transform)", A_img.dtype, A_img.shape, np.min(A_img), np.max(A_img)) # A_img (pre-clip) uint8 (128, 128, 16) 36 152
        # print("B_img (pre-transform)", B_img.dtype, B_img.shape, np.min(B_img), np.max(B_img)) # B_img (pre-clip) uint8 (128, 128, 16) 12 252


        # Apply image transformation
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        # transform = get_transform(modified_opt, grayscale=True)

        # sample_path = Path(A_path)
        # labeled_path = self.dir_labeled / f"synth_sample_{sample_path.stem.split("_")[-1]}.tif"
        # base_mask = Image.fromarray(skio.imread(labeled_path)).convert("RGB")

        # A, base_mask = transform(A_img, base_mask)  # Applies the same transformation
        # A = transform(A_img)
        # B = transform(B_img)

        
        

        # SAM GT
        # uniques, counts = np.unique(base_mask, return_counts=True)
        # uniques = np.delete(uniques, np.argmax(counts))
        # A_exploded = base_mask == uniques[:, None, None]
        # A_centroids = self.get_centroids(A_exploded)

        # Semi-paired GT
        i_paired = np.random.randint(len(self.paired_A_paths))
        A_paired = skio.imread(self.paired_A_paths[i_paired]).astype(np.uint8)
        A_paired = np.expand_dims(A_paired, (0, 1)) #.transpose(3, 0, 1, 2)
        
        B_paired = skio.imread(self.paired_B_paths[i_paired]).astype(np.uint8)
        B_paired = np.expand_dims(B_paired, (0, 1)) #.transpose(3, 0, 1, 2)

        
        # print("A_paired (pre-transform)", A_paired.dtype, A_paired.shape, np.min(A_paired), np.max(A_paired))
        # print("B_paired (pre-transform)", B_paired.dtype, B_paired.shape, np.min(B_paired), np.max(B_paired))
        # A_paired, B_paired = transform(A_paired, B_paired)
        # print("A_paired (post-transform)", A_paired.dtype, A_paired.shape, torch.min(A_paired), torch.max(A_paired)) # A_paired (post-transform) torch.float32 torch.Size([1, 256, 256]) tensor(-0.9843) tensor(-0.6314)
        # print("B_paired (post-transform)", B_paired.dtype, B_paired.shape, torch.min(B_paired), torch.max(B_paired)) # B_paired (post-transform) torch.float32 torch.Size([1, 256, 256]) tensor(-1.) tensor(-0.3961)

        
        # Random affine 
        if self.opt.isTrain:
            A = self.synchronized_3d_augment(A_img, is_mask_A=True, aug=True)
            B = self.synchronized_3d_augment(B_img, aug=True)
            A_paired, B_paired = self.synchronized_3d_augment(A_paired, B_paired, is_mask_A=True, aug=True)

            # print("A (post-transform)", A.dtype, A.shape, torch.min(A), torch.max(A)) #
            # print("B (post-transform)", B.dtype, B.shape, torch.min(B), torch.max(B)) # 
            # print("A_paired (post-transform)", A_paired.dtype, A_paired.shape, torch.min(A_paired), torch.max(A_paired)) # A_paired (post-transform) torch.float32 torch.Size([1, 256, 256]) tensor(-0.9843) tensor(-0.6314)
            # print("B_paired (post-transform)", B_paired.dtype, B_paired.shape, torch.min(B_paired), torch.max(B_paired)) # B_paired (post-transform) torch.float32 torch.Size([1, 256, 256]) tensor(-1.) tensor(-0.3961)

        # from matplotlib import pyplot as plt
        # plt.figure(figsize=(10, 10))
        # plt.subplot(221)
        # plt.imshow(A[0], cmap='gray')
        # plt.title("A")
        # plt.subplot(222)
        # plt.imshow(B[0], cmap='gray')
        # plt.title("B")
        # plt.subplot(223)
        # plt.imshow(A_paired[0], cmap='gray')
        # plt.title("A_paired")
        # plt.subplot(224)
        # plt.imshow(B_paired[0], cmap='gray')
        # plt.title("B_paired")
        # plt.savefig(f"examples/{i_paired}.png")

        return {'A': A, 'B': B, 
                'A_paths': A_path, 'B_paths': B_path, 
                # 'A_exploded': A_exploded, 'A_centroids': A_centroids,  # SAM GT
                'A_paired': A_paired, 'B_paired': B_paired
                }
    
    def get_centroids(self, masks_onehot):
        """
        Fast vectorized centroid calculation for NumPy arrays.
        Args:
            masks_onehot: Boolean or integer array of shape (N, H, W).
        Returns:
            centroids: Array of shape (N, 2) in [x, y] format.
        """
        N, H, W = masks_onehot.shape
        
        # 1. Create coordinate grids (y_grid is [0], x_grid is [1])
        grid = np.indices((H, W))
        y_grid = grid[0]
        x_grid = grid[1]

        # 2. Calculate areas
        areas = masks_onehot.sum(axis=(1, 2))
        areas = np.maximum(areas, 1e-8) # Prevent zero division

        # 3. Multiply and sum
        sum_y = (masks_onehot * y_grid).sum(axis=(1, 2))
        sum_x = (masks_onehot * x_grid).sum(axis=(1, 2))

        # 4. Average to get centroids
        cent_y = sum_y / areas
        cent_x = sum_x / areas

        # 5. Stack and convert to int - (X, Y) FORMAT IS REQUIRED BY SAM
        centroids = np.stack((cent_x, cent_y), axis=1).astype(int)

        return centroids
       
    def random_photometric_distort_3d(self, image_1, image_2=None, p=0.9):
        """
        Applies brightness and contrast adjustments to a 3D volume.
        Safely handles CUT's standard [-1, 1] tensor normalization.
        """
        if random.random() > p:
            if image_2 is None:
                return image_1
            else:
                return image_1, image_2
            
        # Map from [-1, 1] to [0, 1] so brightness math works correctly
        # img_aug_1 = (image_1.clone() + 1.0) / 2.0
        # if image_2 is not None:
        #     img_aug_2 = (image_2.clone() + 1.0) / 2.0

        img_aug_1 = image_1.clone() # (image_1.clone() - torch.min(image_1)) / (torch.max(image_1) - torch.min(image_1))
        if image_2 is not None:
            img_aug_2 = image_2.clone() #  (image_2.clone() - torch.min(image_2)) / (torch.max(image_2) - torch.min(image_2))


        # Torchvision randomly decides whether to do Brightness or Contrast first
        do_contrast_first = random.random() < 0.5

        # 1. First Pass
        if do_contrast_first:
            c_factor = random.uniform(0.7, 1.3)
            mean_1 = img_aug_1.mean(dim=[-3, -2, -1], keepdim=True)
            img_aug_1 = (img_aug_1 - mean_1) * c_factor + mean_1
            if image_2 is not None:
                mean_2 = img_aug_2.mean(dim=[-3, -2, -1], keepdim=True)
                img_aug_2 = (img_aug_2 - mean_2) * c_factor + mean_2
        else:
            b_factor = random.uniform(0.6, 1.4)
            img_aug_1 = img_aug_1 * b_factor
            if image_2 is not None:
                img_aug_2 = img_aug_2 * b_factor

        # 2. Second Pass
        if do_contrast_first:
            b_factor = random.uniform(0.6, 1.4)
            img_aug_1 = img_aug_1 * b_factor
            if image_2 is not None:
                img_aug_2 = img_aug_2 * b_factor
        else:
            c_factor = random.uniform(0.7, 1.3)
            mean_1 = img_aug_1.mean(dim=[-3, -2, -1], keepdim=True)
            img_aug_1 = (img_aug_1 - mean_1) * c_factor + mean_1
            if image_2 is not None:
                mean_2 = img_aug_2.mean(dim=[-3, -2, -1], keepdim=True)
                img_aug_2 = (img_aug_2 - mean_2) * c_factor + mean_2

        # Map back to [-1, 1] and clamp to valid bounds
        # img_aug_1 = (img_aug_1 * 2.0) - 1.0
        # img_aug_1 = torch.clamp(img_aug_1, -1.0, 1.0)
        # if image_2 is not None:
        #     img_aug_2 = (img_aug_2 * 2.0) - 1.0
        #     img_aug_2 = torch.clamp(img_aug_2, -1.0, 1.0)

        if image_2 is not None:
            return img_aug_1, img_aug_2

        return img_aug_1


    def synchronized_3d_augment(self, vol_A, vol_B=None, is_mask_A=False, is_mask_B=False, aug=True):
        """
        Applies true 3D affine transformations (Rotation, Translation, Scale, Flips).
        vol_A must be [1, C, Z, H, W]
        """
        vol_A = torch.from_numpy(vol_A.astype(np.float32))
        if vol_B is not None:
            vol_B = torch.from_numpy(vol_B.astype(np.float32))
    
        if aug:
            device = vol_A.device
            B, C, Z, H, W = vol_A.shape
            
            # ==========================================
            # 1. Generate Geometric Parameters
            # ==========================================
            angle_rad = math.radians(random.uniform(0.0, 360.0))
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            
            tx = random.uniform(-0.2, 0.2)
            ty = random.uniform(-0.2, 0.2)
            tz = random.uniform(-0.1, 0.1) 
            
            s = random.uniform(0.9, 1.1)
            
            # Random Flips (-1 or 1)
            flip_x = random.choice([-1.0, 1.0])
            flip_y = random.choice([-1.0, 1.0])
            flip_z = random.choice([-1.0, 1.0])

            # Multiply scale by flip to embed the flip directly into the matrix
            sx = s * flip_x
            sy = s * flip_y
            sz = 1.0 * flip_z # Z is not scaled, only flipped

            # ==========================================
            # 2. Build the 3D Affine Matrix
            # ==========================================
            theta = torch.tensor([
                [sx * cos_a, -sy * sin_a,  0.0,  tx],
                [sx * sin_a,  sy * cos_a,  0.0,  ty],
                [       0.0,         0.0,   sz,  tz]  
            ], dtype=torch.float32, device=device).unsqueeze(0) 

            grid = F.affine_grid(theta, size=(B, C, Z, H, W), align_corners=False)

            # ==========================================
            # 3. Apply Geometric Transforms
            # ==========================================
            mode_A = 'nearest' if is_mask_A else 'bilinear'
            aug_A = F.grid_sample(vol_A, grid, mode=mode_A, padding_mode='reflection', align_corners=False)
            
            if vol_B is not None:
                mode_B = 'nearest' if is_mask_B else 'bilinear'
                aug_B = F.grid_sample(vol_B, grid, mode=mode_B, padding_mode='reflection', align_corners=False)
                
                # ==========================================
                # 4. Apply Photometric Transforms (Images ONLY)
                # ==========================================
                aug_A, aug_B = self.random_photometric_distort_3d(aug_A, aug_B)
            else:
                aug_A = self.random_photometric_distort_3d(aug_A)
        else:
            aug_A = vol_A
            aug_B = vol_B
        
        # Normalize NumPy volume
        # if grayscale:
        aug_A = (aug_A / 255. - 0.5) / 0.5
        # else:
        #     mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)[:, None, None, None]
        #     std = np.array([0.5, 0.5, 0.5], dtype=np.float32)[:, None, None, None]
        #     transform_list.append(transforms.Lambda(lambda vol: (vol - mean) / std))
        if vol_B is not None:
            # aug_B = torch.from_numpy(aug_A.astype(np.float32))
            # Normalize NumPy volume
            # if grayscale:
            aug_B = (aug_B / 255. - 0.5) / 0.5
            
            return aug_A[0], aug_B[0]
        return aug_A[0]