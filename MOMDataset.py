import os
import numpy as np
import torch
import random
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MOMDataset(Dataset):
    def __init__(self,
                 root_dir,
                 perturbed=True,
                 transform=None,
                 transform_ae=None,
                 skip_background=False):
        """
        root_dir: path to the 'train' folder of MVTec (or MVTeC-loco).
        perturbed: whether to add synthetic anomaly patches to 'student' image.
        transform: transform for final resize/normalize on the 'student' image.
        transform_ae: transform for autoencoder image (could be color jitter etc.),
                      followed by the same resizing & normalization steps, or as a Compose.
        skip_background: optional logic to skip background patches (like in CDO).
        """
        self.root_dir = root_dir
        self.perturbed = perturbed
        self.transform = transform
        self.transform_ae = transform_ae
        self.img_paths = self._list_images(root_dir)   # Collect all *.png or *.jpg
        self.skip_background = skip_background

    def _list_images(self, directory):
        exts = ('.jpg', '.png', '.jpeg', '.bmp', '.tif')
        all_files = []
        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.lower().endswith(exts):
                    all_files.append(os.path.join(root, f))
        return all_files

    def __len__(self):
        return len(self.img_paths)

    def estimate_background(self, image, thr_low=30, thr_high=220):
        """
        Similar to CDO's background estimation. 
        image: [H, W, 3], float range [0..255].
        """
        gray = np.mean(image, axis=2)
        bkg_msk_high = (gray > thr_high).astype(np.uint8)
        bkg_msk_low = (gray < thr_low).astype(np.uint8)
        bkg_msk = cv2.bitwise_or(bkg_msk_high, bkg_msk_low)
        bkg_msk = cv2.medianBlur(bkg_msk, 7)
        kernel = np.ones((19, 19), np.uint8)
        bkg_msk = cv2.dilate(bkg_msk, kernel)
        return bkg_msk

    def augment_image(self, image):
        """
        Inject random noise patches, returning (augmented_image, patch_mask).
        image: [H, W, 3] float32
        """
        h, w, _ = image.shape
        noise_image = np.random.randint(0, 255, size=(h, w, 3)).astype(np.float32)
        patch_mask = np.zeros((h, w), dtype=np.float32)

        # Optional: skip background
        if self.skip_background:
            bkg_msk = self.estimate_background(image, thr_low=30, thr_high=220)
        else:
            bkg_msk = np.zeros((h, w), dtype=np.uint8)

        patch_number = np.random.randint(0, 5)
        for _ in range(patch_number):
            for _try in range(200):
                patch_dim1 = np.random.randint(h // 40, h // 10)
                patch_dim2 = np.random.randint(w // 40, w // 10)
                center_dim1 = np.random.randint(patch_dim1, h - patch_dim1)
                center_dim2 = np.random.randint(patch_dim2, w - patch_dim2)

                # If skip_background, ensure center is not in background region
                if self.skip_background and bkg_msk[center_dim1, center_dim2] > 0:
                    continue

                min_d1 = max(0, center_dim1 - patch_dim1)
                max_d1 = min(h, center_dim1 + patch_dim1)
                min_d2 = max(0, center_dim2 - patch_dim2)
                max_d2 = min(w, center_dim2 + patch_dim2)

                patch_mask[min_d1:max_d1, min_d2:max_d2] = 1.0
                break

        augmented_image = image.copy()
        augmented_image[patch_mask > 0] = noise_image[patch_mask > 0]
        return augmented_image, patch_mask

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # 1) Load original image as np.float32
        pil_img = Image.open(img_path).convert('RGB')
        np_img = np.array(pil_img).astype(np.float32)  # shape: (H, W, 3)

        # 2) Possibly inject noise patches for the 'student' image
        if self.perturbed:
            aug_img, mask = self.augment_image(np_img)
        else:
            aug_img = np_img
            mask = np.zeros((np_img.shape[0], np_img.shape[1]), dtype=np.float32)

        # Convert to PIL for transforms
        pil_aug_img_st = Image.fromarray(aug_img.astype(np.uint8))

        # 3) Create an "AE" image with color jitter from the *original* PIL image
        #    so that it does NOT get the noise patches. 
        if self.transform_ae is not None:
            # transform_ae is typically a PIL-based transform
            # (e.g., ColorJitter) => returns a PIL or transforms a PIL
            pil_img_ae = self.transform_ae(pil_img)
        else:
            # fallback: no color jitter
            pil_img_ae = pil_img

        # 4) Now apply final resizing/normalization (transform) to both
        if self.transform is not None:
            image_st = self.transform(pil_aug_img_st)  # [3, H, W]
            image_ae = self.transform(pil_img_ae)      # [3, H, W]
        else:
            image_st = transforms.ToTensor()(pil_aug_img_st)
            image_ae = transforms.ToTensor()(pil_img_ae)

        # 5) Convert mask to tensor [1, H, W]
        pil_mask = Image.fromarray((mask * 255.0).astype(np.uint8))
        mask_t = transforms.ToTensor()(pil_mask)

        # Label = 0 if normal (typical for MVTec). Adapt if you have real anomalies.
        label = 0
        name = os.path.splitext(os.path.basename(img_path))[0]

        # 6) Return all four items:
        # (student image, autoencoder image, mask, image_name)
        return image_st, image_ae, mask_t, name
