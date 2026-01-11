#from __future__ import print_function
import os,sys
import numpy as np
from sklearn.feature_extraction import image
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def generate_image_patches_db(in_directory,out_directory,patch_size=64):
  if not os.path.exists(out_directory):
      os.makedirs(out_directory)
 
  total = 2688
  count = 0  
  for split_dir in os.listdir(in_directory):
    if not os.path.exists(os.path.join(out_directory,split_dir)):
      os.makedirs(os.path.join(out_directory,split_dir))
  
    for class_dir in os.listdir(os.path.join(in_directory,split_dir)):
      if not os.path.exists(os.path.join(out_directory,split_dir,class_dir)):
        os.makedirs(os.path.join(out_directory,split_dir,class_dir))
  
      for imname in os.listdir(os.path.join(in_directory,split_dir,class_dir)):
        count += 1
        im = Image.open(os.path.join(in_directory,split_dir,class_dir,imname))
        print(im.size)
        print('Processed images: '+str(count)+' / '+str(total), end='\r')
        patches = image.extract_patches_2d(np.array(im), (64, 64), max_patches=1)
        for i,patch in enumerate(patches):
          patch = Image.fromarray(patch)
          patch.save(os.path.join(out_directory,split_dir,class_dir,imname.split(',')[0]+'_'+str(i)+'.jpg'))
  print('\n')

class PatchDataset(Dataset):
    """
    End-to-end patch-based dataset.
    Returns:
        x -> [N, 3, H, W]
        y -> image label
    """
    def __init__(self, root, transform, patch_size=64, num_patches=4):
        self.dataset = ImageFolder(root)
        self.samples = self.dataset.samples
        self.transform = transform
        self.patch_size = patch_size
        self.num_patches = num_patches

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):        
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
    
        if self.num_patches == 1:
            patches = [img]
        else:
            patches = image.extract_patches_2d(
                img,
                patch_size=(self.patch_size, self.patch_size),
                max_patches=self.num_patches
            )

        patches = torch.stack([
            self.transform(patch) for patch in patches
        ])
        
        # if idx == 0:
        #     self._visualize(img, patches)

        return patches, label
    
    def _visualize(self, img, patches): 
        n = len(patches)
        cols = min(n + 1, 5)
    
        plt.figure(figsize=(3 * cols, 3))
    
        # Original image (already HWC)
        plt.subplot(1, cols, 1)
        plt.imshow(img)
        plt.title("Original")
        plt.axis("off")
    
        for i, patch in enumerate(patches):
            plt.subplot(1, cols, i + 2)
            plt.imshow(patch.permute(1, 2, 0).cpu())
            plt.title(f"Patch {i}")
            plt.axis("off")
    
        plt.tight_layout()
        plt.show()
