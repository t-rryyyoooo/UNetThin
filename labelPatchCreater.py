import argparse
import sys
from importlib import import_module
import torch
import re
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from functions import cropping, rounding, padding, caluculatePaddingSize, getImageWithMeta
from tqdm import tqdm

class LabelPatchCreater():
    """
    This class create labl patch, ex.512-512-32 
    In 13 organs segmentation, unlike kidney cancer segmentation,
    simpleITK shape : [sagittal, coronal, axial]
    numpy shape : [axial, saggital, coronal]
    In this class we use simpleITK method to create patch, so, pay attention to shape.

    """
    def __init__(self, label, patch_size, plane_size, overlap, num_rep, mask=None, is_label=True):
        self.image = label 
        self.patch_size = np.array(patch_size)
        self.plane_size = np.array(plane_size)
        self.overlap = overlap
        self.num_rep = num_rep
        self.is_label = is_label
        self.mask = mask

    def execute(self):
        """ Raw data is clipped or padded for required_shape. """
        print(self.image.GetSize())
        image_shape = np.array(self.image.GetSize())
        required_shape = np.array(self.image.GetSize())
        required_shape[0:2] = self.plane_size
        
        self.diff = required_shape - image_shape
        if (self.diff < 0).any():
            lower_crop_size = (abs(self.diff) // 2).tolist()
            upper_crop_size = [rounding(x, 1) for x in abs(self.diff) / 2]

            self.image = cropping(self.image, lower_crop_size, upper_crop_size)
            if self.mask is not None:
                self.mask= cropping(self.mask, lower_crop_size, upper_crop_size)

        else:
            lower_pad_size = (self.diff // 2).tolist()
            upper_pad_size = [rounding(x, 1) for x in self.diff / 2]

            self.image = padding(self.image, lower_pad_size, upper_pad_size)
            if self.mask is not None:
                self.mask = padding(self.mask, lower_pad_size, upper_pad_size)


        """pad in axial direction. """
        self.slide = self.patch_size // np.array((1, 1, self.overlap))
        self.axial_lower_pad_size, self.axial_upper_pad_size = caluculatePaddingSize(np.array(self.image.GetSize()), self.patch_size, self.patch_size, self.slide)
        self.image = padding(self.image, self.axial_lower_pad_size[0].tolist(), self.axial_upper_pad_size[0].tolist())
        if self.mask is not None:
            self.mask= padding(self.mask, self.axial_lower_pad_size[0].tolist(), self.axial_upper_pad_size[0].tolist())

        print(self.image.GetSize())

        
        """ Make patch. """
        _, _, self.z_length = self.image.GetSize()
        total = self.z_length // self.slide[2]
        self.patch_list = []
        self.patch_array_list = []
        with tqdm(total=total, desc="Clipping images...", ncols=60) as pbar:
            for z in range(0, self.z_length, self.slide[2]):
                z_slice = slice(z, z + self.patch_size[2])
                if self.mask is not None:
                    patch_mask = sitk.GetArrayFromImage(self.mask[:, :, z_slice])
                    if (patch_mask == 0).all():
                        pbar.update(1)
                        continue

                patch = self.image[:,:, z_slice]
                patch.SetOrigin(self.image.GetOrigin())
                
                patch_array = sitk.GetArrayFromImage(patch)

                for _ in range(self.num_rep):
                    self.patch_list.append(patch)
                    self.patch_array_list.append(patch_array)
                
                pbar.update(1)

    def output(self, kind):
        if kind == "Array":
            return self.patch_array_list
        elif kind == "Image":
            return self.patch_list
        else:
            print("[ERROR] kind must be Image/Array.")
            sys.exit()

    def restore(self, predict_array_list):
        predicted_array = np.zeros(self.image.GetSize()[::-1])
        """ Make patch. """
        total = self.z_length // self.slide[2]
        with tqdm(total=total, desc="Restoring images...", ncols=60) as pbar:
            for z, predict_array in zip(range(0, self.z_length, self.slide[2]), predict_array_list):
                z_slice = slice(z, z + self.patch_size[2])
                predicted_array[z_slice, ...] = predict_array

                pbar.update(1)

        predicted = getImageWithMeta(predicted_array, self.image)
        predicted = cropping(predicted, self.axial_lower_pad_size[0].tolist(), self.axial_upper_pad_size[0].tolist())

        """ Raw data is clipped or padded for required_shape. """
        if (self.diff > 0).any():
            lower_crop_size = (abs(self.diff) // 2).tolist()
            upper_crop_size = [rounding(x, 1) for x in abs(self.diff) / 2]

            predicted = cropping(predicted, lower_crop_size, upper_crop_size)
            

        else:
            lower_pad_size = (abs(self.diff) // 2).tolist()
            upper_pad_size = [rounding(x, 1) for x in abs(self.diff) / 2]

            predicted = padding(predicted, lower_pad_size, upper_pad_size)

        return predicted

    def save(self, save_path, kind="Array"):
        if self.is_label:
            name = "label_"
        else:
            name = "image_"

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        if kind == "Array":
            length = len(self.patch_array_list)
            with tqdm(total=length, desc="Saving image_arrays...", ncols=60) as pbar:
                for i, patch_array in enumerate(self.patch_array_list):
                    path = save_path / (name + "{}.npy".format(str(i).zfill(3)))
                    np.save(str(path), patch_array)
                    pbar.update(1)
    
        elif kind == "Image":
            length = len(self.patch_list)
            with tqdm(total=length, desc="Saving images...", ncols=60) as pbar:
                for i, patch in enumerate(self.patch_list):
                    path = save_path / (name + "{}.mha".format(str(i).zfill(3)))
                    sitk.WriteImage(patch, str(path), True)
                    pbar.update(1)

        else:
            print("[ERROR] kind must be Array/Image.")
            sys.exit()


if __name__ == "__main__":
    args = parseArgs()
    main(args)
    

