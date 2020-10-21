import argparse
from importlib import import_module
import torch
import re
import SimpleITK as sitk
import numpy as np
from thinPatchCreater import ThinPatchCreater
from functions import getSizeFromString

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", help="$HOME/Desktop/data/Abdomen/case_00/imaging.nii.gz")
    parser.add_argument("label_path", help="$HOME/Desktop/data/Abdomen/case_00/segmentation.nii.gz")
    parser.add_argument("save_path", help="$HOME/Desktop/data/patch/32-512-512/label/case_00")
    parser.add_argument("--mask_path", default=None)
    parser.add_argument("--image_patch_width", default=8, type=int)
    parser.add_argument("--label_patch_width", default=8, type=int)
    parser.add_argument("--plane_size", default="512-512")
    parser.add_argument("--overlap", type=int, default=1)

    args = parser.parse_args()

    return args

def main(args):
    plane_size = getSizeFromString(args.plane_size, digit=2)

    image = sitk.ReadImage(args.image_path)
    label = sitk.ReadImage(args.label_path)
    
    if args.mask_path is not None:
        mask = sitk.ReadImage(args.mask_path)
    else:
        mask = None

    tpc = ThinPatchCreater(
            image = image, 
            label = label,
            image_patch_width = args.image_patch_width,
            label_patch_width = args.label_patch_width,
            plane_size = plane_size, 
            overlap = args.overlap,
            mask = mask
            )

    tpc.execute()

    """
    array_list = tpc.output("Array")
    res = tpc.restore(array_list)
    print(res.GetDirection(), res.GetOrigin(), res.GetSpacing())
    from functions import DICE
    dice = DICE(sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(res))
    print(dice)
    """

    tpc.save(args.save_path)


if __name__ == "__main__":
    args = parseArgs()
    main(args)
    

