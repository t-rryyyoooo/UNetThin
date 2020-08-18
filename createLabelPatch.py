import argparse
from importlib import import_module
import torch
import re
import SimpleITK as sitk
import numpy as np
from labelPatchCreater import LabelPatchCreater

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("label_path", help="$HOME/Desktop/data/Abdomen/case_00/segmentation.nii.gz")
    parser.add_argument("save_path", help="$HOME/Desktop/data/patch/32-512-512/label/case_00")
    parser.add_argument("--patch_size", default="512-512-32")
    parser.add_argument("--mask_path", default=None)
    parser.add_argument("--plane_size", default="512-512")
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--num_rep", type=int, default=1)
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--save_array", action="store_true")
    parser.add_argument("--is_label", action="store_true")

    args = parser.parse_args()

    return args

def main(args):
    matchobj = re.match("([0-9]+)-([0-9]+)-([0-9]+)", args.patch_size)
    if matchobj is None:
        print("[ERROR] Invalid patch size : {}".format(args.label_patch_size))
    patch_size = np.array([int(s) for s in matchobj.groups()])

    matchobj = re.match("([0-9]+)-([0-9]+)", args.plane_size)
    if matchobj is None:
        print("[ERROR] Invalid patch size : {}".format(args.label_patch_size))

    plane_size = [int(s) for s in matchobj.groups()]

    image = sitk.ReadImage(args.label_path)
    
    if args.mask_path is not None:
        mask = sitk.ReadImage(args.mask_path)
    else:
        mask = None

    lpc = LabelPatchCreater(
            label = image, 
            patch_size = patch_size, 
            plane_size = plane_size, 
            overlap = args.overlap,
            num_rep = args.num_rep,
            is_label = args.is_label,
            mask = mask
            )

    lpc.execute()

#array_list = lpc.output("Array")
#array_list = [ array for (i, array) in enumerate(array_list) if i%2 == 0]
#res = lpc.restore(array_list)
#print(res.GetDirection(), res.GetOrigin(), res.GetSpacing())
#from functions import DICE
#dice = DICE(sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(res))
#print(dice)

    if args.save_image:
        lpc.save(args.save_path, kind="Image")
    if args.save_array:
        lpc.save(args.save_path, kind="Array")


if __name__ == "__main__":
    args = parseArgs()
    main(args)
    

