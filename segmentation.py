import SimpleITK as sitk
import numpy as np
import argparse
from functions import createParentPath, getImageWithMeta
from pathlib import Path
from labelPatchCreater import LabelPatchCreater
from tqdm import tqdm
import torch
import cloudpickle
import re


def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", help="$HOME/Desktop/data/kits19/case_00000/imaging.nii.gz")
    parser.add_argument("modelweightfile", help="Trained model weights file (*.pkl).")
    parser.add_argument("save_path", help="Segmented label file.(.mha)")
    parser.add_argument("--patch_size", default="512-512-8")
    parser.add_argument("--plane_size", default="512-512")
    parser.add_argument("--overlap", default=1, type=int)
    parser.add_argument("--num_rep", default=1, type=int)
    parser.add_argument("-g", "--gpuid", help="0 1", nargs="*", default=0, type=int)

    args = parser.parse_args()
    return args

def main(args):
    """ Get the patch size from string."""
    matchobj = re.match("([0-9]+)-([0-9]+)-([0-9]+)", args.patch_size)
    if matchobj is None:
        print("[ERROR] Invalid patch size : {}.".fotmat(args.patch_size))
        sys.exit()

    patch_size = [int(s) for s in matchobj.groups()]
    """ Get the patch size from string."""
    matchobj = re.match("([0-9]+)-([0-9]+)", args.plane_size)
    if matchobj is None:
        print("[ERROR] Invalid patch size : {}.".fotmat(args.plane_size))
        sys.exit()

    plane_size = [int(s) for s in matchobj.groups()]

    """ Slice module. """
    image = sitk.ReadImage(args.image_path)

    lpc = LabelPatchCreater(
            label = image,
            patch_size = patch_size,
            plane_size = plane_size,
            overlap = args.overlap,
            num_rep = args.num_rep,
            )

    lpc.execute()
    image_array_list = lpc.output("Array")

    use_cuda = torch.cuda.is_available() and True
    device = torch.device("cuda" if use_cuda else "cpu")
    """ Load model. """
    with open(args.modelweightfile, 'rb') as f:
        model = cloudpickle.load(f)
        model = torch.nn.DataParallel(model, device_ids=args.gpuid)

    model.eval()

    """ Segmentation module. """
    segmented_array_list = []
    for image_array in tqdm(image_array_list, desc="Segmenting images...", ncols=60):
        image_array = torch.from_numpy(image_array)[None, None, ...].to(device, dtype=torch.float)

        segmented_array = model(image_array)
        segmented_array = segmented_array.to("cpu").detach().numpy().astype(np.float)
        segmented_array = np.squeeze(segmented_array)
        segmented_array = np.argmax(segmented_array, axis=0).astype(np.uint8)

        segmented_array_list.append(segmented_array)

    """ Restore module. """
    segmented = lpc.restore(segmented_array_list)

    createParentPath(args.save_path)
    print("Saving image to {}".format(args.save_path))
    sitk.WriteImage(segmented, args.save_path, True)


if __name__ == '__main__':
    args = ParseArgs()
    main(args)
    
