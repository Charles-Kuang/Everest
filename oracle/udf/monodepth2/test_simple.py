# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import PIL.ImageDraw as pil_draw
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import cv2

import oracle.udf.monodepth2.networks as networks
from oracle.udf.monodepth2.layers import disp_to_depth
from oracle.udf.monodepth2.utils import download_model_if_doesnt_exist
from oracle.udf.monodepth2.evaluate_depth import STEREO_SCALE_FACTOR

def test_simple(input_image,  x1, x2, y1, y2, count, feed_height, feed_width, device):
    output_directory = "/everest/result/"
    image_path = "/everest/result/" + str(count) + "_disp.jpeg"
    output_name = os.path.splitext(os.path.basename(image_path))[0]

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        # Load image and preprocess
        input_image = pil.fromarray(input_image,'RGB')
        im1 = input_image
        name_dest_im1 = os.path.join(output_directory, "{}_disp1.jpeg".format(output_name))
        name_dest_im2 = os.path.join(output_directory, "{}_disp2.jpeg".format(output_name))
        #imd = pil_draw.Draw(im1)
        #imd.rectangle([(x1, y1), (x2, y2)],  outline="red")
        #im1.save(name_dest_im1)
        #imd.save(name_dest_im2)

        original_width, original_height = input_image.size
        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        input_image = input_image.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)

        # Saving numpy file
        scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
        
        name_dest_npy = os.path.join(output_directory, "{}_depth.npy".format(output_name))
        metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
        #np.save(name_dest_npy, metric_depth)

        # Saving colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)

        name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))

        #imt = pil_draw.Draw(im)
        #imt.rectangle([(x1, y1), (x2, y2)], outline="red")
        #im.save(name_dest_im)

    #print('-> Done!')
    return_val = STEREO_SCALE_FACTOR * scaled_disp.cpu().numpy()#scaled_disp
    return metric_depth[0][0]