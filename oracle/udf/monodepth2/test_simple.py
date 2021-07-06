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

def test_simple(model_name, input_image, pred_metric_depth,  x1, x2, y1, y2, no_cuda=False, count=0):    
    """Function to predict for a single image or folder of images
    """
    assert model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #if pred_metric_depth and "stereo" not in model_name:
        #print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained KITTI "
        #      "models. For mono-trained models, output depths will not in metric space.")

    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("oracle/udf/monodepth2/models", model_name)
    #print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    #print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    #print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    output_directory = "result/"
    image_path = "result/" + str(count) + "_disp.jpeg"
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
        if pred_metric_depth:
            name_dest_npy = os.path.join(output_directory, "{}_depth.npy".format(output_name))
            metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
            #np.save(name_dest_npy, metric_depth)
        else:
            name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            #np.save(name_dest_npy, scaled_disp.cpu().numpy())

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