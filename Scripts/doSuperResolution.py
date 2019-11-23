#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et

import os
import sys
import time
import numpy as np
import keras

import ants
import antspynet

args = sys.argv

if len(args) != 3:
    help_message = ("Usage:  python doSuperResolution.py inputFile outputFile")
    raise AttributeError(help_message)
else:
    input_file_name = args[1]
    output_file_name = args[2]

start_time_total = time.time()

print("Reading ", input_file_name)
start_time = time.time()
input_image = ants.image_read(input_file_name)
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

dimension = len(input_image.shape)

input_image_list = list()
if dimension == 4:
    input_image_list = ants.ndimage_to_list(input_image)
elif dimension == 2:
    raise ValueError("Model for 3-D or 4-D images only.")
elif dimension == 3:
    input_image_list.append(input_image)

model = antspynet.create_deep_back_projection_network_model_3d(
  (*input_image_list[0].shape, 1),
  number_of_outputs=1, number_of_base_filters=64,
  number_of_feature_filters=256, number_of_back_projection_stages=7,
  convolution_kernel_size=(3, 3, 3), strides=(2, 2, 2),
  number_of_loss_functions=1)

print( "Loading weights file" )
start_time = time.time()
weights_file_name = "./mriSuperResolutionWeights.h5"

if not os.path.exists(weights_file_name):
    weights_file_name = antspynet.get_pretrained_network("mriSuperResolution", weights_file_name)

model.load_weights(weights_file_name)
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

number_of_image_volumes = len(input_image_list)

output_image_list = list()
for i in range(number_of_image_volumes):
    print("Applying super resolution to image", i, "of", number_of_image_volumes)
    start_time = time.time()

    input_image = ants.iMath(input_image_list[i], "TruncateIntensity", 0.0001, 0.995)
    output_sr = antspynet.apply_super_resolution_model_to_image(input_image, model, target_range=(127.5, -127.5))
    input_image_resampled = ants.resample_image_to_target(input_image, output_sr)
    output_image_list.append(antspynet.regression_match_image(output_sr, input_image_resampled, poly_order = 2))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("   (elapsed time:", elapsed_time, "seconds)")


print("Writing output image.")
if number_of_image_volumes == 1:
    ants.image_write( output_image_list[0], output_file_name)
else:
    output_image = ants.list_to_ndimage(input_image, output_image_list)
    ants.image_write(output_image, output_file_name)

end_time_total = time.time()
elapsed_time_total = end_time_total - start_time_total
print( "Total elapsed time: ", elapsed_time_total, "seconds" )