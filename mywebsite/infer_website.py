# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import numpy

from caffe2.python import workspace


from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.utils.stylizeimage as style
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils


c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-second',
        dest='image_second',
        help='second replaceble image',
        default=None,
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--method', help='to see hwat you want', default=None,type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default="yo"
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def infer_method(im):
    logger = logging.getLogger(__name__)
    #styleimage = style.style_method()
    merge_cfg_from_file("configs/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.yaml ")
    cfg.NUM_GPUS = 1
    args.weights = cache_url("DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.pkl", cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg("DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.pkl")
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.pdf')
        )
    logger.info('Processing {} -> {}'.format(im_name, out_name))
    im2 = cv2.imread("tools/iron8.jpg")
    mymethod = "back"
    timers = defaultdict(Timer)
    t = time.time()
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(
                model, im, None, timers=timers)
        if im2 is not None:
            cls_boxes2, cls_segms2, cls_keyps2, cls_bodys2 = infer_engine.im_detect_all(
                model, im2, None, timers=timers)

        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )

        if mymethod == "back":
          vis_utils.change_background(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im2[:, :, ::-1],
            im_name,
            "tools/output",
            cls_boxes,
            cls_segms,
            cls_keyps,
            cls_bodys,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2)
        elif mymethod == "iron":
            vis_utils.ironman(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im2[:, :, ::-1],
            im_name,
            args.output_dir,
            cls_boxes,
            cls_boxes2,
            cls_segms,
            cls_keyps,
            cls_bodys,
            cls_segms2,
            cls_keyps2,
            cls_bodys2,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2)
        elif mymethod == 'style_b':
            styleimage = cv2.cvtColor(numpy.array(style.stylize_img(im_name,args.image_second)),cv2.COLOR_RGB2BGR)
            resized_im = style.tensor_to_image(style.load_to_mask(im_name))
            opencvImage = cv2.cvtColor(numpy.array(resized_im), cv2.COLOR_RGB2BGR)
            print(opencvImage)
            with c2_utils.NamedCudaScope(0):
              bo,se,ke,bod = infer_engine.im_detect_all(model, opencvImage, None, timers=timers)
            vis_utils.change_background(
            opencvImage[:, :, ::-1],  # BGR -> RGB for visualization
            styleimage[:, :, ::-1],
            "stylized_img.jpg",
            args.output_dir,
            bo,
            se,
            ke,
            bod,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2)
        else:
          vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            cls_bodys,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2)



if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
