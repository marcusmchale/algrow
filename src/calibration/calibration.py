"""Interactively define parameters for layout detection and area calculation: circle colour, target hull points"""
import argparse
import logging

import numpy as np

from .app import Calibrator

logger = logging.getLogger(__name__)


def calibrate(args: argparse.Namespace):

    # calibrate on a subset of images only
    vars(args)['images'].sort()
    idx = np.unique(np.round(np.linspace(0, len(args.images) - 1, args.num_calibration)).astype(int))
    sample_image_paths = list(np.unique(np.array(args.images)[idx]))
    logger.info(f"Sample {min(args.num_calibration, len(sample_image_paths))} images for calibration")

    #from ..image_loading import ImageLoaded
    #from .loading import Points
    #from .hull_pixels import AlphaSelection
    #images = [ImageLoaded(image, args) for image in sample_image_paths]
    #points = Points(images)
    #points.pixel_to_lab, points.lab_to_pixel, points.filepath_lab_to_pixel, points.counts_all, points.counts_per_image = points.process_images()
    #points.counts_per_image['distance'] = np.inf
    #
    # alpha_selection = AlphaSelection(points, set(), args.alpha, args.delta)
    # filepath = images[0].filepath
    # lab = points.lab_to_pixel.index[0]

    #import open3d as o3d
    #def plot_o3d(points):
    #    xyz = points.counts_all[[("lab", "L"),("lab",  "a"),("lab", "b")]].to_numpy()
    #    rgb = points.counts_all[[("rgb", "r"),("rgb",  "g"),("rgb", "b")]].to_numpy()
    #    pcd = o3d.geometry.PointCloud()
    #    pcd.points = o3d.utility.Vector3dVector(xyz)
    #    pcd.colors = o3d.utility.Vector3dVector(rgb)
    #    o3d.visualization.draw_geometries([pcd])  # Uncomment this to see the plot
    #
    #import pdb
    #pdb.set_trace()

    calibrator = Calibrator(sample_image_paths, args)
    calibrator.MainLoop()

