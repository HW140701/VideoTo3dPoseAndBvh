# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch

from common.quaternion import qrot, qinverse
from common.utils import wrap
import h5py
from pathlib import Path


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]


def normalize_screen_coordinates_new(X, w, h):
    assert X.shape[-1] == 2

    return (X - (w / 2, h / 2)) / (w / 2, h / 2)


def image_coordinates_new(X, w, h):
    assert X.shape[-1] == 2

    # Reverse camera frame normalization
    return (X * (w / 2, h / 2)) + (w / 2, h / 2)


def image_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Reverse camera frame normalization
    return (X + [1, h / w]) * w / 2


def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R)  # Invert rotation
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t)  # Rotate and translate


def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t


def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    focal length / principal point / radial_distortion / tangential_distortion
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]  # focal lendgth
    c = camera_params[..., 2:4]  # center principal point
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2] ** 2, dim=len(XX.shape) - 1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2 ** 2, r2 ** 3), dim=len(r2.shape) - 1), dim=len(r2.shape) - 1, keepdim=True)
    tan = torch.sum(p * XX, dim=len(XX.shape) - 1, keepdim=True)

    XXX = XX * (radial + tan) + p * r2

    return f * XXX + c


def project_to_2d_linear(X, camera_params):
    """
    使用linear parameters is a little difference for use linear and no-linear parameters
    Project 3D points to 2D using only linear parameters (focal length and principal point).

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)

    return f * XX + c

# added by HuangWang
def load_camera_params(file):
    cam_file = Path(file)
    cam_params = {}
    azimuth = {
        '54138969': 70, '55011271': -70, '58860488': 110, '60457274': -100
    }
    with h5py.File(cam_file) as f:
        subjects = [1, 5, 6, 7, 8, 9, 11]
        for s in subjects:
            cam_params[f'S{s}'] = {}
            for _, params in f[f'subject{s}'].items():
                name = params['Name']
                name = ''.join([chr(c) for c in name])
                val = {}
                val['R'] = np.array(params['R'])
                val['T'] = np.array(params['T'])
                val['c'] = np.array(params['c'])
                val['f'] = np.array(params['f'])
                val['k'] = np.array(params['k'])
                val['p'] = np.array(params['p'])
                val['azimuth'] = azimuth[name]
                cam_params[f'S{s}'][name] = val

    return cam_params

# added by HuangWang
def camera2world(pose, R, T):
    """
    Args:
        pose: numpy array with shape (..., 3)
        R: numpy array with shape (3, 3)
        T: numyp array with shape (3, 1)
    """
    assert pose.shape[-1] == 3
    original_shape = pose.shape
    pose_cam = pose.copy().reshape((-1, 3)).T
    pose_world = np.matmul(R, pose_cam) + T
    pose_world = pose_world.T.reshape(original_shape)
    return pose_world