import cv2

import numpy as np


def assemble_extrinsics(rotation_matrix, tvec):
    '''
    assembles the 4x4 rotation matrix given the rotation
    matrix and translation vector
    '''
    extrinsics = np.zeros((4, 4))
    extrinsics[:3, :3] = rotation_matrix
    extrinsics[:3, 3] = np.array(tvec).flatten()
    extrinsics[3, 3] = 1

    return extrinsics


def disassemble_extrinsics(extrinsics): 
    ''' 
    returns the rotation and translation vectors 
    given the extrinsics matrix
    '''
    extrinsics = np.array(extrinsics)
    rotation_matrix = extrinsics[:3, :3]
    rvec, _ = cv2.Rodrigues(rotation_matrix)
    tvec = extrinsics[:3, 3]

    return rvec, tvec