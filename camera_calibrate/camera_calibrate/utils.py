import numpy as np


def invert_homogeneous(H):
    R = H[:3, :3]          # rotation part (3×3)
    t = H[:3, 3]           # translation (3,)
    R_T = R.T              # since R is orthonormal, R.T == R⁻¹
    H_inv = np.eye(4)
    H_inv[:3, :3] = R_T
    H_inv[:3, 3]  = -R_T @ t
    return H_inv