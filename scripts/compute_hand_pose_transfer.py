import numpy as np
from mr_utils.utils_calc import posQuat2Isometry3d, quatWXYZ2XYZW, sciR


# TODO: make it general

rot_mat = sciR.from_euler("zyx", [-90, 0, -30], degrees=True).as_matrix()

print(rot_mat)
