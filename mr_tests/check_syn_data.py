import numpy as np

file = "src/curobo/content/assets/output/sim_dual_dummy_arm_shadow/fc/data_0/graspdata/core_bowl_3f56833e91054d2518e800f0d88f9019/floating/scale016_grasp.npy"

data = np.load(file, allow_pickle=True).item()

a = 1
