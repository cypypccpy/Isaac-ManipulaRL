import numpy as np
import os

from isaacgym import gymtorch
from isaacgym import gymapi
import torch

class Demostration():
    def __init__(self, recorder_path) -> None:
        self.dof_pos_recorder = np.loadtxt(recorder_path)
        self.dof_pos=torch.from_numpy(self.dof_pos_recorder[:19, :])
        self.gripper_flag = torch.from_numpy(self.dof_pos_recorder[19:, :])
        self.step_size = self.dof_pos.shape[1]
        
        self.gripper_init()

    def gripper_init(self):
        for i in range(self.step_size):
            if self.gripper_flag[0, i] == 0:
                self.dof_pos[17, i] = 0.0
                self.dof_pos[18, i] = 0.0
            if self.gripper_flag[0, i] == 1:
                self.dof_pos[17, i] = 0.02
                self.dof_pos[18, i] = -0.02

    def get_dof_pos(self, step):
        if step < self.dof_pos.shape[1]:
            return self.dof_pos[:, step]
        else:
            return self.dof_pos[:, self.dof_pos.shape[1] - 1]

if __name__ == '__main__':
    demostration = Demostration('/home/lohse/isaac_ws/src/isaac-gym/scripts/Isaac-drlgrasp/envs_test/npresult1.txt')
    for i in range(demostration.step_size):
        if demostration.get_dof_pos(i)[17] == 0.02:
            print(1)

    print(demostration.step_size)
    print(demostration.gripper_flag.shape)
