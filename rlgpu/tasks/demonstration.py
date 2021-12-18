import numpy as np
import os

from isaacgym import gymtorch
from isaacgym import gymapi
import torch

class Demonstration():
    def __init__(self, recorder_path) -> None:
        self.dof_pos_recorder = np.loadtxt(recorder_path)
        self.dof_pos=torch.from_numpy(self.dof_pos_recorder[:, :])
        # self.gripper_flag = torch.from_numpy(self.dof_pos_recorder[19:20, :])
        self.step_size = self.dof_pos.shape[0]
        
        # self.gripper_init()

    def gripper_init(self):
        for i in range(self.step_size):
            if self.gripper_flag[0, i] == 0:
                self.dof_pos[17, i] = 0.0
                self.dof_pos[18, i] = 0.0
            if self.gripper_flag[0, i] == 1:
                self.dof_pos[17, i] = 0.02
                self.dof_pos[18, i] = -0.02

        self.dof_pos1 = self.dof_pos[0:19, :]
        self.dof_pos2 = self.dof_pos[19+1:, :]
        self.dof_pos = torch.cat((self.dof_pos1, self.dof_pos2),dim=0)

    def get_dof_pos(self, step):
        if step < self.step_size:
            return self.dof_pos[step, :]
        else:
            return self.dof_pos[self.step_size - 1, :]

if __name__ == '__main__':
    demonstration = Demonstration('/home/lohse/isaac_ws/src/isaac-gym/scripts/Isaac-drlgrasp/assets/ur_assemble/track_data/assemble_joints_0.1s/dataFile_joints.txt')
    for i in range(demonstration.step_size):
        print(demonstration.get_dof_pos(i))
