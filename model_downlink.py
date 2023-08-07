# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 23:16:31 2023

@author: panhao
"""

import torch
import torch.nn as nn
import scipy.constants as C
import numpy as np


class Model(nn.Module):

    def __init__(self, frequency=60e9, antenna_num=8, antenna_space=0.5, metasurface_len=20, metasurface_space=0.5, antenna_metasurface_distance=0.02, start_angle=0, end_angle=80, angle_num=9, long_distance=100, w1=10, w2=10, device='cpu'):
        super(Model, self).__init__()

        self.frequency = frequency
        wave_length = C.c / frequency
        self.wave_length = wave_length
        self.antenna_num = antenna_num
        antenna_space = wave_length*antenna_space
        self.antenna_space = antenna_space
        self.metasurface_len = metasurface_len
        metasurface_space = wave_length * metasurface_space
        self.metasurface_space = metasurface_space

        self.antenna_metasurface_distance = antenna_metasurface_distance
        
        result_az = torch.linspace(0, 0, 1, device=device)/180*np.pi
        result_el = torch.linspace(start_angle, end_angle, angle_num, device=device)/180*np.pi
        result_az_pad = result_az.unsqueeze(
            1).expand(-1, len(result_el)).flatten()
        result_el_pad = result_el.unsqueeze(
            0).expand(len(result_az), -1).flatten()
        out_angle = torch.stack((result_az_pad, result_el_pad), dim=-1)

        self.out_angle = out_angle

        show_az = torch.linspace(0, 0, 1, device=device)/180*np.pi
        show_el = torch.linspace(-90, 90, 361, device=device)/180*np.pi
        show_az_pad = show_az.unsqueeze(1).expand(-1, len(show_el)).flatten()
        show_el_pad = show_el.unsqueeze(0).expand(len(show_az), -1).flatten()
        show_angle = torch.stack((show_az_pad, show_el_pad), dim=-1)

        self.long_distance = long_distance
        # Parameters
        self.antenna_A = torch.ones(len(out_angle),
                                    antenna_num, dtype=torch.float32, device=device)
        self.antenna_theta = nn.Parameter(
            torch.ones(len(out_angle), antenna_num, dtype=torch.float32, device=device))

        self.metasurface_A = torch.ones(
            metasurface_len**2, dtype=torch.float32, device=device)
        self.metasurface_theta = nn.Parameter(
            torch.ones(metasurface_len**2, dtype=torch.float32, device=device))

        # calc position
        metasurface_position_y = torch.linspace(
            (metasurface_len-1)*metasurface_space/2, -(metasurface_len-1)*metasurface_space/2,  metasurface_len, dtype=torch.float32, device=device).expand(metasurface_len, metasurface_len)

        metasurface_position_z = torch.linspace(
            (metasurface_len-1)*metasurface_space/2, -(metasurface_len-1)*metasurface_space/2,  metasurface_len, dtype=torch.float32, device=device).expand(metasurface_len, metasurface_len).transpose(0, 1)

        metasurface_position = torch.stack((torch.zeros(metasurface_len, metasurface_len, dtype=torch.float32, device=device),
                                            metasurface_position_y,
                                            metasurface_position_z),
                                           dim=-1)

        metasurface_position = torch.flatten(
            metasurface_position, start_dim=0, end_dim=1)

        antenna_position = torch.linspace(
            (antenna_num-1)*antenna_space/2, -(antenna_num-1)*antenna_space/2, antenna_num, dtype=torch.float32, device=device)
        antenna_position = torch.stack((self.antenna_metasurface_distance * torch.ones(antenna_num, device=device),
                                        antenna_position,
                                        torch.zeros(
                                            antenna_num, device=device),
                                        ),
                                       dim=-1)

        view_position = torch.stack((torch.cos(out_angle[:, 1])*long_distance,
                                    torch.sin(
                                        out_angle[:, 1])*torch.cos(out_angle[:, 0])*long_distance,
                                    torch.sin(out_angle[:, 1])*torch.sin(out_angle[:, 0])*long_distance),
                                    dim=-1)
        show_position = torch.stack((torch.cos(show_angle[:, 1])*long_distance,
                                    torch.sin(
                                        show_angle[:, 1])*torch.cos(show_angle[:, 0])*long_distance,
                                    torch.sin(show_angle[:, 1])*torch.sin(show_angle[:, 0])*long_distance),
                                    dim=-1)
        dist_func = nn.PairwiseDistance(p=2)
        dist_metasurface = dist_func(
            metasurface_position.unsqueeze(0).expand(antenna_num, -1, -1),
            antenna_position.unsqueeze(1).expand(-1, metasurface_len**2,  -1),
        )
        dist_view = dist_func(
            view_position.unsqueeze(0).expand(metasurface_len**2, -1, -1),
            metasurface_position.unsqueeze(1).expand(-1, len(out_angle), -1)
        )
        dist_show = dist_func(
            show_position.unsqueeze(0).expand(metasurface_len**2, -1, -1),
            metasurface_position.unsqueeze(1).expand(-1, len(show_angle), -1)
        )
        air_A = self.wave_length * \
            (torch.ones(antenna_num, metasurface_len **
             2, device=device) / dist_metasurface**2)
        air_theta = 2*torch.pi*dist_metasurface/wave_length
        view_A = self.wave_length*(torch.ones(
            metasurface_len**2, len(out_angle), device=device) / dist_view**2)
        view_theta = 2*torch.pi*dist_view/wave_length
        show_A = self.wave_length*(torch.ones(
            metasurface_len**2, len(show_angle), device=device) / dist_show**2)
        show_theta = 2*torch.pi*dist_show/wave_length
        self.air_spread = air_A * torch.exp(air_theta*1j)
        self.view_spread = view_A * \
            torch.exp(view_theta*1j)
        self.show_spread = show_A * \
            torch.exp(show_theta*1j)
        
        self.w1=w1
        self.w2=w2

    def forward(self):
        metasurface = self.metasurface_A * \
            torch.exp(1j*self.metasurface_theta) # 400x1
        metasurface_out = torch.matmul(self.view_spread.T, torch.diag(metasurface)) # 1x400 x 400x400
        # print(self.air_spread.T.shape)
        antenna =  torch.matmul(metasurface_out , self.air_spread.T) # 1x400 x 400x6
        result = torch.sum(torch.abs(antenna)) + 50*torch.diag(antenna).sum()
        return -result


        # antenna = self.antenna_A * \
        #     torch.exp(1j*self.antenna_theta)
        # metasurface_in = torch.matmul(antenna, self.air_spread)
        # metasurface = self.metasurface_A * \
        #     torch.exp(1j*self.metasurface_theta)
        # metasurface_out = metasurface_in * metasurface
        # result = torch.matmul(metasurface_out, self.view_spread).abs()
        # loss1 = - torch.diag(result).sum()
        # loss2 = - torch.diag(result).min()
        # loss3 = (result - torch.diag(torch.diag(result))).norm('fro')
        # return self.w1*loss1+self.w2*loss2+loss3, result

    def show(self):
        antenna = self.antenna_A * \
            torch.exp(1j*self.antenna_theta)
        metasurface_in = torch.matmul(antenna, self.air_spread)
        metasurface = self.metasurface_A * \
            torch.exp(1j*self.metasurface_theta)
        metasurface_out = metasurface_in * metasurface
        result = 20 * \
            torch.log10(torch.matmul(metasurface_out, self.show_spread).abs())
        return result