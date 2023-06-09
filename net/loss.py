import torch
import torch.nn.functional as F
from typing import Dict, List


class Loss(torch.nn.Module):
    def __init__(self, ) -> None:
        super(Loss,self).__init__()

    def forward(self,
                outputs: Dict,
                depth_gt: torch.Tensor,
                depth_range: torch.Tensor
                ) -> torch.Tensor:
        depths = outputs["depth"]

        # depth loss
        loss_depth = 0.0
        for depth, depth_g in zip(depths, depth_gt.values()):
            mask = depth_g > depth_range[:, 0].view(-1, 1, 1)  # depth_min
            #print(depth[mask].min(), depth[mask].max())
            loss_tmp = F.smooth_l1_loss(depth[mask], depth_g[mask], reduction='mean')
            # if torch.isnan(loss_tmp).int().sum()>0 or torch.isinf(loss_tmp).int().sum()>0:
            #     return None
            # else:
            loss_depth += loss_tmp
            
        return loss_depth


if __name__=="__main__":
    l = Loss()


