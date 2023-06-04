import torch
import torch.nn as nn
import torch.nn.functional as F

from net.unit.base import ConvBNReLU, Res, ConvBNReLU3D, homo_warping
from net.unit import depthhypos

# from checkgrid import grads, save_grad


class RefineNet(nn.Module):
    def __init__(self,
                 ngroups,
                 ):
        super(RefineNet, self).__init__()
        self.ngroups = ngroups

        # (B, C, D, H, W) -> (B, 1, D, H, W)
        self.weight_conv = nn.Sequential(
            ConvBNReLU3D(ngroups, 1, 1, 1, 0),
            nn.Conv3d(1, 1, 1, 1, 0, ),
            nn.Sigmoid(),
        )

        print('{} parameters: {}'
              .format(self._get_name(), sum([p.data.nelement() for p in self.parameters()])))

    def forward(self, features, ref_proj, src_projs, depth_hypos, ):
        D = depth_hypos.shape[1]
        ref_feature, src_features = features[0], features[1:]  # (B,C,H,W),(nviews-1)*（B,C,H,W）

        B, C, H, W = ref_feature.shape
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, D, 1, 1)  # （B,C,D,H,W）
        # convert to unit vector
        ref_volume_ = F.softmax(ref_volume, dim=1)
        ref_volume_weight = F.softmax(ref_volume.view(B, self.ngroups, C // self.ngroups, D, H, W), dim=2)

        volume_sum, weight_sum = 0.0, 0.0
        for src_fea, src_proj in zip(src_features, src_projs):
            # torch.cuda.empty_cache()
            volume = homo_warping(src_fea, src_proj, ref_proj, depth_hypos)
            volume_weight = F.softmax(volume.view(B, self.ngroups, C // self.ngroups, D, H, W), dim=2).detach()
            volume_tmp = torch.sum(volume_weight * ref_volume_weight, dim=2).detach()
            weight = self.weight_conv(volume_tmp).squeeze(1)
            # weight.register_hook(save_grad('weight'))

            volume = torch.sum(F.softmax(volume.view(B, C, D, H, W), dim=1) * ref_volume_.view(B, C, D, H, W), dim=1)
            weight_sum += weight
            volume_sum += weight * volume

            # del volume, weight
        cost_volume = volume_sum / weight_sum

        return F.softmax(cost_volume, dim=1).squeeze(1)

