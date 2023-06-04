import torch
import torch.nn as nn
import torch.nn.functional as F

from net.unit.base import ConvBNReLU3D, homo_warping


# def homo_aggregate_by_variance(features, ref_proj, src_projs, depth_hypos):
#
#     ndepths = depth_hypos.shape[1]
#     ref_feature, src_features = features[0], features[1:]  # (B,C,H,W),(nviews-1)*（B,C,H,W）
#     # ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, ndepths, 1, 1)  # （B,C,D,H,W）
#
#     volume_sum, volume_sq_sum = 0.0, 0.0
#
#     for src_fea, src_proj in zip(src_features, src_projs):
#         # torch.cuda.empty_cache()
#         warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_hypos)
#         volume_sum = volume_sum + warped_volume
#         volume_sq_sum = volume_sq_sum + warped_volume ** 2
#         del warped_volume
#
#     cost_volume = volume_sq_sum.div_(len(src_features)).sub_(volume_sum.div_(len(src_features)).pow_(2))
#     del volume_sum, volume_sq_sum
#
#     return cost_volume



def homo_aggregate_by_variance(features, ref_proj, src_projs, depth_hypos):

    ndepths = depth_hypos.shape[1]
    ref_feature, src_features = features[0], features[1:]  # (B,C,H,W),(nviews-1)*（B,C,H,W）
    # ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, ndepths, 1, 1)  # （B,C,D,H,W）

    ref_feature = ref_feature.unsqueeze(2)
    volume_sum, volume_sq_sum = ref_feature, ref_feature**2
    for src_fea, src_proj in zip(src_features, src_projs):
        # torch.cuda.empty_cache()
        warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_hypos)
        volume_sum = volume_sum + warped_volume
        volume_sq_sum = volume_sq_sum + warped_volume ** 2
        del warped_volume

    cost_volume = volume_sq_sum.div_(len(src_features)+1).sub_(volume_sum.div_(len(src_features)+1).pow_(2))
    del volume_sum, volume_sq_sum

    return cost_volume




if __name__=="__main__":
    pass


