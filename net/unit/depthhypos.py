import torch
import torch.nn.functional as F
# from checkgrid import grads, save_grad

def atv_hypos(depth, depth_range, ndepths, depth_hypos, prob_volume, neighbors_grid, lamb=1.5, eps = 1e-12):

    B = depth_range.shape[0]
    # init hypos
    if depth is None:
        depth_min, depth_max = depth_range[:, 0].float(), depth_range[:, 1].float()
        depth_min, depth_max = depth_min.view(B, 1, 1, 1), depth_max.view(B, 1, 1, 1)
        depth_interval = (depth_max - depth_min) / (ndepths - 1)
        depth_hypos = depth_min.unsqueeze(1) \
                      + (torch.arange(0, ndepths, device=depth_range.device).reshape(1, -1)
                         * depth_interval.unsqueeze(1))

        return depth_hypos.view(B, ndepths, 1, 1)
    else:
        depth = depth.unsqueeze(1).detach()
        depth_hypos = depth_hypos.detach()
        prob_volume = prob_volume.detach()
        samp_variance = (depth_hypos - depth) ** 2
        exp_variance = (lamb * torch.sum(samp_variance * prob_volume, dim=1, keepdim=False) ** 0.5).unsqueeze(1)

        # In-plane weighting
        if neighbors_grid is not None:
            # if torch.isnan(neighbors_grid.min()) :
            #     print("Error neighbors_grid, not use!")
            # else:
            B, _, H, W = depth.shape
            nneighbors = neighbors_grid.shape[1]//H
            neighbors_exp = F.grid_sample(exp_variance, neighbors_grid, mode="bilinear", padding_mode="border", align_corners=False
                                          ).view(B, 1, nneighbors, H, W)    #
            exp_variance = 0.5*exp_variance+0.5*torch.mean(neighbors_exp, 2)

            # neighbors_exp, _ = torch.sort(neighbors_exp, 2)
            # exp_variance = 0.5*exp_variance+0.5*torch.mean(neighbors_exp[:, :, 1:-1, :, :], 2) #//nneighbors

            # exp_variance = torch.mean(exp_variance + neighbors_exp,2) #//nneighbors
            # neighbors_exp.register_hook(save_grad('neighbors_exp'))
            # neighbors_grid.register_hook(save_grad('neighbors_grid'))

        depth = F.interpolate(depth, scale_factor=2, mode='bilinear')
        exp_variance = F.interpolate(exp_variance, scale_factor=2, mode='bilinear')

        low_bound = -torch.min(depth, exp_variance)
        high_bound = exp_variance
        step = (high_bound - low_bound) / (ndepths - 1)

        depth_hypos = []
        for i in range(ndepths):
            depth_hypos.append(depth + low_bound + step * i + eps)

        depth_hypos = torch.cat(depth_hypos, 1)
        # assert depth_range_samples.min() >= 0, depth_range_samples.min()
        return depth_hypos
