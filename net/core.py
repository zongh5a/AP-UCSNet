import torch
import torch.nn.functional as F


class CoreNet(torch.nn.Module):
    def __init__(self, stages, ndepths, Backbone, Neighbors, Depth_hypo, scale,
                 Homoaggre, Regular, Regress, Refine):
        """

        @param feature: unit.backbone
        @param depthhypos: list[unit.depthhypos...]
        @param scale: unit.scale
        @param homoaggregate: list[unit.scale, unit.scale, unit.scale]
        @param regular: list[unit.regualr, unit.regualr, unit.regualr]
        @param regress: list[unit.regress.depth_regress, unit.regress.confidence_regress]
        @param refine: unit.refine
        @param ndepths: [48, 24, 8]
        """
        super(CoreNet, self).__init__()
        self.stages = stages
        self.ndepths = ndepths
        self.Backbone = Backbone
        self.Neighbors = Neighbors
        self.Depth_hypo = Depth_hypo
        self.scale = scale
        self.Homoaggre = Homoaggre
        self.Regular = Regular
        self.Refine = Refine
        self.Depth_regress, self.Confidence_regress = Regress

        print('{} parameters: {}'.format(self._get_name(), sum([p.data.nelement() for p in self.parameters()])))

    def forward(self, origin_imgs, extrinsics, intrinsics, depth_range):
        """
        predict depth
        @param origin_imgs: （B,VIEW,C,H,W） view0 is ref img
        @param extrinsics: （B,VIEW,4,4）
        @param intrinsics: （B,VIEW,3,3）
        @param depth_range: (B, 2) B*(depth_min, depth_max) dtu: [425.0, 935.0] tanks: [-, -]
        @return:
        """
        origin_imgs = torch.unbind(origin_imgs.float(), 1)  # VIEW*(B,C,H,W)

        # 0. feature extraction
        features = [self.Backbone(img) for img in origin_imgs] #views * 3 * fea

        depth, depth_hypos, prob_volume, neighbors_grid, depths = None, None, None, None, []
        # for stage, (ndepths, Neighbor, Regular) in enumerate(self.ndepths, self.Regular, self.Neighbors):
        for stage in range(self.stages):
            ndepths = self.ndepths[stage]

            # 1. get features
            feature = [fea[stage] for fea in features]

            # 2.scale intrinsic matrix & cal proj matrix
            ref_proj, src_projs = self.scale(intrinsics, extrinsics, stage)

            # 3. depth hypos
            if stage in [1, 2]:
                neighbors_grid = self.Neighbors[stage-1](features[0][stage-1].detach())
            else:
                neighbors_grid = None
            depth_hypos= self.Depth_hypo(depth, depth_range, ndepths, depth_hypos, prob_volume, neighbors_grid)

            if stage < 3:
                # 4.homo & aggrate
                cost_volume = self.Homoaggre(feature, ref_proj, src_projs, depth_hypos)
                # 5.regular
                prob_volume = self.Regular[stage](cost_volume)  # (B,D,H,W)
            else:
                # 7.
                prob_volume = self.Refine(feature, ref_proj, src_projs, depth_hypos)

            # 6.depth regress
            depth = self.Depth_regress(prob_volume, depth_hypos)
            depths.append(depth)

        if self.training:
            return {"depth": depths, }

        # 8. confidence
        confidence = self.Confidence_regress(prob_volume)

        return {"depth": depth, "confidence": confidence}


if __name__=="__main__":
    pass

