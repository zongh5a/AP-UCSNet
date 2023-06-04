import torch
import torch.nn as nn
# from checkgrid import grads, save_grad

class Neighbors(nn.Module):
    def __init__(self,
                 in_chs,
                 neighbors = 8,
                 dilation = 2,
                 ):
        super(Neighbors, self).__init__()

        self.neighbors = neighbors
        self.dilation = dilation

        self.neighbors_conv = nn.Conv2d(in_chs, 2 * neighbors, 3, 1, dilation, dilation=dilation, bias=True)

        # nn.init.constant_(self.neighbors_conv.weight, 0.0)
        # nn.init.constant_(self.neighbors_conv.bias, 0.0)

    def forward(self, ref_feature: torch.Tensor,):
        B, C, H, W = ref_feature.shape
        device = ref_feature.device
        offset = self.neighbors_conv(ref_feature)

        offset = offset.view(B, 2 * self.neighbors, H * W)
        # offset.register_hook(save_grad('offset'))

        return self.get_grid(B, H, W, offset, device, self.neighbors, self.dilation)

    def get_grid(
            self,
            batch: int,
            height: int,
            width: int,
            offset: torch.Tensor,
            device: torch.device,
            neighbors: int,
            dilation: int,
    ) -> torch.Tensor:
        """Compute the offset for adaptive propagation or spatial cost aggregation in adaptive evaluation

        Args:
            grid_type: type of grid - propagation (1) or evaluation (2)
            batch: batch size
            height: grid height
            width: grid width
            offset: grid offset
            device: device on which to place tensor

        Returns:
            generated grid: in the shape of [batch, propagate_neighbors*H, W, 2]
        """

        if neighbors == 4:
            original_offset = [[-dilation, 0], [0, -dilation], [0, dilation], [dilation, 0]]
        elif neighbors == 8:
            original_offset = [
                [-dilation, -dilation],
                [-dilation, 0],
                [-dilation, dilation],
                [0, -dilation],
                [0, dilation],
                [dilation, -dilation],
                [dilation, 0],
                [dilation, dilation],
            ]
        elif neighbors == 9:
                original_offset = [
                    [-dilation, -dilation],
                    [-dilation, 0],
                    [-dilation, dilation],
                    [0, -dilation],
                    [0, 0],
                    [0, dilation],
                    [dilation, -dilation],
                    [dilation, 0],
                    [dilation, dilation],
                ]
        elif neighbors == 16:
            original_offset = [
                [-dilation, -dilation],
                [-dilation, 0],
                [-dilation, dilation],
                [0, -dilation],
                [0, dilation],
                [dilation, -dilation],
                [dilation, 0],
                [dilation, dilation],
            ]
            for i in range(len(original_offset)):
                offset_x, offset_y = original_offset[i]
                original_offset.append([2 * offset_x, 2 * offset_y])
        elif neighbors == 17:
            original_offset = [
                [-dilation, -dilation],
                [-dilation, 0],
                [-dilation, dilation],
                [0, -dilation],
                [0, 0],
                [0, dilation],
                [dilation, -dilation],
                [dilation, 0],
                [dilation, dilation],
            ]
            for i in range(len(original_offset)):
                offset_x, offset_y = original_offset[i]
                if offset_x != 0 or offset_y != 0:
                    original_offset.append([2 * offset_x, 2 * offset_y])
        else:
            raise NotImplementedError

        # with torch.no_grad():
        y_grid, x_grid = torch.meshgrid(
            [
                torch.arange(0, height, dtype=torch.float32, device=device),
                torch.arange(0, width, dtype=torch.float32, device=device),
            ]
        )
        y_grid, x_grid = y_grid.contiguous().view(height * width), x_grid.contiguous().view(height * width)
        xy = torch.stack((x_grid, y_grid))  # [2, H*W]
        xy = torch.unsqueeze(xy, 0).repeat(batch, 1, 1)  # [B, 2, H*W]

        xy_list = []
        for i in range(len(original_offset)):
            original_offset_y, original_offset_x = original_offset[i]
            offset_x = original_offset_x + offset[:, 2 * i, :].unsqueeze(1)
            offset_y = original_offset_y + offset[:, 2 * i + 1, :].unsqueeze(1)
            xy_list.append((xy + torch.cat((offset_x, offset_y), dim=1)).unsqueeze(2))

        xy = torch.cat(xy_list, dim=2)  # [B, 2, 9, H*W]

        del xy_list
        del x_grid
        del y_grid

        x_normalized = xy[:, 0, :, :] / ((width - 1) / 2) - 1
        y_normalized = xy[:, 1, :, :] / ((height - 1) / 2) - 1
        del xy
        grid = torch.stack((x_normalized, y_normalized), dim=3)  # [B, 9, H*W, 2]
        del x_normalized
        del y_normalized
        return grid.view(batch, len(original_offset) * height, width, 2)


