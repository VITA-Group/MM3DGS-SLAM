import torch
import numpy as np
import matplotlib.pyplot as plt

# DepthAnything init
import cv2
import torch.nn.functional as F

# from torchvision.transforms import Compose
# from depth_anything.dpt import DepthAnything
# from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


def depth_to_rgb(depth_image, min_depth=None, max_depth=None, colormap="viridis"):
    """
    Convert a raw depth image to an RGB image by mapping depths to a colormap
    """
    # Clip depth values to the specified range or use the min/max values in the image
    min_depth = min_depth if min_depth is not None else torch.min(depth_image)
    max_depth = max_depth if max_depth is not None else torch.max(depth_image)

    # Normalize depth values to the range [0, 1]
    normalized_depth = torch.clamp(
        (depth_image - min_depth) / (max_depth - min_depth), min=0, max=1
    )

    # Apply colormap to normalized depth values
    cmap = plt.get_cmap(colormap)
    colored_depth = torch.tensor(
        cmap(normalized_depth.cpu().detach().numpy())
    ).cuda()  # H,W,C
    colored_depth = colored_depth.permute(2, 0, 1)[:3, :, :]  # C,H,W

    return colored_depth


def depth_to_distance(depth_value, depth_scale):
    return 1.0 / (depth_value * depth_scale)


# TODO: get_scale_shift robust to outliers


def get_scale_shift_LS(est_depth, render_depth, mask=None, num_samples=-1):
    """
    Calculate a scale and shift that fits the depth estimate est to the rendered depth render
    using Least Squares.
    """
    render_depth = 1 / render_depth  # estimated depth is an inverse depth

    # Mask out invalid pixels
    if mask is not None:
        render_depth[~mask] = 0
    valid_depth_indices = torch.where(render_depth > 0)
    valid_depth_indices = torch.stack(valid_depth_indices, dim=1)

    # Sample num_samples pixels, or use all pixels if num_samples==-1
    if num_samples == -1:
        sampled_indices = valid_depth_indices
    else:
        indices = torch.randint(valid_depth_indices.shape[0], (num_samples,))
        sampled_indices = valid_depth_indices[indices]

    # Calculate Least Squares solution
    try:
        H = est_depth[sampled_indices[:, 0], sampled_indices[:, 1]]
        H1 = torch.ones(H.shape).to(H)
        H = torch.stack([H, H1], dim=1)
        z = render_depth[sampled_indices[:, 0], sampled_indices[:, 1]].unsqueeze(1)
        x = torch.inverse(H.t() @ H) @ (H.t() @ z)
        scale = x[0]
        shift = x[1]
    except:
        # print("non zero render_depth")
        # print(torch.count_nonzero(render_depth))
        # print("mask")
        # print(torch.count_nonzero(mask))
        # print("H: \n")
        # print(H)
        # print("sampled_indices size: \n")
        # print(sampled_indices.shape)
        # print("est_depth: \n")
        # print(est_depth)
        # print("# 0 elems in est_depth: \n")
        # print(torch.count_nonzero(est_depth == 0))
        print("get_scale_shift failed")
        plt.figure()
        plt.imshow(est_depth.detach().cpu().numpy())
        plt.colorbar()
        plt.figure()
        plt.imshow(render_depth.detach().cpu().numpy())
        plt.colorbar()
        plt.show()

    return scale, shift


def get_scale_shift(est_depth, render_depth, mask=None, num_samples=-1, method="LS"):
    return get_scale_shift_LS(est_depth, render_depth, mask, num_samples)


class MiDaS:
    def __init__(self, device):
        self.device = device
        self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        self.midas.to(self.device)
        self.midas.eval()
        for param in self.midas.parameters():
            param.requires_grad = False

        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = self.midas_transforms.dpt_transform
        self.downsampling = 1

    def estimate_depth(self, img):
        h, w = img.shape[1:3]
        norm_img = (img[None] - 0.5) / 0.5
        norm_img = F.interpolate(
            norm_img, size=(384, 512), mode="bilinear", align_corners=False
        )

        with torch.no_grad():
            prediction = self.midas(norm_img)
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=(h // self.downsampling, w // self.downsampling),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

        return prediction


# class DA:
#     def __init__(self, device):
#         self.device = device
#         self.encoder = "vits"  # can also be 'vitb' or 'vitl'
#         self.depth_anything = DepthAnything.from_pretrained(
#             "LiheYoung/depth_anything_{:}14".format(self.encoder)
#         )
#         self.depth_anything.to(self.device)
#         self.depth_anything.eval()
#
#         self.transform = Compose(
#             [
#                 Resize(
#                     width=518,
#                     height=518,
#                     resize_target=False,
#                     keep_aspect_ratio=True,
#                     ensure_multiple_of=14,
#                     resize_method="lower_bound",
#                     image_interpolation_method=cv2.INTER_LINEAR,
#                 ),
#                 NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                 PrepareForNet(),
#             ]
#         )
#
#     def estimate_depth(self, img):
#         h, w = img.shape[1:3]
#
#         # TODO: figure out preprocessing
#         image = img.clone().permute(1, 2, 0).cpu().numpy()  # C,H,W -> H,W,C
#         image = self.transform({"image": image})["image"]
#         image = torch.from_numpy(image).unsqueeze(0).to(self.device)
#
#         with torch.no_grad():
#             depth = self.depth_anything(image)
#
#             depth = F.interpolate(
#                 depth[None], (h, w), mode="bilinear", align_corners=False
#             )[0, 0]
#
#         return depth


def get_dpt(model, device):
    if model.lower() == "midas":
        return MiDaS(device)
    # elif model.lower() == "depthanything":
    #     return DA(device)
    else:
        raise ValueError(f"Unknown depth estimate model {model}")
