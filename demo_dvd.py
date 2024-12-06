import os
import argparse

import cv2
import torch
import torch.nn as nn
import numpy as np

from archs.DVD_arch import NSDNGAN
from pre_dehazing.network.dehaze_net import ResnetGenerator, DCPDehazeGenerator


@torch.no_grad()
def pre_process(
    image: np.array,
    device: str,
    dcp_generator: DCPDehazeGenerator,
    dehaze_model: ResnetGenerator,
    scale: int = 2,
    cf_patch_cropsize: int = 512,
) -> torch.Tensor:
    """
    Lines Referenced from:
    - research/dehaze/DVD/data/video_GoPro_hazy_dataset.py

    Parameters referenced from
    - options/test/test_DVD.yml
    - cf_cropsize=512, scale=2

    :param image: Input image, BGR format read with OpenCV
    :param device: Device to use for DCP & SI Dehaze
    :param dcp_generator: Dark Channel Prior Generator
    :param dehaze_model: Model to apply to dehaze the image
    :returns: torch.Tensor, [1, 3, hf_patch_cropsize, hf_patch_cropsize] = [1, 3, 256, 256]
    """
    # Line 114
    image = image.astype(np.float32) / 255.0

    # Line 131
    hf_patch_cropsize = cf_patch_cropsize // scale
    image = cv2.resize(image, (hf_patch_cropsize, hf_patch_cropsize))

    # Line 143
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image.transpose(2, 0, 1))
    image = image.float()

    # Line 150-161
    image = image.unsqueeze(dim=0)
    image = image.to(device)

    image = dcp_generator(image)

    image = dehaze_model(image)
    image = (image + 1) / 2
    # image = image.squeeze(dim=0)

    return image


def post_process(output_tensor: torch.Tensor):
    output_tensor = output_tensor.squeeze()
    output_image = output_tensor.cpu().numpy().transpose(1, 2, 0)
    output_image = (output_image * 255).clip(0, 255).astype(np.uint8)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    return output_image


if __name__ == "__main__":
    video_path = "/Users/shaun/datasets/image_enhancement/dehaze/DVD/DrivingHazy/29_hazy_video.mp4"
    device = "cpu"
    model_weights_path = 'checkpoint/net_g_latest.pth'
    si_dehaze_weights_path = 'pre_dehazing/models/remove_hazy_model_256x256.pth'
    flow_model_path = 'pretrained/spynet_sintel_final-3d2a1287.pth'

    # Initialize DVD Model
    model = NSDNGAN(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_frame=2,
        deformable_groups=8,
        num_extract_block=5,
        num_reconstruct_block=10,
        center_frame_idx=None,
        hr_in=False,
        with_tsa=True,
        spynet_path=flow_model_path,
    )
    weights = torch.load(model_weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights["params"])
    model.to(device)
    model.eval()

    # Initialize Single Image Dehaze Model
    si_dehaze_model = ResnetGenerator(
        input_nc=3,
        output_nc=3,
        norm_layer=nn.InstanceNorm2d,
    )
    si_dehaze_weights = torch.load(si_dehaze_weights_path, map_location="cpu", weights_only=True)
    si_dehaze_model.load_state_dict(si_dehaze_weights)
    si_dehaze_model.to(device)
    si_dehaze_model.eval()

    # Dark Channel Prior Generator
    dcp_generator = DCPDehazeGenerator().to(device)
    dcp_generator.eval()

    cap = cv2.VideoCapture(video_path)

    frame_bank = []

    while True:
        frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)

        ret, frame = cap.read()
        if not ret:
            break

        # pre_processed_frame = [1, 3, 256, 256]
        dehazed_frame = pre_process(frame, device, dcp_generator, si_dehaze_model)

        if frame_no == 0:
            frame_bank = [dehazed_frame, dehazed_frame]
        else:
            frame_bank.append(dehazed_frame)
            frame_bank.pop(0)

        dehazed_frame_np_01 = dehazed_frame.cpu().squeeze().permute(1, 2, 0).numpy().astype(np.float32).clip(0, 1)[::, ::, ::-1]
        cv2.imshow("dehazed_frame", dehazed_frame_np_01)

        assert len(frame_bank) == 2, "Model requires 2 frames for inference"

        # Make input tensor => [1, 2, 3, 256, 256] = [b, t, c, h, w]
        input_tensor = torch.concat(frame_bank, dim=0)
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor.to(device)

        with torch.no_grad():
            model_output = model(input_tensor)
            # ========== Model Output ================
            #               out = torch.Size([1, 3, 512, 512])
            #    aligned_frames = torch.Size([1, 1, 3, 256, 256])
            #          flow_vis = torch.Size([1, 256, 256, 2])
            #         nbr_fea_l = torch.Size([1, 64, 256, 256])
            #   aligned_ref_fea = torch.Size([1, 2, 64, 256, 256])
            out = model_output[0]

        out_image = post_process(out)
        cv2.imshow("out_image", out_image)
        key = cv2.waitKey(1)
        if key & 255 == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
