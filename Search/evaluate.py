import torch
import cv2
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


@torch.no_grad()
def evaluate(model, arch, dataloader, device, maxIter=50):
    model.eval()
    model.to(device)
    PSNRSum = 0
    SSIMSum = 0
    imageNum = 0

    for LLow_tensor, LHigh_tensor, name in dataloader:
        if not imageNum < maxIter:
            break

        LLow = LLow_tensor.to(device)
        RLow, ILow = model(LLow, arch=arch)

        RLow_np = RLow.detach().cpu().numpy()[0]
        LHigh_np = LHigh_tensor.numpy()[0]

        # compute PSNR, SSIM
        PSNRSum += peak_signal_noise_ratio(RLow_np, LHigh_np)
        SSIMSum += structural_similarity(cv2.cvtColor(np.transpose(RLow_np, [1, 2, 0]), cv2.COLOR_BGR2GRAY),
                                         cv2.cvtColor(np.transpose(LHigh_np, [1, 2, 0]), cv2.COLOR_BGR2GRAY))
        imageNum += 1

    score = (0.9 * PSNRSum + 40 * SSIMSum) / imageNum
    return score
