from metric.psnr_ssim import calculate_psnr, calculate_ssim
import matplotlib.pyplot as plt
import os
import cv2
fig_root = 'figs'
sub_dir = 'mix_loss'
file_name = 'center_94'

fig_dir = os.path.join(fig_root, sub_dir, file_name)
sample_root = 'samples'
gt_path = os.path.join(sample_root, f'{file_name}.png')
sr_hat_l1_path = os.path.join(sample_root, f'{file_name}_HAT_SRx4_finetune_custom_sumi.png')
sr_hat_mix_path = os.path.join(sample_root, f'{file_name}_train_HAT_mix_loss_SRx4_finetune.png')
save_sr_hat_l1_path = os.path.join(fig_dir, f'{file_name}_sr_hat_l1.png')
save_sr_hat_mix_path = os.path.join(fig_dir, f'{file_name}_sr_hat_mix.png')

# RGB -> Y channel
def rgb2y(img_rgb):
    return 0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]

# save with scores
def save_with_scores(img_rgb, psnr_val, ssim_val, filename, title):
    plt.figure()
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f"{title}\n{psnr_val:.2f}/{ssim_val:.4f}", y=-0.13)
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)[:, :, ::-1]
sr_hat_l1 = cv2.imread(sr_hat_l1_path, cv2.IMREAD_COLOR)[:, :, ::-1]
sr_hat_mix = cv2.imread(sr_hat_mix_path, cv2.IMREAD_COLOR)[:, :, ::-1]

gt_y = rgb2y(gt)
sr_hat_l1_y = rgb2y(sr_hat_l1)
sr_hat_mix_y = rgb2y(sr_hat_mix)

psnr_l1 = calculate_psnr(gt_y, sr_hat_l1_y)
psnr_mix = calculate_psnr(gt_y, sr_hat_mix_y)
print(f"PSNR_L1: {psnr_l1:.2f}, PSNR_MIX: {psnr_mix:.2f}")

ssim_l1 = calculate_ssim(gt_y, sr_hat_l1_y)
ssim_mix = calculate_ssim(gt_y, sr_hat_mix_y)
print(f"SSIM_L1: {ssim_l1:.4f}, SSIM_MIX: {ssim_mix:.4f}")

save_with_scores(sr_hat_l1, psnr_l1, ssim_l1, save_sr_hat_l1_path, "HAT + L1 Loss")
save_with_scores(sr_hat_mix, psnr_mix, ssim_mix, save_sr_hat_mix_path, "HAT + Mix Loss")
