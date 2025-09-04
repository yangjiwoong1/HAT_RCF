from metric.psnr_ssim import calculate_ssim
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
save_ssim_diff_map_path = os.path.join(fig_dir, f'{file_name}_ssim_diff_map.png')

gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
sr_hat_l1_loss = cv2.imread(sr_hat_l1_path, cv2.IMREAD_GRAYSCALE)
sr_hat_mix_loss = cv2.imread(sr_hat_mix_path, cv2.IMREAD_GRAYSCALE)

ssim_score_hat_l1_loss, ssim_map_hat_l1_loss = calculate_ssim(gt, sr_hat_l1_loss, win_size=11, sigma=1.5, full=True)
ssim_score_hat_mix_loss, ssim_map_hat_mix_loss = calculate_ssim(gt, sr_hat_mix_loss, win_size=11, sigma=1.5, full=True)
ssim_diff_map = ssim_map_hat_mix_loss - ssim_map_hat_l1_loss

# save ssim_diff_map
plt.figure()
plt.imshow(ssim_diff_map, cmap='seismic', vmin=-0.1, vmax=0.1)
plt.axis('off')
cbar = plt.colorbar()
cbar.set_ticks([-0.1, -0.05, 0, 0.05, 0.1])
plt.title("SSIM Difference Map (HAT_mix_loss - HAT_L1_loss)", y=-0.1)

os.makedirs(os.path.dirname(save_ssim_diff_map_path), exist_ok=True)
plt.savefig(save_ssim_diff_map_path, dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

plt.show()
