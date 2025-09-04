from basicsr.utils.registry import MODEL_REGISTRY, LOSS_REGISTRY
from .hat_model import HATModel

@MODEL_REGISTRY.register()
class HATMixedLossModel(HATModel):

    def init_training_settings(self):
        super().init_training_settings()

        train_opt = self.opt['train']

        if train_opt.get('mse_opt'):
            self.cri_mse = LOSS_REGISTRY.get(train_opt['mse_opt'].pop('type'))(**train_opt['mse_opt']).to(self.device)
        else:
            self.cri_mse = None

        if train_opt.get('ssim_opt'):
            self.cri_ssim = LOSS_REGISTRY.get(train_opt['ssim_opt'].pop('type'))(**train_opt['ssim_opt']).to(self.device)
        else:
            self.cri_ssim = None

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = {}

        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        if self.cri_mse:
            l_mse = self.cri_mse(self.output, self.gt)
            l_total += l_mse
            loss_dict['l_mse'] = l_mse

        if self.cri_ssim:
            l_ssim = self.cri_ssim(self.output, self.gt)
            l_total += l_ssim
            loss_dict['l_ssim'] = l_ssim

        loss_dict['l_total'] = l_total

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)