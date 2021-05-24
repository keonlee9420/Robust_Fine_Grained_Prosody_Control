from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self, dist, gpu):
        super(Tacotron2Loss, self).__init__()

        if dist:
            self.mse_criterion  = nn.MSELoss.cuda(gpu)
            self.bce_criterion = nn.BCEWithLogitsLoss.cuda(gpu)
        else:
            self.mse_criterion  = nn.MSELoss()
            self.bce_criterion = nn.BCEWithLogitsLoss()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = self.mse_criterion(mel_out, mel_target) + \
            self.mse_criterion(mel_out_postnet, mel_target)
        gate_loss = self.bce_criterion(gate_out, gate_target)

        return mel_loss + gate_loss
