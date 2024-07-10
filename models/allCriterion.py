import torch.nn.functional as F
def improved_distillation_loss(student_latent, teacher_latent, student_depth, teacher_depth, alpha=0.5, T=2.0):
    latent_loss = F.kl_div(
        F.log_softmax(student_latent / T, dim=1),
        F.softmax(teacher_latent / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    depth_loss = F.smooth_l1_loss(student_depth, teacher_depth)
    return alpha * latent_loss + (1 - alpha) * depth_loss