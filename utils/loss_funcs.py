from utils.constants import *


def my_mpjpe_error_p3d(outputs, all_seq, dim_used):
    dim_used_len = len(dim_used)

    outputs_p3d = outputs.transpose(1, 2)
    pred_3d = outputs_p3d.contiguous().view(-1, dim_used_len).view(-1, 3)
    targ_3d = all_seq[:, :, dim_used].contiguous().view(-1, dim_used_len).view(-1, 3)

    mean_3d_err = torch.mean(torch.norm(pred_3d - targ_3d, 2, 1))

    return mean_3d_err
