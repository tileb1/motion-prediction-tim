import torch
import torch.optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from utils.opt import Options
from utils.h36motion3d import H36motion3D
import utils.model as nnmodel
import utils.data_utils as data_utils
import utils.viz as viz


def main(opt):
    model = nnmodel.InceptionGCN(opt.linear_size, opt.dropout, num_stage=opt.num_stage, node_n=66, opt=opt)
    # If you change the path of the model, change the configuration of opt accordingly e.g. if the model
    # was trained for long term prediction, change opt.output_n to 25
    model_path_len = './checkpoint/pretrained/ckpt_main_3d_3D_in10_out10_best.pth.tar'
    ckpt = torch.load(model_path_len, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])

    # data loading
    acts = data_utils.define_actions('all')
    test_data = dict()
    for act in acts:
        test_dataset = H36motion3D(path_to_data=opt.data_dir, actions=act, input_n=opt.input_n, output_n=opt.output_n,
                                   split=1,
                                   sample_rate=opt.sample_rate)
        test_data[act] = DataLoader(
            dataset=test_dataset,
            batch_size=opt.test_batch,
            shuffle=False,
            num_workers=opt.job,
            pin_memory=True)
    dim_used = test_dataset.dim_used
    print(">>> data loaded !")

    model.eval()
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    for act in acts:
        print(act)
        if opt.output_n == 25 and act not in ['smoking', 'eating', 'discussion', 'walking']:
            continue
        for i, (inputs, all_seq) in enumerate(test_data[act]):
            print(act)
            preds = model(inputs)
            pred_exmap = all_seq.clone()
            pred_exmap[:, :, dim_used] = preds.detach().transpose(1, 2)

            for k in range(0, 8):
                plt.cla()
                figure_title = "action:{}, seq:{},".format(act, (k + 1))
                viz.plot_predictions(all_seq.numpy()[k, :, :], pred_exmap.numpy()[k, :, :], fig, ax, figure_title)


if __name__ == "__main__":
    option = Options().parse()
    main(option)
