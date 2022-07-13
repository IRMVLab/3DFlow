import datetime
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

import cmd_args
import datasets
import transforms
from evaluation_utils import evaluate_2d_mask, evaluate_3d_mask
from main_utils import *
from models_occlusion import ThreeDFlow, multiScaleLoss
from models_occlusion_kitti import ThreeDFlow_Kitti
from utils import geometry


def main():

    # import ipdb; ipdb.set_trace()
    if "NUMBA_DISABLE_JIT" in os.environ:
        del os.environ["NUMBA_DISABLE_JIT"]

    global args
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    os.environ["CUDA_VISIBLE_DEVICES"] = (
        args.gpu if args.multi_gpu is None else "0,1,2,3"
    )

    """CREATE DIR"""
    experiment_dir = Path("./Evaluate_experiment/")
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(
        str(experiment_dir)
        + "/%sFlyingthings3d-" % args.model_name
        + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    )
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath("checkpoints/")
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath("logs/")
    log_dir.mkdir(exist_ok=True)
    os.system("cp %s %s" % ("models_occlusion.py", log_dir))
    os.system("cp %s %s" % ("pointconv_util.py", log_dir))
    os.system("cp %s %s" % ("evaluate_occlusion.py", log_dir))
    os.system("cp %s %s" % ("config_evaluate_occlusion.yaml", log_dir))

    """LOG"""
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(
        str(log_dir) + "evaluate_%s_sceneflow.txt" % args.model_name
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(
        "----------------------------------------TRAINING----------------------------------"
    )
    logger.info("PARAMETER ...")
    logger.info(args)

    blue = lambda x: "\033[94m" + x + "\033[0m"

    val_dataset = datasets.__dict__[args.dataset](
        train=False,
        transform=transforms.ProcessData(
            args.data_process, args.num_points, args.allow_less_points
        ),
        num_points=args.num_points,
        data_root=args.data_root,
    )
    logger.info("val_dataset: " + str(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32)),
    )

    if args.dataset == "Kitti_Occlusion":
        model = ThreeDFlow_Kitti(args.is_training)
    else:
        model = ThreeDFlow(args.is_training)

    # load pretrained model
    pretrain = args.ckpt_dir + args.pretrain
    model.load_state_dict(torch.load(pretrain))
    print("load model %s" % pretrain)
    logger.info("load model %s" % pretrain)

    model.cuda()

    epe3ds = AverageMeter()
    acc3d_stricts = AverageMeter()
    acc3d_relaxs = AverageMeter()
    outliers = AverageMeter()

    epe2ds = AverageMeter()
    acc2ds = AverageMeter()

    total_loss = 0
    total_seen = 0
    total_epe = 0

    for i, data in tqdm(enumerate(val_loader, 0), total=len(val_loader), smoothing=0.9):
        pos1, pos2, norm1, norm2, flow, mask = data

        # move to cuda
        pos1 = pos1.cuda().float()
        pos2 = pos2.cuda().float()
        norm1 = norm1.cuda().float()
        norm2 = norm2.cuda().float()
        flow = flow.cuda().float()
        mask = mask.unsqueeze(2).cuda().float()

        model = model.eval()
        with torch.no_grad():
            pred_flows, gt_flows, pc1, pc2, raw_pc1, raw_pc2, _mask = model(
                pos1, pos2, norm1, norm2, flow, mask
            )

            loss = multiScaleLoss(pred_flows, gt_flows, _mask)

            full_flow = pred_flows[0].permute(0, 2, 1)
            epe3d = torch.norm(full_flow - gt_flows[0].permute(0, 2, 1), dim=2).mean()

        total_loss += loss.cpu().data * args.batch_size
        total_epe += epe3d.cpu().data * args.batch_size
        total_seen += args.batch_size

        pc1_np = raw_pc1.cpu().numpy()
        pc2_np = raw_pc2.cpu().numpy()
        mask = _mask[0].cpu().numpy()
        sf_np = (gt_flows[0].permute(0, 2, 1)).cpu().numpy()
        pred_sf = full_flow.cpu().numpy()

        EPE3D, acc3d_strict, acc3d_relax, outlier = evaluate_3d_mask(
            pred_sf, sf_np, mask
        )

        epe3ds.update(EPE3D)
        acc3d_stricts.update(acc3d_strict)
        acc3d_relaxs.update(acc3d_relax)
        outliers.update(outlier)

        # 2D evaluation metrics
        flow_pred, flow_gt = geometry.get_batch_2d_flow(
            pc1_np, pc1_np + sf_np, pc1_np + pred_sf, [[]]
        )

        EPE2D, acc2d = evaluate_2d_mask(flow_pred, flow_gt, mask)

        epe2ds.update(EPE2D)
        acc2ds.update(acc2d)

    mean_loss = total_loss / total_seen
    mean_epe = total_epe / total_seen
    str_out = "%s mean loss: %f mean epe: %f" % (blue("Evaluate"), mean_loss, mean_epe)
    print(str_out)
    logger.info(str_out)

    res_str = (
        " * EPE3D {epe3d_.avg:.4f}\t"
        "ACC3DS {acc3d_s.avg:.4f}\t"
        "ACC3DR {acc3d_r.avg:.4f}\t"
        "Outliers3D {outlier_.avg:.4f}\t"
        "EPE2D {epe2d_.avg:.4f}\t"
        "ACC2D {acc2d_.avg:.4f}".format(
            epe3d_=epe3ds,
            acc3d_s=acc3d_stricts,
            acc3d_r=acc3d_relaxs,
            outlier_=outliers,
            epe2d_=epe2ds,
            acc2d_=acc2ds,
        )
    )

    print(res_str)
    logger.info(res_str)


if __name__ == "__main__":
    main()

