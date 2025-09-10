import argparse

from util.tools import load_yaml


def str2bool(ipt):
    return True if ipt.lower() == 'true' else False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', default="../scripts/config/vit-l-14/ovs_voc21.yaml",
                        help='path to configuration file.',)

    """ mainly used for ablation study """
    parser.add_argument('--fuse-feature', type=str2bool, default=True,
                        help='fusion feature if True', )
    parser.add_argument('--refinement', type=str, default="SFSA",
                        choices=["mean", "SFSA", "selection", 
                        "dino-b8", "dino-s8", "dino-s16", "dino-b16", 
                        "dinov2-s14", "dinov2-b14", "dinov2-l14", "dinov2-g14", 
                        "None"],
                        help='use diffusion model refines the segmentation maps by [mean, SFSA, selection, None]', )
    parser.add_argument('--attention-idx', type=int, default=1,
                        help='using refinement with selection will be work', )
    parser.add_argument('--attn-type', type=str, default="fused-attn",
                        choices=["fused-attn", "q-q", "k-k", "v-v", "identity", "vanilla", "proxy"],
                        help='attention type in the final layer in [fused-attn, q-q, k-k, v-v, identity, vanilla]', )
    parser.add_argument('--time-step', type=int, default=45,
                        help='time-step for stable diffusion model', )
    parser.add_argument('--sd-version', type=str, default="v2.1",
                        help='version of stable diffusion model', )
    parser.add_argument('--size', type=int, default=None, help='short-size', )
    parser.add_argument('--log-path', type=str, default="log", help='path to save', )

    parser.add_argument('--model-name', type=str, default=None,
                        choices=["ViT-B/16", "ViT-L/14", "ViT-H/14"],
                        help='clip model in [ViT-B/16, ViT-L/14, ViT-H/14]', )
    parser.add_argument('--logit-scale', type=int, default=None, help='logit scaling factor', )

    parser.add_argument('--sliding_crop_size', type=int, default=None, help='slidingwindow', )

    """ mainly used for visualize """
    parser.add_argument('--save-path', type=str, default=None,
                        help='the path to save, None for no save', )

    args = parser.parse_args()
    cfg = load_yaml(args.cfg_path)
    args.__dict__.__delitem__("cfg_path")
    for k, v in args.__dict__.items():
        if v is not None:
            cfg.__dict__[k] = v

    return cfg
