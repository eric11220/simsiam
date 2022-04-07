import numpy as np
import os
import pickle as pk
import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import simsiam.resnet
import simsiam.builder

from main_simsiam import test, get_features, arguments
from uniformity_utils import uniform_loss

parser = arguments()
parser.add_argument('--pretrained', default='', type=str,
                    help='path to simsiam pretrained checkpoint')
parser.add_argument('--fig-dir', default='plots', type=str,
                    help='Directory for plotted figures')
parser.add_argument('--feat-dir', default='feats', type=str,
                    help='Directory for computed features')
parser.add_argument('--mode', default='tsne', choices=['tsne', 'uniformity'],
                    help='What analysis to conduct')
args = parser.parse_args()

setting = os.path.basename(os.path.dirname(args.pretrained))
feat_path = os.path.join(args.feat_dir, f'{setting}.pkl')
fig_path = os.path.join(args.fig_dir, f'{setting}.png')

info = {}
if not os.path.exists(feat_path):
    mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    
    train_knn_dataset = datasets.CIFAR10(root='../data', train=True, transform=test_transform)
    test_dataset = datasets.CIFAR10(root='../data', train=False, transform=test_transform)
    
    train_knn_loader = torch.utils.data.DataLoader(
        train_knn_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    
    model = simsiam.builder.SimSiam(
        simsiam.resnet.ResNet18,
        args.dim, args.pred_dim, predictor_reg=args.predictor_reg)
    model = simsiam.builder.SimSiamEncoder(model.encoder)
    
    print("=> loading checkpoint '{}'".format(args.pretrained))
    checkpoint = torch.load(args.pretrained, map_location="cpu")
    
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
        if k.startswith('module'):
            state_dict[k[len("module."):]] = state_dict[k]
            del state_dict[k]
            k = k[len("module.")]
    
        if 'predictor' in k or 'projector' in k or 'target' in k \
            or k.startswith('encoder.fc') or k.startswith('cls') :
            # delete renamed or unused k
            del state_dict[k]
    
    msg = model.load_state_dict(state_dict) # , strict=False)
    print("=> loaded pre-trained model '{}'".format(args.pretrained))
    model.encoder.fc = nn.Identity()
    model = model.cuda()
    
    with torch.no_grad():
        feats, labels = get_features(test_loader, model, args)
        feats, labels = feats.cpu().numpy(), labels.cpu().numpy()
    info['feats'] = feats
    info['labels'] = labels
else:
    info = pk.load(open(feat_path, 'rb'))
    feats, labels = info['feats'], info['labels']

if args.mode == 'tsne':
    if 'tsne_feats' not in info:
        from sklearn.manifold import TSNE
        tsne_feats = TSNE(n_components=2, init='random').fit_transform(feats)
        info['tsne_feats'] = tsne_feats
    else:
        tsne_feats = info['tsne_feats']

    from matplotlib import pyplot as plt
    targets = np.unique(labels)
    for target in targets:
        ind = labels == target
        target_feats = tsne_feats[ind]
        plt.scatter(target_feats[:,0], target_feats[:,1], label=target, s=1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
elif args.mode == 'uniformity':
    feats = torch.tensor(feats)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    loss = uniform_loss(feats)
    print(f'Uniformity: {-loss}')

pk.dump(info, open(feat_path, 'wb'))
