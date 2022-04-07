import torch
import torch.nn as nn
from torchvision import datasets, transforms

import simsiam.builder
import simsiam.resnet
from main_simsiam import test, arguments
from utils import load_from_state_dict


def get_loaders(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], batch_size=512):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=test_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return train_loader, test_loader


def get_model(args):
    model = simsiam.builder.SimSiam(
        simsiam.resnet.ResNet18,
        args.dim, args.pred_dim, predictor_reg=args.predictor_reg)
    model = simsiam.builder.SimSiamEncoder(model.encoder)

    load_from_state_dict(model, args.pretrained)
    model.encoder.fc = nn.Identity()
    model = model.cuda()
    return model


def test_model(ckpt_path, args):
    train_knn_loader, test_loader = get_loaders(batch_size=args.batch_size)
    model = get_model(args)
    return test(train_knn_loader, test_loader, model, 0, 0, args, knn_k=25)

if __name__ == '__main__':
    parser = arguments()
    parser.add_argument('--pretrained', default='', type=str,
                        help='path to simsiam pretrained checkpoint')
    args = parser.parse_args()
    test_model(args.pretrained, args)
