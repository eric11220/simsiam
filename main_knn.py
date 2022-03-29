import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import simsiam.resnet
import simsiam.builder

from main_simsiam import test, get_features, arguments

parser = arguments()
parser.add_argument('--pretrained', default='', type=str,
                    help='path to simsiam pretrained checkpoint')
args = parser.parse_args()

mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

train_knn_dataset = datasets.CIFAR10(root='./data', train=True, transform=test_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=test_transform)

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

    if k.startswith('predictor') or k.startswith('encoder.fc'):
        # delete renamed or unused k
        del state_dict[k]

msg = model.load_state_dict(state_dict) # , strict=False)
print("=> loaded pre-trained model '{}'".format(args.pretrained))
model.encoder.fc = nn.Identity()
model = model.cuda()

test(train_knn_loader, test_loader, model, 0, args, knn_k=25)
