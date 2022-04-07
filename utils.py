import copy
import numpy as np

from matplotlib import pyplot as plt


# Continual Learning
def create_cl_datasets(dataset, num_task=5, class_groups=None):
    datasets = []

    targets = np.array(dataset.targets)
    if class_groups is None:
        classes = np.unique(targets)
        class_groups = np.split(classes, num_task)

    class_indices = []
    for group in class_groups:
        indices = np.sum(np.stack([targets == cls for cls in group]), axis=0)
        indices = indices.astype(np.bool)
        class_indices.append(indices)
        data = dataset.data[indices]
        labels = targets[indices]

        new_dset = copy.deepcopy(dataset)
        new_dset.data = data; new_dset.targets = labels
        new_dset.classes = [dataset.classes[cls] for cls in group]
        new_dset.class_to_idx = {_class: cls for _class, cls in zip(new_dset.classes, group)}

        datasets.append(new_dset)

    return datasets, class_groups, np.array(class_indices)


def load_from_state_dict(model, ckpt_path):
    print("=> loading checkpoint '{}'".format(ckpt_path))
    checkpoint = torch.load(ckpt_path, map_location="cpu")

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
    print("=> loaded pre-trained model '{}'".format(ckpt_path))


# Plotting
def plot_images(x1, x2, n=10):
    x1, x2 = x1[:n], x2[:n]

    x1 = x1.detach().permute(0, 2, 3, 1).cpu().numpy()
    x2 = x2.detach().permute(0, 2, 3, 1).cpu().numpy()

    x1 = (x1 * 255).astype(np.uint8)
    x2 = (x2 * 255).astype(np.uint8)

    fig, ax = plt.subplots(2, 10)
    for i, img in enumerate(x1):
        ax[0, i].imshow(img)

    for i, img in enumerate(x2):
        ax[1, i].imshow(img)

    fig.savefig('test.png')

