import glob
import numpy as np
import os
from main_simsiam import arguments
from main_knn import test_model

nclass = 10

if __name__ == '__main__':
    parser = arguments()
    parser.add_argument('--pretrained-dir', default='', type=str,
                        help='path to simsiam pretrained checkpoint directory')
    parser.add_argument('--epoch-per-task', default=100, type=int,
                        help='how many epochs were each task trained for')
    args = parser.parse_args()
    
    ckpt_paths = [
        glob.glob('{}/checkpoint_{:04d}_*.pth.tar'.format(args.pretrained_dir, epoch * args.epoch_per_task-1))[0] \
        for epoch in range(1, args.ntask+1)
    ]
    for path in ckpt_paths:
        if not os.path.exists(path):
            raise Exception(f'{path} does not exist')

    accu_mat = np.zeros((args.ntask, nclass))
    for task, path in enumerate(ckpt_paths):
        _, class_correct, class_total = test_model(path, args)
        for cls in range(nclass):
            accu_mat[task][cls] = class_correct[cls] / class_total[cls]

    task_mat = np.zeros((args.ntask, args.ntask))
    class_groups =  np.split(np.arange(nclass), args.ntask)
    for task, grp in enumerate(class_groups):
        task_mat[:, task] = np.mean(accu_mat[:, grp], axis=-1)

    forgetting = task_mat[np.arange(args.ntask), np.arange(args.ntask)] - task_mat[-1]
    print(task_mat)
    print(forgetting)
    print(np.mean(forgetting))
