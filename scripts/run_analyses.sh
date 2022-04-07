#!/bin/bash

# targets="simsiam_cifarresnet18_ratio0.2 sup_noncont_cifarresnet18 sup_noncont_cifarresnet18_strongaug lr0.06"
targets="ce_integration"

cd ..
declare -A knn_accu
declare -A linear_accu
for target in $targets; do
    last_ckpt=`ls checkpoints/$target | grep 0799`
    set -x; msg=`python main_knn.py --pretrained checkpoints/$target/$last_ckpt --gpu 0`; set +x
    accu=${msg##*$'\n'}
    knn_accu[$target]=$accu

    set -x; msg=`python main_lincls.py -a resnet18 --pretrained checkpoints/$target/$last_ckpt --lars --gpu 0 --print-freq 100`; set +x
    accu=${msg##*$'\n'}
    linear_accu[$target]=$accu
done

cd analysis
declare -A uniformity
for target in $targets; do
    last_ckpt=`ls ../checkpoints/$target | grep 0799`
    python analysis.py --pretrained ../checkpoints/$target/$last_ckpt --gpu 0 --mode tsne
    set -x; msg=`python analysis.py --pretrained ../checkpoints/$target/$last_ckpt --gpu 0 --mode uniformity`; set +x
    uni=${msg##*$'\n'}
    uniformity[$target]=$uni
done

for x in "${!knn_accu[@]}"; do printf "[%s]=%s\n" "$x" "${knn_accu[$x]}" ; done
for x in "${!linear_accu[@]}"; do printf "[%s]=%s\n" "$x" "${linear_accu[$x]}" ; done
for x in "${!uniformity[@]}"; do printf "[%s]=%s\n" "$x" "${uniformity[$x]}" ; done
