# reimplemention-of-Dual-PRNet

This is a Tensorflow reimplemention of [Dual-Stream Pyramid Registration Network](https://arxiv.org/abs/1909.11966)

## Install
The package and their corresponding version we used in this repository are listed in below.
Tensorflow==1.15.4
Keras==2.3.1
tflearn==0.5.0

## Training
After configuring the environment, please use this command to train the model.

```sh
python train.py -g 0 --batch 1 -d datasets/brain.json -b DUAL -n 1 --round 10000 --epoch 10
```

## Testing
Use this command to obtain the testing results.
```sh
python predict.py -g 0 --batch 1 -d datasets/brain.json -c weights/Dec09-1849
```

## LPBA dataset
We use the same training and testing data as [RCN](https://github.com/microsoft/Recursive-Cascaded-Networks),please refer to their repository to download the pre-processed data.

## Acknowledgment

Some codes are modified from [RCN](https://github.com/microsoft/Recursive-Cascaded-Networks) and [VoxelMorph](https://github.com/voxelmorph/voxelmorph).
Thanks a lot for their great contribution.
