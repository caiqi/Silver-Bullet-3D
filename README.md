# Introduction
This repository is the official implementation of *Silver-Bullet-3D* Solution for SAPIEN ManiSkill Challenge 2021

## Requirement:
* PyTorch 1.8.0+
* Python3.7
* CUDA 10.1+

Other requirements please refer to [environment.yml](xxxx)

## Clone the repository:
```
git clone https://github.com/JDAI-CV/CoTNet.git
```

## No Interaction Track
### Training
Training code is provided in No_Interaction/training folder. For example, to train MoveBucket model, using following script:
```
CONFIG_NAME=bucket/1225_bucket_ensemble_v1
SEED=1345
CUDA_VISIBLE_DEVICES=0 python -m tools.run_rl configs/${CONFIG_NAME}.py --gpu-ids=0 --seed ${SEED} --work-dir ${CONFIG_NAME}
```
For final submission, we ensemble multiple models with different network architecture and random seed. All configs are provided in [configs] folder.
## No Restriction Track
The training and evaluation code is the same. 
|Task | Code|
| :---: | :---: |
|OpenCabinetDoor | [user_solution_bucket.py]()|
|OpenCabinetDrawer | [user_solution_bucket.py]()|
|MoveBucket | [user_solution_bucket.py]()|
|PushChair | [user_solution_bucket.py]()|


# The evaluation code and checkpoint is []

## Training
# CUDA_VISIBLE_DEVICES=0 python -m tools.run_rl configs/dt/1225_drawer_ensemble_v1.py --gpu-ids=0 --seed 7777 --work-dir 0111_drawer_ensemble_v7
