# Introduction
This repository is the official implementation of *Silver-Bullet-3D* Solution for SAPIEN ManiSkill Challenge 2021

## Requirement:
* PyTorch 1.8.0+
* Python3.7
* CUDA 10.1+

Other requirements please refer to [environment.yml](No_Interaction/evaluation/bucket_track1/environment.yml)

## Clone the repository:
```
git clone https://github.com/caiqi/Silver-Bullet-3D
```

## No Interaction Track

### Data preparation
* Download ManiSkill dataset from [here](https://github.com/haosulab/ManiSkill)
* (Optional) Compress the data with [compress_data.py](No_Interaction/training/tools/compress_data.py)

### Training
Training code is provided in [No_Interaction/training](No_Interaction/training) folder. For example, to train MoveBucket model, using following script:
```
CONFIG_NAME=bucket/1225_bucket_ensemble_v1
SEED=1345
CUDA_VISIBLE_DEVICES=0 python -m tools.run_rl configs/${CONFIG_NAME}.py --gpu-ids=0 --seed ${SEED} --work-dir ${CONFIG_NAME}
```
For final submission, we ensemble multiple models with different network architecture and random seed. All configs are provided in [configs](No_Interaction/training/configs) folder.

### Evaluation
Evaluation code and checkpoints are provided in [No_Interaction/evaluation](No_Interaction/evaluation). For example, to evaluate the pre-trained models on MoveBucket, use ManiSkill official evaluation code: 
```
PYTHONPATH=No_Interaction/evaluation/bucket_track1:$PYTHONPATH python evaluate_policy.py --env MoveBucket-v0 --level-range "0-300"
```

[evaluate_policy.py](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/tools/evaluate_policy.py) is from [ManiSkill](https://github.com/haosulab/ManiSkill) repo. Checkpoints can be downloaded from release page.
|       Task        |                                              Models                                              |
| :---------------: | :----------------------------------------------------------------------------------------------: |
|  OpenCabinetDoor  |  [Checkpoint](https://github.com/caiqi/Silver-Bullet-3D/releases/download/v1.0/door_track1.zip)  |
| OpenCabinetDrawer | [Checkpoint](https://github.com/caiqi/Silver-Bullet-3D/releases/download/v1.0/drawer_track1.zip) |
|    MoveBucket     | [Checkpoint](https://github.com/caiqi/Silver-Bullet-3D/releases/download/v1.0/bucket_track1.zip) |
|     PushChair     | [Checkpoint](https://github.com/caiqi/Silver-Bullet-3D/releases/download/v1.0/chair_track1.zip)  |

## No Restriction Track
The training and evaluation code is the same. 
|       Task        |                               Code                                |
| :---------------: | :---------------------------------------------------------------: |
|  OpenCabinetDoor  |   [user_solution_door.py](No_Restriction/user_solution_door.py)   |
| OpenCabinetDrawer | [user_solution_drawer.py](No_Restriction/user_solution_drawer.py) |
|    MoveBucket     | [user_solution_bucket.py](No_Restriction/user_solution_bucket.py) |
|     PushChair     |  [user_solution_chair.py](No_Restriction/user_solution_chair.py)  |
