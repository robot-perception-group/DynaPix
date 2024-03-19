<p align="center">

  <h1 align="center">DynaPix SLAM: A Pixel-Based Dynamic Visual SLAM Approach</h1>
   <h3 align="center">
    <a href="https://kyle-xu001.github.io/"><strong>Chenghao Xu*</strong></a>
    ·
    <a href="https://eliabntt.github.io/"><strong>Elia Bonetto*</strong></a>
    ·
    <a href="https://www.aamirahmad.de/"><strong>Aamir Ahmad</strong></a>
  </h3>
  <p align="center"> (* Equal Contribution) </p>
  <div align="center">
    <img src="arch.png" alt="Logo" width="100%">
  </div>
</p>
DynaPix SLAM is the approach towards dynamic environments based on ORB-SLAM2 framework. This repository main includes the code for DynaPix SLAM, DynaPix-D SLAM, and corresponding motion estimation module, where the specific instructions are inside each folder.

## Motion Probability Estimation
- Generate motion probability on GRADE dataset
  ```bash
  cd motion_estimation
  python3 motion_grade.py --output
  ```
- Generate motion probability on TUM-RGBD dataset
  ```bash
  cd motion_estimation
  python3 motion_tum.py --output
  ```
> Note: Please specify the input directory and output directory in `configs/GRADE.yaml` and `configs/TUM.yaml` before generation.

## DynaPix / DynaPix+
DynaPix is derived from ORB-SLAM2, while DynaPix-D is derived from DynaSLAM. In this version, we only implement the RGB-D tracking for testing. Plase follow the instructions in each folder for more details.

## Test Dataset & Results
The details of all reported experiments are available [here](https://docs.google.com/spreadsheets/d/1U17h2E4B3n4F_5GlqwzeLfyTZJXzkuGsULJei4SDDGU/edit?usp=sharing).

We provide two sequences [here](https://drive.google.com/drive/folders/1P0XqJlqzV9Td4lYP0-Q_BVkSz7u8TzQR?usp=drive_link) for testing, involving the `FH` sequence from GRADE dataset and `halfsphere` sequence from TUM RGBD dataset, which have following structure:
```
Sequence
    ├── rgb/     # images in dynamic scenes
    ├── background/     # images with static background
    ├── prob/     # estimated motion probability for SLAM
    ├── depth/     # depth map
    ├── (depth_bg)/     # depth map for static background
    ├── groundtruth.txt     # camera pose groundtruth
    └── association.txt     # correspondence between depth and rgb images
```

## Evaluation
- Evaluate absoluate trajectory error (ATE) and tracking rate (TR):
  ```bash
  mkdir RESULT && cd RESULT/
  mkdir DynaPix
  cp ${ESTIMATED_FILE} RESULT/DynaPix/ 
  python3 eval/eval.py ${SEQ}/groundtruth.txt ${SEQ}/association.txt RESULT/
  ```
> Note: The evaluation scripts are derived from TUM SLAM evaluation tool for multiple trajectories testing and evaluation.

## Citation

```bibtex
@misc{xu2023dynapix,
      title={DynaPix SLAM: A Pixel-Based Dynamic SLAM Approach}, 
      author={Chenghao Xu and Elia Bonetto and Aamir Ahmad},
      year={2023},
      eprint={2309.09879},
      url={https://arxiv.org/abs/2309.09879},
      copyright = {arXiv.org perpetual, non-exclusive license}
}
```