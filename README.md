<p align="center">

  <h1 align="center">DynaPix SLAM: A Pixel-Based Dynamic Visual SLAM Approach</h1>
  <div align="center">
    <img src="arch.png" alt="Logo" width="100%">
  </div>
</p>
Visual SLAM methods often encounter challenges in dynamic scenes that severely affect their core modules. To avoid that, Dynamic V-SLAM methods often apply semantic cues, geometric constraints, or optical flows to exclude dynamic elements. However, heavy reliance on precise segmentation and the a-priori inclusion of selected classes, along with the inability to recognize unknown moving objects, degrade their performance.
<br>
To address this, we introduce DynaPix, a novel visual SLAM system based on per-pixel motion probability estimation. Our method includes the semantic-free estimation module and an improved pose optimization process. The motion probability is calculated through static background differencing on both images and optical flows from splatted frames. DynaPix fully integrates those motion probabilities into the tracking and optimization modules based on ORB-SLAM2 framework.

## Test Dataset & Results
All reported experiments are detailed [here](https://docs.google.com/spreadsheets/d/1U17h2E4B3n4F_5GlqwzeLfyTZJXzkuGsULJei4SDDGU/edit?usp=sharing). (with additional results from other methods for reference)

We provide two sequences [here](https://drive.google.com/drive/folders/1P0XqJlqzV9Td4lYP0-Q_BVkSz7u8TzQR?usp=drive_link) for direct testing, involving the `FH` sequence from GRADE dataset and `halfsphere` sequence from TUM RGB-D dataset, which have following structure:
```
Sequence
    ├── rgb/     # images in dynamic scenes
    ├── background/     # images with (estimated) static background
    ├── prob/     # estimated motion probability for SLAM
    ├── depth/     # depth map
    ├── (depth_bg)/     # depth map for static background
    ├── groundtruth.txt     # camera pose groundtruth
    └── association.txt     # correspondence between depth and rgb images
```
- Download complete datasets : <br>
GRADE [[Sequence](https://grade.is.tue.mpg.de/)] [[Motion Prob](https://drive.google.com/file/d/1BZwoIId54i4JCz4dIwhfv0x_4kdQwgtN/view)] / TUM RGB-D [[Sequence](https://vision.in.tum.de/data/datasets/rgbd-dataset/download)] [[Inpainted Background/Motion Prob](https://drive.google.com/file/d/1PNl9_rjEcXH_xOaLl8kpLgmCKMxZZNsV/view?usp=drive_link)]
- Download complete experiment results (estimated trajectory files) [here](https://drive.google.com/file/d/1iAxFlMdzA68gBb3Tz2YFyNb1gCfZ_K9g/view?usp=drive_link)

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
      copyright = {arXiv.org perpetual,non-exclusive license}
}
```