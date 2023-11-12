# Motion Probability Estimation

This section focuses on **estimating motion probability** based on a remarkble flow estimation approach, [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official). We make use of a two-staged approach by blending **movable** and **moving** estimation to obtain a reliable motion probability distribution.

## Prerequisties
- Underlying FlowFormer Installation:
    ```
    conda create --name dynapix python=3.6.13
    conda activate flowformer
    pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu102
    # test with CUDA 10.2
    conda install cudatoolkit=10.2 matplotlib scipy
    pip install yacs loguru einops timm==0.4.12 imageio
    ```
- Motion Estimation Module:
    ```
    pip3 install pycocotools heapq pyquaternion confuse opencv-python==4.6.0.66
    ```

## Run
- Generate motion probability on GRADE dataset
    ```
    python3 motion_grade.py --debug --output
    ```

- Generate motion probability on TUM-RGBD dataset
    ```
    python3 motion_tum.py --debug --output
    ```

