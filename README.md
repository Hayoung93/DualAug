## Repository of the paper: "On the Importance of Dual-Space Augmentation for Domain Generalized Object Detection"
- Note: Our codes are based on [DINO](https://github.com/IDEA-Research/DINO).

### Preparation
- Build environment using Dokcerfile:  
    `docker build - < Dockerfile -t dualaug:latest`
- Clone this repo:  
    `git clone git@github.com:Hayoung93/DualAug.git`
- Install DINO:  
    ```
    cd /workspace/DualAug/models/dino/ops
    bash make.sh
    ```

### Inference on sample image
- Download model weight from the [release](https://0.0.0.0) page.
```
python visualize.py --input_image {INPUT_IMAGE} --model_weight checkpoint-inference.pth
```