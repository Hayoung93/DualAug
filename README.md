## Repository of the paper: "On the Importance of Dual-Space Augmentation for Domain Generalized Object Detection"
- Note: Our codes are based on [DINO](https://github.com/IDEA-Research/DINO).

### Preparation
- Build environment using Dokcerfile:  
    `docker build - < Dockerfile -t dualaug:latest`
- Create docker container  
    `docker run --name dualaug --gpus all -it --shm-size 64G -v /home/work/Desktop/:/data dualaug:latest`  
    You can add additional options
- Clone this repo:  
    `git clone https://github.com/Hayoung93/DualAug.git`
- Install packages:  
    ```
    pip install -r requirements.txt
    pip install git+https://github.com/openai/CLIP.git
    ```
- Install DINO:  
    ```
    cd /workspace/DualAug/models/dino/ops
    bash make.sh
    apt install libgl1-mesa-glx libglib2.0-0 -y
    ```

### Inference on sample image
- Download model weight from the [release](https://github.com/Hayoung93/DualAug/releases) page.
```
python visualize.py --input_image {INPUT_IMAGE} --model_weight checkpoint-inference.pth
```
- The result will be saved as `{INPUT_IMAGE_NAME}_predBox.png`
- Sample results
<p align="center">
  <img src="samples/0b95721d-fb1789c4.jpg" width="200" />
  <img src="samples/0c7c9049-7e4e5ed5.jpg" width="200" />
  <img src="samples/3b2cc921-dd124456.jpg" width="200" />
  <img src="samples/foggy-036.jpg" width="200" /> 
</p>
<p align="center">
  <img src="samples/0b95721d-fb1789c4_predBox.png" width="200" />
  <img src="samples/0c7c9049-7e4e5ed5_predBox.png" width="200" />
  <img src="samples/3b2cc921-dd124456_predBox.png" width="200" />
  <img src="samples/foggy-036_predBox.png" width="200" /> 
</p>
