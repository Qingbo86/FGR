### Environment Setup

```bash
conda create -n FGR python=3.10
# cuda and pytorch
conda install cuda-toolkit -c nvidia/label/cuda-11.8.0
# pytorch<=2.1 is required
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# dependencies
pip install -r requirements.txt
pip install opencv-python-headless ftfy regex
pip install mamba-ssm[causal-conv1d]
# mamba kernel
cd kernels/selective_scan && pip install . && cd ../.. # takes ~15 mins
# mmcv and mmagic (mmcv<=2.1.0 is required)
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
cd mmagic && pip install -r requirements.txt && pip install -e .
```


```

### Training

```bash
export CONFIG_PATH=configs/pixmamba/final.py
export NUM_GPUS=1
cd mmagic
bash tools/dist_train.sh $CONFIG_PATH $NUM_GPUS # ~5hrs training on single RTX 4090 GPU
```

### Testing

```bash
export CONFIG_PATH=configs/pixmamba/final.py
cd mmagic
python tools/test.py $CONFIG_PATH <pth_filepath> --input_dir <input_image_dir> --output_dir <output_image_dir>
```
