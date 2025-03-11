conda create -n GenSeg python==3.9
conda activate GenSeg
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install imgaug
pip install numpy==1.23
pip install pyyaml
pip install yacs
pip install einops
pip install timm
pip install wandb
pip install pip==23.3
pip install betty-ml==0.2.0
