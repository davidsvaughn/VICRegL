virtualenv -p python3.10 venv
source venv/bin/activate

python -m pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 torchaudio==0.11.0+cu115 -f https://download.pytorch.org/whl/cu115/torch_stable.html

pip install timm

-------------

python -m torch.distributed.launch --nproc_per_node=8 main_vicregl.py 
--fp16 
--exp-dir /media/david/Elements/imagenet/VICRegL/exp 
--arch resnet50 
--epochs 100 
--batch-size 512 
--optimizer lars 
--base-lr 0.3 
--weight-decay 1e-06 
--size-crops 224 
--num-crops 2 
--min_scale_crops 0.08 
--max_scale_crops 1.0 
--alpha 0.75


----

torchrun --standalone --nnodes=1 --nproc_per_node=1 main_vicregl.py --fp16 --exp-dir /media/david/Elements/imagenet/VICRegL/exp --arch resnet50 --epochs 10 --batch-size 32 --optimizer lars --base-lr 0.3 --weight-decay 1e-06 --size-crops 224 --num-crops 2 --min_scale_crops 0.08 --max_scale_crops 1.0 --alpha 0.75 --dump cfg_resnet.json

torchrun --standalone --nnodes=1 --nproc_per_node=1 main_vicregl.py --fp16 --exp-dir /media/david/Elements/imagenet/VICRegL/exp --arch convnext_small --epochs 10 --batch-size 384 --optimizer adamw --base-lr 0.00075 --alpha 0.75 --dump cfg_convnext.json