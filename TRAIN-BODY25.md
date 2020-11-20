# How to Train BODY25

## Requirements
* Put coco dataset on `<project>/coco`

## Commands
* `python train1.py --checkpoint-path mobilenet_weights/mobilenet_sgd_68.848.pth.tar --from-mobilenet --num-workers 6 --batch-size 80 --batches-per-iter 1`
* `python train1.py --checkpoint-path body25_checkpoints/checkpoint_iter_420000.pth --weights-only --num-workers 6 --batch-size 80 --batches-per-iter 1`
* `python train1.py --checkpoint-path body25_checkpoints/checkpoint_iter_280000.pth --weights-only --num-refinement-stages 3 --num-workers 6 --batch-size 80 --batches-per-iter 1`

## GCP Notes
* `gcloud compute instances create "pytorch-gpu-instance" --zone="asia-northeast3-c" --image-family="pytorch-latest-gpu" --image-project=deeplearning-platform-release --maintenance-policy=TERMINATE --accelerator="type=nvidia-tesla-t4,count=1" --metadata="install-nvidia-driver=True" --boot-disk-type pd-ssd --boot-disk-size 100GB --custom-cpu 8 --custom-memory 32`
* `unzip -q filename.zip`