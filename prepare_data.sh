cd coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q train2017.zip
unzip -q val2017.zip
unzip -q annotations_trainval2017.zip
cd ..
cd data
wget https://github.com/RedPillLab/lightweight-human-pose-estimation.pytorch/releases/download/v0.1/prepared_train_annotation_1.pkl
wget https://github.com/RedPillLab/lightweight-human-pose-estimation.pytorch/releases/download/v0.1/val_subset_1.json
cd ..
pip install opencv-python-headless
pip install pycocotools
