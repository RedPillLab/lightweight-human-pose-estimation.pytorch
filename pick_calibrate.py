import glob, random, os
from shutil import copyfile
from PIL import Image

image_files = []
for file in glob.glob("coco/val2017/*.jpg"):
    im = Image.open(file)
    if im.width / im.height > 1.3:
        image_files.append(file)
        if len(image_files) > 0 and len(image_files) % 100 == 0:
            print(len(image_files))

sample_files = random.sample(image_files, 1000)
for file in sample_files:
    target_file = "coco/calibrate/" + os.path.basename(file)
    copyfile(file, target_file)