import json

def convert(coco_annotations_file, whole_annotations_file, out_file):
    with open(coco_annotations_file, 'r') as f:
        data = json.load(f)

    with open(whole_annotations_file, 'r') as f:
        data_whole = json.load(f)

    for i in range(len(data['annotations'])):
        foot_kpts = data_whole['annotations'][i]['foot_kpts']
        foot_kpts = [ int(x) for x in foot_kpts ]
        num_foot_kpts = sum(v > 0 for v in foot_kpts[2::3])
        if len(data['annotations'][i]['keypoints']) == 51:
            data['annotations'][i]['keypoints'].extend(foot_kpts)
            data['annotations'][i]['num_keypoints'] += num_foot_kpts
        else:
            print(i, 'keypoints length is not 51 !?')

    with open(out_file, 'w') as f:
        json.dump(data, f)

def test(annotations_file):
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    for annotation in data['annotations']:
        print(annotation['num_keypoints'], annotation['keypoints'])

if __name__ == '__main__':
    #convert('coco/annotations/person_keypoints_train2017.json',
    #        'coco/annotations/coco_wholebody_train_v1.0.json',
    #        'coco/annotations/body25_train2017.json')
    # convert('coco/annotations/person_keypoints_val2017.json',
    #         'coco/annotations/coco_wholebody_val_v1.0.json',
    #         'coco/annotations/body25_val2017.json')
    test('coco/annotations/body25_val2017.json')
