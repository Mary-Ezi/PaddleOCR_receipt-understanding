# 
from collections import defaultdict
import json
from os import listdir
from os.path import isfile, join


def read_image_labels(json_path):
    global labels, label_cnt
    
    with open(json_path, encoding='utf-8') as file:
        data = json.load(file)

    metadata_labels = []
    for line in data['valid_line']:
        category = line['category']
        group_id = line['group_id']
        if category not in labels:
            labels[category] = label_cnt
            label_cnt += 1

        for word in line['words']:
            label_info = {'transcription': word['text'], 'label': labels[category], 'id': group_id}
            points = word['quad']
            label_info['points'] = [[points['x1'], points['y1']], [points['x2'], points['y1']], [points['x2'], points['y2']], [points['x1'], points['y2']]]
            metadata_labels.append(label_info)
    
    return metadata_labels

def write_metadata(json_pathes, output_file):

    with open(output_file, "w", encoding='utf-8') as fout:
        for json_path in json_pathes:
            metadata_labels = read_image_labels(json_path)

            json_file_name = json_path.split('/')[-1]
            file_name = json_file_name.split('.')[0] + '.png'
            fout.write(file_name + "\t" + json.dumps(metadata_labels, ensure_ascii=False) + "\n")


def prepare_data(dir, stage='train'):

    image_json_dir = f"{dir}/{stage}/json"

    json_pathes = [join(image_json_dir, f) for f in listdir(image_json_dir) if isfile(join(image_json_dir, f))]

    output_file = f"data/{stage}.txt"
    write_metadata(json_pathes, output_file)

def write_labels(dir):
    with open(f"{dir}/labels.txt", "w") as fout:
        for label, index in labels.items():
            fout.write(f"{index} {label}\n")

dir = 'data'
labels = {}
label_cnt = 0

for stage in ['train','dev', 'test']:
    prepare_data(dir, stage)

write_labels(dir)

