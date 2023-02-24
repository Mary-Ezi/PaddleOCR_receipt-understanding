# 
import json
from os import listdir
from os.path import isfile, join

set_names = set()
def read_image_labels(json_path):
    
    with open(json_path, encoding='utf-8') as file:
        data = json.load(file)

    labels = []
    for line in data['valid_line']:
        category = line['category']
        group_id = line['group_id']
        set_names.add(category)
        for word in line['words']:
            label_info = {'transcription': word['text'], 'label': category, 'id': group_id, "linking": []}
            points = word['quad']
            label_info['points'] = [[points['x1'], points['y1']], [points['x2'], points['y1']], [points['x2'], points['y2']], [points['x1'], points['y2']]]
            labels.append(label_info)
    
    return labels

def write_meta_labels(json_pathes, output_file):

    with open(output_file, "w", encoding='utf-8') as fout:
        for json_path in json_pathes:
            labels = read_image_labels(json_path)

            json_file_name = json_path.split('/')[-1]
            file_name = json_file_name.split('.')[0] + '.png'
            fout.write(file_name + "\t" + json.dumps(labels, ensure_ascii=False) + "\n")


def prepare_data(stage='train'):

    image_json_dir = f"data/{stage}/json"

    json_pathes = [join(image_json_dir, f) for f in listdir(image_json_dir) if isfile(join(image_json_dir, f))]

    output_file = f"data/{stage}.txt"
    write_meta_labels(json_pathes, output_file)

def write_labels(dir='data'):
    with open(f"{dir}/labels.txt", "w") as fout:
        #fout.write(f"Ignore\n")
        for label in set_names:
            fout.write(f"{label}\n")
        #fout.write(f"Others")

for stage in ['train','dev', 'test']:
    prepare_data(stage)

write_labels()