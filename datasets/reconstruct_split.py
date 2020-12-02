import json

json_path = '/media/itri/Files_2TB/chaoyu/pointcloud3d/Pix2Vox/datasets/ShapeNet.json'

# Load all taxonomies of the dataset
with open(json_path, encoding='utf-8') as file:
    dataset_taxonomy = json.loads(file.read())

reconstruct_data = {}
for taxonomy in dataset_taxonomy:
    class_data = {}
    class_data['taxonomy_id'] = taxonomy['taxonomy_id']
    class_data['test'] = taxonomy['test']
    class_data['train'] = taxonomy['train']
    class_data['val'] = taxonomy['val']    
    
    reconstruct_data[taxonomy['taxonomy_name']] = class_data

with open('rec.json', 'w') as fp:
    json_dumps_str = json.dumps(reconstruct_data, indent=4)
    print(json_dumps_str, file=fp)

# for key, value in reconstruct_data.items():
#     print(key)
#     print(value['taxonomy_id'])
#     print(len(value['test']))
#     print(len(value['train']))
#     print(len(value['val']))
#     print(" ")


    