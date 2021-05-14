# Split the dataset of each component in shapenet into train, test and valid data
# Train: 70 %
# Test: 15 %
# Valid: 15 %

import json
import os

data_path = '/media/caig/FECA2C89CA2C406F/sketch3D/sketch_part_rec/datasets/'
shapenet_part_folder = '/media/caig/423ECD443ECD3229/dataset/partnet_sketch/03001627/'

shapenet_data_dict = {}
shapenet_data_dict['chair'] = {}
shapenet_data_dict['chair']['taxonomy_id'] = '03001627'

for component in os.listdir(shapenet_part_folder):
    shapenet_data_dict['chair'][component] = {}
    
    component_path = os.path.join(shapenet_part_folder, component)
    samples = os.listdir(component_path)
    sample_num = len(samples)
    
    train_num = int(sample_num*0.7)
    test_num = int(sample_num*0.15)
    valid_num = sample_num - train_num - test_num

    train_samples = samples[:train_num]
    test_samples = samples[train_num:train_num + test_num]
    valid_samples = samples[train_num + test_num:]

    shapenet_data_dict['chair'][component]['train'] = train_samples
    shapenet_data_dict['chair'][component]['test'] = test_samples
    shapenet_data_dict['chair'][component]['val'] = valid_samples
    
    print(component, len(samples))
    print(len(train_samples))
    print(len(test_samples))
    print(len(valid_samples))


# the json file where the output must be stored
shapenet_part_data = os.path.join(data_path, "ShapeNetPart.json")
out_file = open(shapenet_part_data, "w")  
json.dump(shapenet_data_dict, out_file, indent = 6)
out_file.close()