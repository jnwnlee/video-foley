from glob import glob
import os
import librosa
import argparse
from tqdm import tqdm

def gen_list(data_path, output_dir, video_segment_len=10):
    # get file ids from unique YYYY-MM-DD-HH-MM-SS_* files
    file_ids = list(set([os.path.basename(file).split('_')[0] for file in glob(os.path.join(data_path, '*-*-*-*-*-*_*.*'))]))
    print('Found {} unique file ids'.format(len(file_ids)))

    # make train/test filelist for each material
    train_list_path = os.path.join(data_path, 'train.txt')
    test_list_path = os.path.join(data_path, 'test.txt')

    train_list = []
    test_list = []

    with open(train_list_path, 'r') as f:
        for line in f:
            train_list.append(line.strip())
    with open(test_list_path, 'r') as f:
        for line in f:
            test_list.append(line.strip())

    # make train/test filelist for each material
    train_dict = {} # key: material, value: list of file ids
    test_dict = {} # key: material, value: list of file ids

    for file_id in tqdm(file_ids):
        target_dict = None
        if file_id in train_list:
            target_dict = train_dict
        elif file_id in test_list:
            target_dict = test_dict
        else:
            raise ValueError('file_id {} not found in train/test list'.format(file_id))
        
        # get materials from annotation file
        annotation_path = os.path.join(data_path, file_id + '_times.txt')
        materials_dict = {}
        with open(annotation_path, 'r') as f:
            for line in f:
                onset, material, _, _ = line.split()
                onset_id = str(int(float(onset)//video_segment_len))
                materials_dict[onset_id] = materials_dict.get(onset_id, set()) 
                materials_dict[onset_id].add(material)
        # remove the last segment as it's shorter than video_segment_len
        audio, sr = librosa.load(os.path.join(data_path, file_id + '_denoised.wav'))
        audio_len = len(audio)/sr
        for onset_id, materials in materials_dict.copy().items():
            if int(onset_id) >= int(audio_len//video_segment_len):
                del materials_dict[onset_id]
        
        # make train/test filelist for each material
        for onset_id, materials in materials_dict.items():
            if len(materials) == 1 and 'None' in materials:
                target_dict['None'] = target_dict.get('None', []) + [file_id+'_'+onset_id]
            elif len(materials) == 1 or (len(materials) == 2 and 'None' in materials):
                if 'None' in materials:
                    materials.remove('None')
                material = materials.pop()
                target_dict[material] = target_dict.get(material, []) + [file_id+'_'+onset_id]
            else:
                target_dict['multiple'] = target_dict.get('multiple', []) + [file_id+'_'+onset_id]

    # save train/test filelist for each material
    os.makedirs(output_dir, exist_ok=True)
    for material, id_list in train_dict.items():
        with open(os.path.join(output_dir, '{}_train.txt'.format(material)), 'w') as f:
            for file_id in id_list:
                f.write(file_id + '\n')
                
    for material, id_list in test_dict.items():
        with open(os.path.join(output_dir, '{}_test.txt'.format(material)), 'w') as f:
            for file_id in id_list:
                f.write(file_id + '\n')
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", default="/media/daftpunk4/dataset/GreatestHits/vis-data")
    parser.add_argument("-o", "--output_dir", default="/media/daftpunk4/dataset/GreatestHits/features/filelists")
    parser.add_argument("-d", "--duration", type=int, default=10) # seconds
    args = parser.parse_args()
    
    gen_list(data_path=args.input_dir, output_dir=args.output_dir, video_segment_len=args.duration)