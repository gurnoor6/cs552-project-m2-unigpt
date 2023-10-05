import json
import argparse

parser = argparse.ArgumentParser(description='Convert data format')
parser.add_argument('-path', '-p', type=str, help='path to a file with the data')
parser.add_argument('-save_filename', '-f', type=str, help='filename to save the converted data')
parser.add_argument('--split', action='store_true', help='set if data split required')
args = parser.parse_args()

def clean_data(path):
    with open (path, 'r', encoding="cp437", errors='ignore') as file:
        data = json.load(file)
    valid_scores = [x for x in range(1, 6)] # or in range(6) if you want ot include score 0
    data_clean =  filter(lambda x: x['confidence'] in valid_scores, data)
    return data_clean

def make_entry(sample, entry_id):
    if sample['confidence'] > 3:
        label = 1
    else:
        label = 0
    all_entries = []
    for interaction in sample['interaction']:
        if interaction['role'] == 'system':
            if len(interaction['content']) != 0:
                entry = ''.join('System: ' + interaction['content'] + '\n\n')
            else:
                entry = ''
        elif interaction['role'] == 'user':
            entry = ''.join('Human: ' + interaction['content'] + '\n\n')
        elif interaction['role'] == 'assistant':
            entry = ''.join('Assistant: ' + interaction['content'] + '\n\n')
        
        
        all_entries.append(entry)
    chat = ''.join(all_entries)
    dict_entry = {'entry_id': entry_id, 'label': label, 'chat': chat}
    return dict_entry

def convert(clean_data, split=False):
    data_to_write = []
    for num, sample in enumerate(clean_data):
        dict_entry = make_entry(sample, num)
        data_to_write.append(dict_entry)

    if split:
        data_to_write = split_interactions(data_to_write)
        k=0
        for entry in data_to_write:
            entry['entry_id'] = k 
            k+=1
    return data_to_write

def write_json(filename, clean_data, split=False):
    data_to_write = convert(clean_data, split=split)

    with open(filename, "w") as file:
        json.dump(data_to_write, file, indent=4)

def split_interactions(data):
    new_data = []
    for interaction in data:
        messages = interaction['chat'].split("\n\nSystem: ")
        new_chat = messages[0]
        dict_entry = {'entry_id': 0, 'label': interaction['label'], 'chat': new_chat}
        new_data.append(dict_entry)
        for message in messages[1:]:
            new_chat = "System: " + message
            dict_entry = {'entry_id': 0, 'label': interaction['label'], 'chat': new_chat}
            new_data.append(dict_entry)
    return new_data


if __name__ == '__main__':
    path = args.path
    save_name = args.save_filename
    split = args.split

    data_clean = clean_data(path)
    write_json(save_name, data_clean, split=split)
