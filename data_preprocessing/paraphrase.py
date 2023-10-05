import gpt_wrapper
from gpt_wrapper.chat import Chat
import json
import argparse
from convert_format import *

parser = argparse.ArgumentParser(description='Augment by paraphrasing')
parser.add_argument('-path', '-p', type=str, help='path to a file with interactions to augment')
parser.add_argument('-save_filename', '-f', type=str, help='filename to save the augmented interactions')
parser.add_argument('-api_key', type=str, help='gpt wrapper api key')
parser.add_argument('--split', action='store_true', help='set if data split required')
parser.add_argument('--add_to_file', '--a', type=str, help='path to an existing file to add more paraphrased interactions')
parser.add_argument('--start_stop', type=int, nargs='+', help='define the first and the last interaction to paraphrase')
args = parser.parse_args()

def paraphrase(num, content):
    chat = Chat.create(f'Q{num+1}')
    q = 'Please, paraphrase this text: ' + content
    instruction = 'Output no more than two sentences.'
    message = chat.ask(q, instruction=instruction)
    message = message.to_dict()
    return message

def make_paraphrased_ineraction(interaction):
    new_interaction = []
    for j in interaction:
        if j['role'] != 'assistant':
            new_interaction.append(j)
        elif j['role'] == 'assistant':
            paraphrased = paraphrase(0, j['content'])
            new_interaction.append({'role': 'assistant', 'content': paraphrased['content']})
    return new_interaction

def take_low_scores(data_clean):
    data_low = []
    for i in data_clean:
        if i['confidence'] <= 3:
            data_low.append(i)
    return data_low

if __name__ == '__main__':
    gpt_wrapper.api_key = args.api_key
    path = args.path
    save_name = args.save_filename
    split = args.split
    add_to_file = args.add_to_file
    start, stop = args.start_stop[0], args.start_stop[1]

    data_clean = clean_data(path)
    data_low = take_low_scores(data_clean)
    
    new_data = []

    for i in data_low[start:stop+1]: # don't paraphrase everything at once
        interaction = i['interaction']
        new_interaction = make_paraphrased_ineraction(interaction)
        new_data.append({'confidence': i['confidence'], 'interaction': new_interaction, "sol_id": i['sol_id'],
            "interaction_id": i['interaction_id']})
        
    data_to_write = convert(new_data, split=split)

    if add_to_file is not None:
        with open (add_to_file, 'r', encoding="cp437", errors='ignore') as file:
            new_data = json.load(file)
        data_to_write = new_data + data_to_write
    
    
    with open(save_name, "w") as file:
        json.dump(data_to_write, file, indent=4)

