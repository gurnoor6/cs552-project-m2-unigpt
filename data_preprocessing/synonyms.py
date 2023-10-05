import numpy as np
import nltk
from nltk.corpus import wordnet
from langdetect import detect
import random
from collections import Counter
import argparse
import json
from convert_format import *

nltk.download('wordnet')
nltk.download('omw-1.4')

parser = argparse.ArgumentParser(description='Augment by synonyms replacement')
parser.add_argument('-path', '-p', type=str, help='path to a file with interactions to augment')
parser.add_argument('-save_filename', '-f', type=str, help='filename to save the augmented interactions')
parser.add_argument('-num_words', '-n', type=int, help='number of words per interaction to replace with synonyms')
parser.add_argument('--split', action='store_true', help='set if data split required')
args = parser.parse_args()

def get_synonyms(word, lang):
    synonyms = set()
    for syn in wordnet.synsets(word, lang=lang):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    if word in synonyms:
        synonyms.remove(word)
    
    return list(synonyms)

def synonym_replacement(sentence, n):
    language = detect(sentence)
    if language == 'en':
      lang = 'eng'
    elif language == 'fr':
      lang = 'fra'
    else:
      lang = None
    
    if lang != None:
      words = sentence.split()
      
      words_wo_syn = ['System', 'Human', 'Assistant']
      replaced_ids = []
      
      for i in words:
        if len(get_synonyms(i, lang)) == 0:
          words_wo_syn.append(i)
      random_ids = []
      words_w_syn = len(words) - len(words_wo_syn)
      if words_w_syn < n:
        n = words_w_syn
      while n > 0:
        random_id = random.randint(0, (len(words)-1))
        if words[random_id] not in words_wo_syn and random_id not in replaced_ids:
          random_ids.append(random_id)
          n -= 1
          replaced_ids.append(random_id)

      def get_random_syn(word, lang):
        synonyms = get_synonyms(word, lang)
        if len(synonyms) == 1:
          return synonyms[0]
        else:
          rand_syn = random.randint(0, (len(synonyms)-1))
          return synonyms[rand_syn]

      random_word_list = [get_random_syn(word, lang) if words.index(word) in random_ids  else word for word in words]
      new_sentence = " ".join(random_word_list)
    else:
      new_sentence = ''
    
    return new_sentence


def upsample_with_synonyms(data, num_words_to_replace):
    chats = []
    labels = []
    for i in data:
        chats.append(i['chat'])
        labels.append(i['label'])
    data = {'chat': chats, 'label': labels}
  
    final_data = []
    new_data = []
    scores = [sample for sample in data['label']]
    scores = Counter(scores)
    # dif = scores['positive'] - scores['negative']
    dif = scores[1] - scores[0]
    inds = [i for i, x in enumerate(data['label']) if x == 0]
    # negative_samples  = [data['chat'] for sample in data['label'] if sample == 'negative']
    inds = np.random.permutation(len(inds))
    inds_to_replace = inds[0:dif]
    for ind in inds_to_replace:
        replaced = synonym_replacement(data['chat'][ind], num_words_to_replace)
        if len(replaced) != 0:
            new_data.append(replaced)
    data['chat'] = data['chat'] + new_data
    print(len(data['chat']))
    data['label'] = data['label'] + [0 for i in range(len(new_data))]
    print(len(data['label']))
    mix = np.random.permutation(len(data['label']))
    print(len(mix))
    new_data = {'chat': [data['chat'][i] for i in mix], 'label': [data['label'][j] for j in mix]}

    for num, (i, j) in enumerate(zip(new_data['label'], new_data['chat'])):
        final_data.append({'entry_id': num, 'label': i, 'chat' : j})
    return final_data

if __name__ == '__main__':
    path = args.path
    filename = args.save_filename
    num_words = args.num_words
    split = args.split

    data_clean = clean_data(path)
    converted_data = convert(data_clean, split)
    new_data = upsample_with_synonyms(converted_data, num_words)

    with open(filename, "w") as file:
        json.dump(new_data, file, indent=4)
