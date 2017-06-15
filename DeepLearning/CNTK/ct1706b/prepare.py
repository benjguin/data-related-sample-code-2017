import collections
import os

# variables you may want to change
## common values for prepare.py and main.py
vocabulary_file_path  = "./data/vocab.txt"
train_ctf_file_path = "./data/train.ctf"
test_ctf_file_path = "./data/test.ctf"
tiny_ctf_file_path = "./data/tiny.ctf"
## values that are only needed in prepare.py
train_file_path = "./data/train.txt"
test_file_path = "./data/test.txt"
tiny_file_path = "./data/tiny.txt"
vocabulary_size = 10000
seq_start = "__{__" # has to be a palindrome
seq_end = "__}__" # has to be another palindrome
vocabulary_unknown = "__?__" # doesn't have to be a palindrome

# get sample text, compute target by inverting words, and prepare it for sequence to sequence

def get_words(file_path):
    input_words=output_words=[]
    with open(file_path, "r") as f:
        for line in f:
            input_words.append(seq_start)
            input_words += [w.strip() for w in line.split()]
            input_words.append(seq_end)
    output_words = [w[::-1] for w in input_words]
    return input_words, output_words

train_input_words, train_output_words = get_words(train_file_path)
test_input_words, test_output_words = get_words(test_file_path)
tiny_input_words, tiny_output_words = get_words(tiny_file_path)

for w in tiny_input_words[0:20]:
    print(w)

print("-----------")
    
for w in tiny_output_words[0:20]:
    print(w)

## define vocabulary with most common source and target words

def build_dataset(words):
  count = [[vocabulary_unknown, -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary[vocabulary_unknown]
      unk_count = unk_count + 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
  return data, count, dictionary, reverse_dictionary

## vocabulary is computed from all files including test file as we assume we woulmd be able to get most of the vocabulary in advance in a specific domain
data, count, dictionary, reverse_dictionary = build_dataset(
    [seq_start, seq_end]
    + train_input_words + train_output_words 
    + test_input_words + test_output_words
    + tiny_input_words + tiny_output_words)
print('Most common words (+UNK)', count[:10])
print('Sample data', data[:10])
del data
del count

# write vocabulary to a file as CNTK format: 1 word per line, in order

with open(vocabulary_file_path, "w") as f:
    for i in sorted(dictionary, key=dictionary.__getitem__):
        print(i, file=f)

# write the training file in CTF format
## S0 in the source sequence, S1 is the target sequence

def get_index_and_word(word):
    if word in dictionary:
        return dictionary[word], word
    else:
        return 0, word + " -> " + vocabulary_unknown

def write_in_ctf_format(input_words, output_words, file_path):
    s=-1
    with open(file_path, "w") as f:
        for i, w1 in enumerate(input_words):
            w2 = output_words[i]
            if w1 == seq_start:
                s = s+1
            inputindex, inputword = get_index_and_word(w1)
            outputindex, outputword = get_index_and_word(w2)
            print("%s |S0 %s:1 |# %s |S1 %s:1 |# %s"%(
                str(s), 
                str(inputindex), inputword,
                str(outputindex), outputword),
                file=f)

write_in_ctf_format(train_input_words, train_output_words, train_ctf_file_path)
write_in_ctf_format(test_input_words, test_output_words, test_ctf_file_path)
write_in_ctf_format(tiny_input_words, tiny_output_words, tiny_ctf_file_path)

print("files were written. Here is the tiny.ctf content:")
print(open(tiny_ctf_file_path, "r").read())