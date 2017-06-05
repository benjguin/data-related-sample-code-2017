

```python
from IPython.display import Image
```

# CNTK 599A: Sequence to Sequence Networks with Text Data


## Introduction and Background

This hands-on tutorial will take you through both the basics of sequence-to-sequence networks, and how to implement them in the Microsoft Cognitive Toolkit. In particular, we will implement a sequence-to-sequence model to perform grapheme to phoneme translation. We will start with some basic theory and then explain the data in more detail, and how you can download it.

Andrej Karpathy has a [nice visualization](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) of the five paradigms of neural network architectures:


```python
# Figure 1
Image(url="http://cntk.ai/jup/paradigms.jpg", width=750)
```




<img src="http://cntk.ai/jup/paradigms.jpg" width="750"/>



In this tutorial, we are going to be talking about the fourth paradigm: many-to-many, also known as sequence-to-sequence networks. The input is a sequence with a dynamic length, and the output is also a sequence with some dynamic length. It is the logical extension of the many-to-one paradigm in that previously we were predicting some category (which could easily be one of `V` words where `V` is an entire vocabulary) and now we want to predict a whole sequence of those categories.

The applications of sequence-to-sequence networks are nearly limitless. It is a natural fit for machine translation (e.g. English input sequences, French output sequences); automatic text summarization (e.g. full document input sequence, summary output sequence); word to pronunciation models (e.g. character [grapheme] input sequence, pronunciation [phoneme] output sequence); and even parse tree generation (e.g. regular text input, flat parse tree output).

## Basic theory

A sequence-to-sequence model consists of two main pieces: (1) an encoder; and (2) a decoder. Both the encoder and the decoder are recurrent neural network (RNN) layers that can be implemented using a vanilla RNN, an LSTM, or GRU cells (here we will use LSTM). In the basic sequence-to-sequence model, the encoder processes the input sequence into a fixed representation that is fed into the decoder as a context. The decoder then uses some mechanism (discussed below) to decode the processed information into an output sequence. The decoder is a language model that is augmented with some "strong context" by the encoder, and so each symbol that it generates is fed back into the decoder for additional context (like a traditional LM). For an English to German translation task, the most basic setup might look something like this:


```python
# Figure 2
Image(url="http://cntk.ai/jup/s2s.png", width=700)
```




<img src="http://cntk.ai/jup/s2s.png" width="700"/>



The basic sequence-to-sequence network passes the information from the encoder to the decoder by initializing the decoder RNN with the final hidden state of the encoder as its initial hidden state. The input is then a "sequence start" tag (`<s>` in the diagram above) which primes the decoder to start generating an output sequence. Then, whatever word (or note or image, etc.) it generates at that step is fed in as the input for the next step. The decoder keeps generating outputs until it hits the special "end sequence" tag (`</s>` above).

A more complex and powerful version of the basic sequence-to-sequence network uses an attention model. While the above setup works well, it can start to break down when the input sequences get long. At each step, the hidden state `h` is getting updated with the most recent information, and therefore `h` might be getting "diluted" in information as it processes each token. Further, even with a relatively short sequence, the last token will always get the last say and therefore the thought vector will be somewhat biased/weighted towards that last word. To deal with this problem, we use an "attention" mechanism that allows the decoder to look not only at all of the hidden states from the input, but it also learns which hidden states, for each step in decoding, to put the most weight on. We will discuss an attention implementation in a later version of this tutorial.

## Problem: Grapheme-to-Phoneme Conversion

The [grapheme](https://en.wikipedia.org/wiki/Grapheme) to [phoneme](https://en.wikipedia.org/wiki/Phoneme) problem is a translation task that takes the letters of a word as the input sequence (the graphemes are the smallest units of a writing system) and outputs the corresponding phonemes; that is, the units of sound that make up a language. In other words, the system aims to generate an unambigious representation of how to pronounce a given input word.

### Example

The graphemes or the letters are translated into corresponding phonemes: 

> **Grapheme** : **|** T **|** A **|** N **|** G **|** E **|** R **|**  
**Phonemes** : **|** ~T **|** ~AE **|** ~NG **|** ~ER **|** null **|** null **|** 




## Task and Model Structure

As discussed above, the task we are interested in solving is creating a model that takes some sequence as an input, and generates an output sequence based on the contents of the input. The model's job is to learn the mapping from the input sequence to the output sequence that it will generate. The job of the encoder is to come up with a good representation of the input that the decoder can use to generate a good output. For both the encoder and the decoder, the LSTM does a good job at this.

We will use the LSTM implementation from the CNTK Blocks library. This implements the "smarts" of the LSTM and we can more or less think of it as a black box. What is important to understand, however, is that there are two pieces to think of when implementing an RNN: the recurrence, which is the unrolled network over a sequence, and the block, which is the piece of the network run for each element of the sequence. We only need to implement the recurrence.

It helps to think of the recurrence as a function that keeps calling `step(x)` on the block (in our case, LSTM). At a high level, it looks like this:

```
class LSTM {
    float hidden_state

    init(initial_value):
        hidden_state = initial_value

    step(x):
        hidden_state = LSTM_function(x, hidden_state)
        return hidden_state
}
```

So, each call to the `step(x)` function takes some input `x`, modifies the internal `hidden_state`, and returns it. Therefore, with every input `x`, the value of the `hidden_state` evolves. Below we will import some required functionality, and then implement the recurrence that makes use of this mechanism.

## Importing CNTK and other useful libraries

CNTK is a Python module that contains several submodules like `io`, `learner`, `graph`, etc. We make extensive use of numpy as well.


```python
from __future__ import print_function
import numpy as np
import os
from cntk import Trainer, Axis
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT
from cntk.learners import momentum_sgd, momentum_as_time_constant_schedule, learning_rate_schedule, UnitType
from cntk import input, cross_entropy_with_softmax, classification_error, sequence, element_select, \
                 alias, hardmax, placeholder, combine, parameter, plus, times
from cntk.ops.functions import CloneMethod
from cntk.layers import LSTM, Stabilizer
from cntk.initializer import glorot_uniform

# Select the right target device when this notebook is being tested:
if 'TEST_DEVICE' in os.environ:
    import cntk
    if os.environ['TEST_DEVICE'] == 'cpu':
        cntk.device.set_default_device(cntk.device.cpu())
    else:
        cntk.device.set_default_device(cntk.device.gpu(0))

```

## Downloading the data

In this tutorial we will use a lightly pre-processed version of the CMUDict (version 0.7b) dataset from http://www.speech.cs.cmu.edu/cgi-bin/cmudict. The CMUDict data is the Carnegie Mellon University Pronouncing Dictionary is an open-source machine-readable pronunciation dictionary for North American English. The data is in the CNTKTextFormatReader format. Here is an example sequence pair from the data, where the input sequence (S0) is in the left column, and the output sequence (S1) is on the right:

```
0	|S0 3:1 |# <s>	|S1 3:1 |# <s>
0	|S0 4:1 |# A	|S1 32:1 |# ~AH
0	|S0 5:1 |# B	|S1 36:1 |# ~B
0	|S0 4:1 |# A	|S1 31:1 |# ~AE
0	|S0 7:1 |# D	|S1 38:1 |# ~D
0	|S0 12:1 |# I	|S1 47:1 |# ~IY
0	|S0 1:1 |# </s>	|S1 1:1 |# </s>
```

The code below will download the required files (training, the single sequence above for validation, and a small vocab file) and put them in a local folder (the training file is ~34 MB, testing is ~4MB, and the validation file and vocab file are both less than 1KB).


```python
import requests

def download(url, filename):
    """ utility function to download a file """
    response = requests.get(url, stream=True)
    with open(filename, "wb") as handle:
        for data in response.iter_content():
            handle.write(data)

data_dir = os.path.join('..', 'Examples', 'SequenceToSequence', 'CMUDict', 'Data')
# If above directory does not exist, just use current.
if not os.path.exists(data_dir):
    data_dir = '.'

valid_file = os.path.join(data_dir, 'tiny.ctf')
train_file = os.path.join(data_dir, 'cmudict-0.7b.train-dev-20-21.ctf')
vocab_file = os.path.join(data_dir, 'cmudict-0.7b.mapping')

files = [valid_file, train_file, vocab_file]

for file in files:
    if os.path.exists(file):
        print("Reusing locally cached: ", file)
    else:
        url = "https://github.com/Microsoft/CNTK/blob/v2.0.rc1/Examples/SequenceToSequence/CMUDict/Data/%s?raw=true"%file
        print("Starting download:", file)
        download(url, file)
        print("Download completed")

```

    Reusing locally cached:  ..\Examples\SequenceToSequence\CMUDict\Data\tiny.ctf
    Reusing locally cached:  ..\Examples\SequenceToSequence\CMUDict\Data\cmudict-0.7b.train-dev-20-21.ctf
    Reusing locally cached:  ..\Examples\SequenceToSequence\CMUDict\Data\cmudict-0.7b.mapping


### Select the notebook run mode

There are two run modes:
- *Fast mode*: `isFast` is set to `True`. This is the default mode for the notebooks, which means we train for fewer iterations or train / test on limited data. This ensures functional correctness of the notebook though the models produced are far from what a completed training would produce.

- *Slow mode*: We recommend the user to set this flag to `False` once the user has gained familiarity with the notebook content and wants to gain insight from running the notebooks for a longer period with different parameters for training. 


```python
isFast = True
```

## Reader

To efficiently collect our data, randomize it for training, and pass it to the network, we use the CNTKTextFormat reader. We will create a small function that will be called when training (or testing) that defines the names of the streams in our data, and how they are referred to in the raw training data.


```python
# Helper function to load the model vocabulary file
def get_vocab(path):
    # get the vocab for printing output sequences in plaintext
    vocab = [w.strip() for w in open(path).readlines()]
    i2w = { i:ch for i,ch in enumerate(vocab) }

    return (vocab, i2w)

# Read vocabulary data and generate their corresponding indices
vocab, i2w = get_vocab(vocab_file)

input_vocab_size = len(vocab)
label_vocab_size = len(vocab)
```


```python
# Print vocab and the correspoding mapping to the phonemes
print("Vocabulary size is", len(vocab))
print("First 15 letters are:")
print(vocab[:15])
print()
print("Print dictionary with the vocabulary mapping:")
print(i2w)
```

    Vocabulary size is 69
    First 15 letters are:
    ["'", '</s>', '<s/>', '<s>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    
    Print dictionary with the vocabulary mapping:
    {0: "'", 1: '</s>', 2: '<s/>', 3: '<s>', 4: 'A', 5: 'B', 6: 'C', 7: 'D', 8: 'E', 9: 'F', 10: 'G', 11: 'H', 12: 'I', 13: 'J', 14: 'K', 15: 'L', 16: 'M', 17: 'N', 18: 'O', 19: 'P', 20: 'Q', 21: 'R', 22: 'S', 23: 'T', 24: 'U', 25: 'V', 26: 'W', 27: 'X', 28: 'Y', 29: 'Z', 30: '~AA', 31: '~AE', 32: '~AH', 33: '~AO', 34: '~AW', 35: '~AY', 36: '~B', 37: '~CH', 38: '~D', 39: '~DH', 40: '~EH', 41: '~ER', 42: '~EY', 43: '~F', 44: '~G', 45: '~HH', 46: '~IH', 47: '~IY', 48: '~JH', 49: '~K', 50: '~L', 51: '~M', 52: '~N', 53: '~NG', 54: '~OW', 55: '~OY', 56: '~P', 57: '~R', 58: '~S', 59: '~SH', 60: '~T', 61: '~TH', 62: '~UH', 63: '~UW', 64: '~V', 65: '~W', 66: '~Y', 67: '~Z', 68: '~ZH'}


We will use the above to create a reader for our training data. Let's create it now:


```python
def create_reader(path, randomize, size=INFINITELY_REPEAT):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        features  = StreamDef(field='S0', shape=input_vocab_size, is_sparse=True),
        labels    = StreamDef(field='S1', shape=label_vocab_size, is_sparse=True)
    )), randomize=randomize, epoch_size = size)

# Train data reader
train_reader = create_reader(train_file, True)

# Validation/Test data reader
valid_reader = create_reader(valid_file, False)
```

### Now let's set our model hyperparameters...

Our input vocabulary size is 69, and those ones represent the label as well. Additionally we have 1 hidden layer with 128 nodes.


```python
model_dir = "." # we downloaded our data to the local directory above # TODO check me

# model dimensions
input_vocab_dim  = input_vocab_size
label_vocab_dim  = label_vocab_size
hidden_dim = 128
num_layers = 1
```

## Step 1: setup the input to the network

### Dynamic axes in CNTK (Key concept)

One of the important concepts in understanding CNTK is the idea of two types of axes:
- **static axes**, which are the traditional axes of a variable's shape, and
- **dynamic axes**, which have dimensions that are unknown until the variable is bound to real data at computation time.

The dynamic axes are particularly important in the world of recurrent neural networks. Instead of having to decide a maximum sequence length ahead of time, padding your sequences to that size, and wasting computation, CNTK's dynamic axes allow for variable sequence lengths that are automatically packed in minibatches to be as efficient as possible.

When setting up sequences, there are *two dynamic axes* that are important to consider. The first is the *batch axis*, which is the axis along which multiple sequences are batched. The second is the dynamic axis particular to that sequence. The latter is specific to a particular input because of variable sequence lengths in your data. For example, in sequence to sequence networks, we have two sequences: the **input sequence**, and the **output (or 'label') sequence**. One of the things that makes this type of network so powerful is that the length of the input sequence and the output sequence do not have to correspond to each other. Therefore, both the input sequence and the output sequence require their own unique dynamic axis.

When defining the input to a network, we set up the required dynamic axes and the shape of the input variables. Below, we define the shape (vocabulary size) of the inputs, create their dynamic axes, and finally create input variables that represent input nodes in our network.


```python
# Source and target inputs to the model
input_seq_axis = Axis('inputAxis')
label_seq_axis = Axis('labelAxis')

raw_input = sequence.input(shape=(input_vocab_dim), sequence_axis=input_seq_axis, name='raw_input')

raw_labels = sequence.input(shape=(label_vocab_dim), sequence_axis=label_seq_axis, name='raw_labels')
```

### Questions

1. Why do the shapes of the input variables correspond to the size of our dictionaries in sequence to sequence networks?

## Step 2: define the network

As discussed before, the sequence-to-sequence network is, at its most basic, an RNN encoder followed by an RNN decoder, and a dense output layer. We could do this in a few lines with the layers library, but let's go through things in a little more detail without adding too much complexity. The first step is to perform some manipulations on the input data; let's look at the code below and then discuss what we're doing. 


```python
# Instantiate the sequence to sequence translation model
input_sequence = raw_input

# Drop the sentence start token from the label, for decoder training
label_sequence = sequence.slice(raw_labels,
                       1, 0, name='label_sequence') # <s> A B C </s> --> A B C </s>
label_sentence_start = sequence.first(raw_labels)   # <s>

is_first_label = sequence.is_first(label_sequence)  # 1 0 0 0 ...
label_sentence_start_scattered = sequence.scatter(  # <s> 0 0 0 ... (up to the length of label_sequence)
    label_sentence_start, is_first_label)
```

We have two input variables, `raw_input` and `raw_labels`. Typically, the labels would not have to be part of the network definition because they would only be used in a criterion node when we compare the network's output with the ground truth. However, in sequence-to-sequence networks, the labels themselves form part of the input to the network during training as they are fed as the input into the decoder.

To make use of these input variables, we will pass them through computation nodes. We first set `input_sequence` to `raw_input` as a convenience step. We then perform several modifications to `label_sequence` so that it will work with our network. For now you'll just have to trust that we will make good use of this stuff later.

First, we slice the first element off of `label_sequence` so that it's missing the sentence-start token. This is because the decoder will always first be primed with that token, both during training and evaluation. When the ground truth isn't fed into the decoder, we will still feed in a sentence-start token, so we want to consistently view the input to the decoder as a sequence that starts with an actual value.

Then, we get `label_sequence_start` by getting the `first` element from the sequence `raw_labels`. This will be used to compose a sequence that is the first input to the decoder regardless of whether we're training or decoding. Finally, the last two statements set up an actual sequence, with the correct dynamic axis, to be fed into the decoder. The function `sequence.scatter` takes the contents of `label_sentence_start` (which is `<s>`) and turns it into a sequence with the first element containing the sequence start symbol and the rest of the elements containing 0's.

### Let's create the LSTM recurrence


```python
def LSTM_layer(input, output_dim, recurrence_hook_h=sequence.past_value, recurrence_hook_c=sequence.past_value):
    # we first create placeholders for the hidden state and cell state which we don't have yet
    dh = placeholder(shape=(output_dim), dynamic_axes=input.dynamic_axes)
    dc = placeholder(shape=(output_dim), dynamic_axes=input.dynamic_axes)

    # we now create an LSTM_cell function and call it with the input and placeholders
    LSTM_cell = LSTM(output_dim)
    f_x_h_c = LSTM_cell(dh, dc, input)
    h_c = f_x_h_c.outputs

    # we setup the recurrence by specifying the type of recurrence (by default it's `past_value` -- the previous value)
    h = recurrence_hook_h(h_c[0])
    c = recurrence_hook_c(h_c[1])

    replacements = { dh: h.output, dc: c.output }
    f_x_h_c.replace_placeholders(replacements)

    h = f_x_h_c.outputs[0]
    c = f_x_h_c.outputs[1]

    # and finally we return the hidden state and cell state as functions (by using `combine`)
    return combine([h]), combine([c])
```

### Exercise 1: Create the encoder

We will use the LSTM recurrence that we defined just above. Remember that its function signature is:

`def LSTM_layer(input, output_dim, recurrence_hook_h=sequence.past_value, recurrence_hook_c=sequence.past_value):`

and it returns a tuple `(hidden_state, hidden_cell)`. We will complete the following four exercises below. If possible, try them out before looking at the answers.

1. Create the encoder (set the `output_dim` and `cell_dim` to `hidden_dim` which we defined earlier).
2. Set `num_layers` to something higher than 1 and create a stack of LSTMs to represent the encoder.
3. Get the output of the encoder and put it into the right form to be passed into the decoder [hard]
4. Reverse the order of the `input_sequence` (this has been shown to help especially in machine translation)


```python
# 1.
# Create the encoder (set the output_dim to hidden_dim which we defined earlier).

(encoder_output_h, encoder_output_c) = LSTM_layer(input_sequence, hidden_dim)

# 2.
# Set num_layers to something higher than 1 and create a stack of LSTMs to represent the encoder.
num_layers = 2
output_h = alias(input_sequence) # get a copy of the input_sequence
for i in range(0, num_layers):
    (output_h, output_c) = LSTM_layer(output_h.output, hidden_dim)

# 3.
# Get the output of the encoder and put it into the right form to be passed into the decoder [hard]
thought_vector_h = sequence.first(output_h)
thought_vector_c = sequence.first(output_c)

thought_vector_broadcast_h = sequence.broadcast_as(thought_vector_h, label_sequence)
thought_vector_broadcast_c = sequence.broadcast_as(thought_vector_c, label_sequence)

# 4.
# Reverse the order of the input_sequence (this has been shown to help especially in machine translation)
(encoder_output_h, encoder_output_c) = LSTM_layer(input_sequence, hidden_dim, sequence.future_value, sequence.future_value)
```

### Exercise 2: Create the decoder

In our basic version of the sequence-to-sequence network, the decoder generates an output sequence given the input sequence by setting the initial state of the decoder to the final hidden state of the encoder. The hidden state is represented by a tuple `(encoder_h, encoder_c)` where `h` represents the output hidden state and `c` represents the value of the LSTM cell.

Besides setting the initial state of the decoder, we also need to give the decoder LSTM some input. The first element will always be the special sequence start tag `<s>`. After that, there are two ways that we want to wire up the decoder's input: one during training, and the other during evaluation (i.e. generating sequences on the trained network).

For training, the input to the decoder is the output sequence from the training data, also known as the label(s) for the input sequence. During evaluation, we will instead redirect the output from the network back into the decoder as its history. Let's first set up the input for training...


```python
decoder_input = element_select(is_first_label, label_sentence_start_scattered, sequence.past_value(label_sequence))
```

Above, we use the function `element_select` which will return one of two options given the condition `is_first_label`. Remember that we're working with sequences so when the decoder LSTM is run its input will be unrolled along with the network. The above allows us to to have a dynamic input that will return a specific element given what time step we're currently processing.

Therefore, the `decoder_input` will be `label_sentence_start_scattered` (which is simply `<s>`) when we are at the first time step, and otherwise it will return the `past_value` (i.e. the previous element given what time step we're currently at) of `label_sequence`.

Next, we need to setup our actual decoder. Before, for the encoder, we did the following:


```python
(output_h, output_c) = LSTM_layer(input_sequence, hidden_dim,
                                  recurrence_hook_h=sequence.past_value, recurrence_hook_c=sequence.past_value)
```

To be able to set the first hidden state of the decoder to be equal to the final hidden state of the encoder, we can leverage the parameters `recurrence_hookH` and `recurrent_hookC`. The default `past_value` is a function that returns, for time `t`, the element in the sequence at time `t-1`. See if you can figure out how to set that up.

1. Create the recurrence hooks for the decoder LSTM.
 * Hint: you'll have to create a `lambda operand:` and you will make use of the `is_first_label` mask we used earlier and the `thought_vector_broadcast_h` and `thought_vector_broadcast_c` representations of the output of the encoder.

2. With your recurrence hooks, create the decoder.
 * Hint: again we'll use the `LSTMP_component_with_self_stabilization()` function and again use `hidden_dim` for the `output_dim` and `cell_dim`.

3. Create a decoder with multiple layers. Note that you will have to use different recurrence hooks for the lower layers that feed back into the stack of layers.


```python
# 1.
# Create the recurrence hooks for the decoder LSTM.

recurrence_hook_h = lambda operand: element_select(is_first_label, thought_vector_broadcast_h, sequence.past_value(operand))
recurrence_hook_c = lambda operand: element_select(is_first_label, thought_vector_broadcast_c, sequence.past_value(operand))

# 2.
# With your recurrence hooks, create the decoder.

(decoder_output_h, decoder_output_c) = LSTM_layer(decoder_input, hidden_dim, recurrence_hook_h, recurrence_hook_c)

# 3.
# Create a decoder with multiple layers.
# Note that you will have to use different recurrence hooks for the lower layers

num_layers = 3
decoder_output_h = alias(decoder_input)
for i in range(0, num_layers):
    if (i > 0):
        recurrence_hook_h = sequence.past_value
        recurrence_hook_c = sequence.past_value
    else:
        recurrence_hook_h = lambda operand: element_select(
            is_first_label, thought_vector_broadcast_h, sequence.past_value(operand))
        recurrence_hook_c = lambda operand: element_select(
            is_first_label, thought_vector_broadcast_c, sequence.past_value(operand))

    (decoder_output_h, decoder_output_c) = LSTM_layer(decoder_output_h.output, hidden_dim,
                                                      recurrence_hook_h, recurrence_hook_c)
```

### Exercise 3: Fully connected layer (network output)

Now we're almost at the end of defining the network. All we need to do is take the output of the decoder, and run it through a linear layer. Ultimately it will be put into a `softmax` to get a probability distribution over the possible output words. However, we will include that as part of our criterion nodes (below).

1. Add the linear layer (a weight matrix, a bias parameter, a times, and a plus) to get the final output of the network


```python
# 1.
# Add the linear layer

W = parameter(shape=(decoder_output_h.shape[0], label_vocab_dim), init=glorot_uniform())
B = parameter(shape=(label_vocab_dim), init=0)
z = plus(B, times(decoder_output_h, W))
```

## Putting the model together

With the above we have defined some of the network and asked you to define parts of it as exercises. Here let's put the whole thing into a function called `create_model()`. Remember, all this does is create a skeleton of the network that defines how data will flow through it. No data is running through it yet.


```python
def create_model():

    # Source and target inputs to the model
    input_seq_axis = Axis('inputAxis')
    label_seq_axis = Axis('labelAxis')

    raw_input = sequence.input(
        shape=(input_vocab_dim), sequence_axis=input_seq_axis, name='raw_input')

    raw_labels = sequence.input(
        shape=(label_vocab_dim), sequence_axis=label_seq_axis, name='raw_labels')

    # Instantiate the sequence to sequence translation model
    input_sequence = raw_input

    # Drop the sentence start token from the label, for decoder training
    label_sequence = sequence.slice(raw_labels, 1, 0,
                                    name='label_sequence') # <s> A B C </s> --> A B C </s>
    label_sentence_start = sequence.first(raw_labels)      # <s>

    # Setup primer for decoder
    is_first_label = sequence.is_first(label_sequence)  # 1 0 0 0 ...
    label_sentence_start_scattered = sequence.scatter(
        label_sentence_start, is_first_label)

    # Encoder
    stabilize = Stabilizer()
    encoder_output_h = stabilize(input_sequence)
    for i in range(0, num_layers):
        (encoder_output_h, encoder_output_c) = LSTM_layer(
            encoder_output_h.output, hidden_dim, sequence.future_value, sequence.future_value)

    # Prepare encoder output to be used in decoder
    thought_vector_h = sequence.first(encoder_output_h)
    thought_vector_c = sequence.first(encoder_output_c)

    thought_vector_broadcast_h = sequence.broadcast_as(
        thought_vector_h, label_sequence)
    thought_vector_broadcast_c = sequence.broadcast_as(
        thought_vector_c, label_sequence)

    # Decoder
    decoder_history_hook = alias(label_sequence, name='decoder_history_hook') # copy label_sequence

    decoder_input = element_select(is_first_label, label_sentence_start_scattered, sequence.past_value(
        decoder_history_hook))

    decoder_output_h = stabilize(decoder_input)
    for i in range(0, num_layers):
        if (i > 0):
            recurrence_hook_h = sequence.past_value
            recurrence_hook_c = sequence.past_value
        else:
            recurrence_hook_h = lambda operand: element_select(
                is_first_label, thought_vector_broadcast_h, sequence.past_value(operand))
            recurrence_hook_c = lambda operand: element_select(
                is_first_label, thought_vector_broadcast_c, sequence.past_value(operand))

        (decoder_output_h, decoder_output_c) = LSTM_layer(
            decoder_output_h.output, hidden_dim, recurrence_hook_h, recurrence_hook_c)

    # Linear output layer
    W = parameter(shape=(decoder_output_h.shape[0], label_vocab_dim), init=glorot_uniform())
    B = parameter(shape=(label_vocab_dim), init=0)
    z = plus(B, times(stabilize(decoder_output_h), W))

    return z
```

## Training

Now that we've created the model, we are ready to train the network and learn its parameters. For sequence-to-sequence networks, the loss we use is cross-entropy. Note that we have to find the `label_sequences` node from the model because it was defined in our network and we want to compare the model's predictions specifically to the outputs of that node.


```python
model = create_model()
label_sequence = model.find_by_name('label_sequence')

# Criterion nodes
ce = cross_entropy_with_softmax(model, label_sequence)
errs = classification_error(model, label_sequence)

# let's show the required arguments for this model
print([x.name for x in model.arguments])
```

    ['raw_labels', 'raw_input']


Next, we'll setup a bunch of parameters to drive our learning, we'll create the learner, and finally create our trainer:


```python
# training parameters
lr_per_sample = learning_rate_schedule(0.007, UnitType.sample)
minibatch_size = 72
momentum_time_constant = momentum_as_time_constant_schedule(1100)
clipping_threshold_per_sample = 2.3
gradient_clipping_with_truncation = True
learner = momentum_sgd(model.parameters,
                        lr_per_sample, momentum_time_constant,
                        gradient_clipping_threshold_per_sample=clipping_threshold_per_sample,
                        gradient_clipping_with_truncation=gradient_clipping_with_truncation)
trainer = Trainer(model, (ce, errs), learner)
```

And now we bind the features and labels from our `train_reader` to the inputs that we setup in our network definition. First however, we'll define a convenience function to help find an argument name when pointing the reader's features to an argument of our model.


```python
# helper function to find variables by name
def find_arg_by_name(name, expression):
    vars = [i for i in expression.arguments if i.name == name]
    assert len(vars) == 1
    return vars[0]

train_bind = {
        find_arg_by_name('raw_input' , model) : train_reader.streams.features,
        find_arg_by_name('raw_labels', model) : train_reader.streams.labels
    }
```

Finally, we define our training loop and start training the network!


```python
training_progress_output_freq = 100
max_num_minibatch = 100 if isFast else 1000

for i in range(max_num_minibatch):
    # get next minibatch of training data
    mb_train = train_reader.next_minibatch(minibatch_size, input_map=train_bind)
    trainer.train_minibatch(mb_train)

    # collect epoch-wide stats
    if i % training_progress_output_freq == 0:
        print("Minibatch: {0}, Train Loss: {1:.3f}, Train Evaluation Criterion: {2:2.3f}".format(i,
                        trainer.previous_minibatch_loss_average, trainer.previous_minibatch_evaluation_average))
```

    Minibatch: 0, Train Loss: 4.234, Train Evaluation Criterion: 0.982


## Model evaluation: greedy decoding

Once we have a trained model, we of course then want to make use of it to generate output sequences! In this case, we will use greedy decoding. What this means is that we will run an input sequence through our trained network, and when we generate the output sequence, we will do so one element at a time by taking the `hardmax()` of the output of our network. This is obviously not optimal in general. Given the context, some word may always be the most probable at the first step, but another first word may be preferred given what is output later on. Decoding the optimal sequence is intractable in general. But we can do better doing a beam search where we keep around some small number of hypotheses at each step. However, greedy decoding can work surprisingly well for sequence-to-sequence networks because so much of the context is kept around in the RNN.

To do greedy decoding, we need to hook in the previous output of our network as the input to the decoder. During training we passed the `label_sequences` (ground truth) in. You'll notice in our `create_model()` function above the following lines:


```python
decoder_history_hook = alias(label_sequence, name='decoder_history_hook') # copy label_sequence
decoder_input = element_select(is_first_label, label_sentence_start_scattered, sequence.past_value(decoder_history_hook))
```

This gives us a way to modify the `decoder_history_hook` after training to something else. We've already trained our network, but now we need a way to evaluate it without using a ground truth. We can do that like this:


```python
model = create_model()

# get some references to the new model
label_sequence = model.find_by_name('label_sequence')
decoder_history_hook = model.find_by_name('decoder_history_hook')

# and now replace the output of decoder_history_hook with the hardmax output of the network
def clone_and_hook():
    # network output for decoder history
    net_output = hardmax(model)

    # make a clone of the graph where the ground truth is replaced by the network output
    return model.clone(CloneMethod.share, {decoder_history_hook.output : net_output.output})

# get a new model that uses the past network output as input to the decoder
new_model = clone_and_hook()
```

The `new_model` now contains a version of the original network that shares parameters with it but that has a different input to the decoder. Namely, instead of feeding the ground truth labels into the decoder, it will feed in the history that the network has generated!

Finally, let's see what it looks like if we train, and keep evaluating the network's output every `100` iterations by running a word's graphemes ('A B A D I') through our network. This way we can visualize the progress learning the best model... First we'll define a more complete `train()` action. It is largely the same as above but has some additional training parameters included; some additional smarts for printing out statistics as we go along; we now see progress over our data as epochs (one epoch is one complete pass over the training data); and we setup a reader for the single validation sequence we described above so that we can visually see our network's progress on that sequence as it learns.


```python
########################
# train action         #
########################

def train(train_reader, valid_reader, vocab, i2w, model, max_epochs):

    # do some hooks that we won't need in the future
    label_sequence = model.find_by_name('label_sequence')
    decoder_history_hook = model.find_by_name('decoder_history_hook')

    # Criterion nodes
    ce = cross_entropy_with_softmax(model, label_sequence)
    errs = classification_error(model, label_sequence)

    def clone_and_hook():
        # network output for decoder history
        net_output = hardmax(model)

        # make a clone of the graph where the ground truth is replaced by the network output
        return model.clone(CloneMethod.share, {decoder_history_hook.output : net_output.output})

    # get a new model that uses the past network output as input to the decoder
    new_model = clone_and_hook()

    # Instantiate the trainer object to drive the model training
    lr_per_sample = learning_rate_schedule(0.007, UnitType.sample)
    minibatch_size = 72
    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    clipping_threshold_per_sample = 2.3
    gradient_clipping_with_truncation = True
    learner = momentum_sgd(model.parameters,
                           lr_per_sample, momentum_time_constant,
                           gradient_clipping_threshold_per_sample=clipping_threshold_per_sample,
                           gradient_clipping_with_truncation=gradient_clipping_with_truncation)
    trainer = Trainer(model, (ce, errs), learner)

    # Get minibatches of sequences to train with and perform model training
    i = 0
    mbs = 0

    # Set epoch size to a larger number for lower training error
    epoch_size = 5000 if isFast else 908241

    training_progress_output_freq = 100

    # bind inputs to data from readers
    train_bind = {
        find_arg_by_name('raw_input' , model) : train_reader.streams.features,
        find_arg_by_name('raw_labels', model) : train_reader.streams.labels
    }
    valid_bind = {
        find_arg_by_name('raw_input' , new_model) : valid_reader.streams.features,
        find_arg_by_name('raw_labels', new_model) : valid_reader.streams.labels
    }

    for epoch in range(max_epochs):
        loss_numer = 0
        metric_numer = 0
        denom = 0

        while i < (epoch+1) * epoch_size:
            # get next minibatch of training data
            mb_train = train_reader.next_minibatch(minibatch_size, input_map=train_bind)
            trainer.train_minibatch(mb_train)

            # collect epoch-wide stats
            samples = trainer.previous_minibatch_sample_count
            loss_numer += trainer.previous_minibatch_loss_average * samples
            metric_numer += trainer.previous_minibatch_evaluation_average * samples
            denom += samples

            # every N MBs evaluate on a test sequence to visually show how we're doing; also print training stats
            if mbs % training_progress_output_freq == 0:

                print("Minibatch: {0}, Train Loss: {1:2.3f}, Train Evaluation Criterion: {2:2.3f}".format(mbs,
                      trainer.previous_minibatch_loss_average, trainer.previous_minibatch_evaluation_average))

                mb_valid = valid_reader.next_minibatch(minibatch_size, input_map=valid_bind)
                e = new_model.eval(mb_valid)
                print_sequences(e, i2w)

            i += mb_train[find_arg_by_name('raw_labels', model)].num_samples
            mbs += 1

        print("--- EPOCH %d DONE: loss = %f, errs = %f ---" % (epoch, loss_numer/denom, 100.0*(metric_numer/denom)))
        return 100.0*(metric_numer/denom)
```

Now that we have our three important functions defined -- `create_model()` and `train()`, let's make use of them:


```python
# Given a vocab and tensor, print the output
def print_sequences(sequences, i2w):
    for s in sequences:
        print([i2w[np.argmax(w)] for w in s], sep=" ")

# hook up data
train_reader = create_reader(train_file, True)
valid_reader = create_reader(valid_file, False)
vocab, i2w = get_vocab(vocab_file)

# create model
model = create_model()

# train
error = train(train_reader, valid_reader, vocab, i2w, model, max_epochs=1)
```

    Minibatch: 0, Train Loss: 4.234, Train Evaluation Criterion: 1.000
    ['</s>', '</s>', '</s>', '</s>', '</s>', 'J']
    ['</s>', '</s>', '</s>', '</s>', '</s>', 'J']
    ['</s>', '</s>', '</s>', '</s>', '</s>', 'J']
    ['</s>', '</s>', '</s>', '</s>', '</s>', 'J']
    ['</s>', '</s>', '</s>', '</s>', '</s>', 'J']
    ['</s>', '</s>', '</s>', '</s>', '</s>', 'J']
    ['</s>', '</s>', '</s>', '</s>', '</s>', 'J']
    ['</s>', '</s>', '</s>', '</s>', '</s>', 'J']
    ['</s>', '</s>', '</s>', '</s>', '</s>', 'J']
    ['</s>', '</s>', '</s>', '</s>', '</s>', 'J']
    --- EPOCH 0 DONE: loss = 3.827359, errs = 86.811952 ---



```python
# Print the training error
print(error)
```

    86.81195237025388


## Task
Note the error is very high. This is largely due to the minimum training we have done so far. Please change the `epoch_size` to be a much higher number and re-run the `train` function. This might take considerably longer time but you will see a marked reduction in the error.

## Next steps

An important extension to sequence-to-sequence models, especially when dealing with long sequences, is to use an attention mechanism. The idea behind attention is to allow the decoder, first, to look at any of the hidden state outputs from the encoder (instead of using only the final hidden state), and, second, to learn how much attention to pay to each of those hidden states given the context. This allows the outputted word at each time step `t` to depend not only on the final hidden state and the word that came before it, but instead on a weighted combination of *all* of the input hidden states!

In the next version of this tutorial, we will talk about how to include attention in your sequence to sequence network.


```python

```
