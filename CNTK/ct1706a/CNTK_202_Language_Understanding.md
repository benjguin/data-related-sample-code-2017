
# Hands-On Lab: Language Understanding with Recurrent Networks

This hands-on lab shows how to implement a recurrent network to process text,
for the [Air Travel Information Services](https://catalog.ldc.upenn.edu/LDC95S26) 
(ATIS) task of slot tagging (tag individual words to their respective classes, 
where the classes are provided as labels in the training data set).
We will start with a straight-forward embedding of the words followed by a recurrent LSTM.
This will then be extended to include neighboring words and run bidirectionally.
Lastly, we will turn this system into an intent classifier.  

The techniques you will practice are:
* model description by composing layer blocks, a convenient way to compose 
  networks/models without requiring the need to write formulas,
* creating your own layer block
* variables with different sequence lengths in the same network
* training the network

We assume that you are familiar with basics of deep learning, and these specific concepts:
* recurrent networks ([Wikipedia page](https://en.wikipedia.org/wiki/Recurrent_neural_network))
* text embedding ([Wikipedia page](https://en.wikipedia.org/wiki/Word_embedding))

### Prerequisites

We assume that you have already [installed CNTK](https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-on-your-machine).
This tutorial requires CNTK V2. We strongly recommend to run this tutorial on a machine with
a capable CUDA-compatible GPU. Deep learning without GPUs is not fun.

#### Downloading the data

In this tutorial we are going to use a (lightly preprocessed) version of the ATIS dataset. You can download the data automatically by running the cells below or by executing the manual instructions.

#### Fallback manual instructions
Download the ATIS [training](https://github.com/Microsoft/CNTK/blob/v2.0.rc1/Tutorials/SLUHandsOn/atis.train.ctf) 
and [test](https://github.com/Microsoft/CNTK/blob/v2.0.rc1/Tutorials/SLUHandsOn/atis.test.ctf) 
files and put them at the same folder as this notebook. If you want to see how the model is 
predicting on new sentences you will also need the vocabulary files for 
[queries](https://github.com/Microsoft/CNTK/blob/v2.0.rc1/Examples/LanguageUnderstanding/ATIS/BrainScript/query.wl) and
[slots](https://github.com/Microsoft/CNTK/blob/v2.0.rc1/Examples/LanguageUnderstanding/ATIS/BrainScript/slots.wl)


```python
from __future__ import print_function
import requests
import os

def download(url, filename):
    """ utility function to download a file """
    response = requests.get(url, stream=True)
    with open(filename, "wb") as handle:
        for data in response.iter_content():
            handle.write(data)

locations = ['Tutorials/SLUHandsOn', 'Examples/LanguageUnderstanding/ATIS/BrainScript']

data = {
  'train': { 'file': 'atis.train.ctf', 'location': 0 },
  'test': { 'file': 'atis.test.ctf', 'location': 0 },
  'query': { 'file': 'query.wl', 'location': 1 },
  'slots': { 'file': 'slots.wl', 'location': 1 }
}

for item in data.values():
    location = locations[item['location']]
    path = os.path.join('..', location, item['file'])
    if os.path.exists(path):
        print("Reusing locally cached:", item['file'])
        # Update path
        item['file'] = path
    elif os.path.exists(item['file']):
        print("Reusing locally cached:", item['file'])
    else:
        print("Starting download:", item['file'])
        url = "https://github.com/Microsoft/CNTK/blob/v2.0.rc1/%s/%s?raw=true"%(location, item['file'])
        download(url, item['file'])
        print("Download completed")

```

    Starting download: slots.wl
    Download completed
    Starting download: query.wl
    Download completed
    Starting download: atis.test.ctf
    Download completed
    Starting download: atis.train.ctf
    Download completed


#### Importing CNTK and other useful libraries

CNTK's Python module contains several submodules like `io`, `learner`, and `layers`. We also use NumPy in some cases since the results returned by CNTK work like NumPy arrays.


```python
import math
import numpy as np

import cntk as C
from cntk.logging import ProgressPrinter, log_number_of_parameters
from cntk.io import MinibatchSource, CTFDeserializer
from cntk.io import StreamDef, StreamDefs, INFINITELY_REPEAT
from cntk import *
from cntk.learners import fsadagrad, learning_rate_schedule
from cntk.layers import * # CNTK Layers library

# Select the right target device when this notebook is being tested:
if 'TEST_DEVICE' in os.environ:
    import cntk
    if os.environ['TEST_DEVICE'] == 'cpu':
        cntk.device.try_set_default_device(cntk.device.cpu())
    else:
        cntk.device.try_set_default_device(cntk.device.gpu(0))

```


```python
print(os.environ)
```

    environ({'HOME': '/home/benjguin', 'PYTHONPATH': '/opt/caffe/python:/opt/caffe2/build:', 'MPLBACKEND': 'module://ipykernel.pylab.backend_inline', 'LANG': 'en_US.UTF-8', 'PAGER': 'cat', 'SHELL': '/bin/bash', 'JPY_PARENT_PID': '74500', 'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin', 'USER': 'benjguin', 'TERM': 'xterm-color', 'GIT_PAGER': 'cat', 'CLICOLOR': '1', 'LD_LIBRARY_PATH': '/dsvm/tools/cntk/cntk/lib:/dsvm/tools/cntk/cntk/dependencies/lib:/usr/local/mpi/lib:/opt/acml5.3.1/ifort64/lib:/opt/acml5.3.1/ifort64_mp/lib::/usr/lib64/microsoft-r/8.0/lib64/R/lib:/opt/intel/tbb/lib/intel64/gcc4.7:/opt/intel/lib/intel64:/opt/intel/mkl/lib/intel64:/usr/local/cuda-8.0/lib64'})


## Task and Model Structure

The task we want to approach in this tutorial is slot tagging.
We use the [ATIS corpus](https://catalog.ldc.upenn.edu/LDC95S26).
ATIS contains human-computer queries from the domain of Air Travel Information Services,
and our task will be to annotate (tag) each word of a query whether it belongs to a
specific item of information (slot), and which one.

The data in your working folder has already been converted into the "CNTK Text Format."
Let's look at an example from the test-set file `atis.test.ctf`:

    19  |S0 178:1 |# BOS      |S1 14:1 |# flight  |S2 128:1 |# O
    19  |S0 770:1 |# show                         |S2 128:1 |# O
    19  |S0 429:1 |# flights                      |S2 128:1 |# O
    19  |S0 444:1 |# from                         |S2 128:1 |# O
    19  |S0 272:1 |# burbank                      |S2 48:1  |# B-fromloc.city_name
    19  |S0 851:1 |# to                           |S2 128:1 |# O
    19  |S0 789:1 |# st.                          |S2 78:1  |# B-toloc.city_name
    19  |S0 564:1 |# louis                        |S2 125:1 |# I-toloc.city_name
    19  |S0 654:1 |# on                           |S2 128:1 |# O
    19  |S0 601:1 |# monday                       |S2 26:1  |# B-depart_date.day_name
    19  |S0 179:1 |# EOS                          |S2 128:1 |# O

This file has 7 columns:

* a sequence id (19). There are 11 entries with this sequence id. This means that sequence 19 consists
of 11 tokens;
* column `S0`, which contains numeric word indices;
* a comment column denoted by `#`, to allow a human reader to know what the numeric word index stands for;
Comment columns are ignored by the system. `BOS` and `EOS` are special words
to denote beginning and end of sentence, respectively;
* column `S1` is an intent label, which we will only use in the last part of the tutorial;
* another comment column that shows the human-readable label of the numeric intent index;
* column `S2` is the slot label, represented as a numeric index; and
* another comment column that shows the human-readable label of the numeric label index.

The task of the neural network is to look at the query (column `S0`) and predict the
slot label (column `S2`).
As you can see, each word in the input gets assigned either an empty label `O`
or a slot label that begins with `B-` for the first word, and with `I-` for any
additional consecutive word that belongs to the same slot.

The model we will use is a recurrent model consisting of an embedding layer,
a recurrent LSTM cell, and a dense layer to compute the posterior probabilities:


    slot label   "O"        "O"        "O"        "O"  "B-fromloc.city_name"
                  ^          ^          ^          ^          ^
                  |          |          |          |          |
              +-------+  +-------+  +-------+  +-------+  +-------+
              | Dense |  | Dense |  | Dense |  | Dense |  | Dense |  ...
              +-------+  +-------+  +-------+  +-------+  +-------+
                  ^          ^          ^          ^          ^
                  |          |          |          |          |
              +------+   +------+   +------+   +------+   +------+   
         0 -->| LSTM |-->| LSTM |-->| LSTM |-->| LSTM |-->| LSTM |-->...
              +------+   +------+   +------+   +------+   +------+   
                  ^          ^          ^          ^          ^
                  |          |          |          |          |
              +-------+  +-------+  +-------+  +-------+  +-------+
              | Embed |  | Embed |  | Embed |  | Embed |  | Embed |  ...
              +-------+  +-------+  +-------+  +-------+  +-------+
                  ^          ^          ^          ^          ^
                  |          |          |          |          |
    w      ------>+--------->+--------->+--------->+--------->+------... 
                 BOS      "show"    "flights"    "from"   "burbank"

Or, as a CNTK network description. Please have a quick look and match it with the description above:
(descriptions of these functions can be found at: [the layers reference](http://cntk.ai/pythondocs/layerref.html))



```python
# number of words in vocab, slot labels, and intent labels
vocab_size = 943 ; num_labels = 129 ; num_intents = 26    

# model dimensions
input_dim  = vocab_size
label_dim  = num_labels
emb_dim    = 150
hidden_dim = 300

def create_model():
    with default_options(initial_state=0.1):
        return Sequential([
            Embedding(emb_dim, name='embed'),
            Recurrence(LSTM(hidden_dim), go_backwards=False),
            Dense(num_labels, name='classify')
        ])
```

Now we are ready to create a model and inspect it.



```python
# peek
model = create_model()
print(model.embed.E.shape)
print(model.classify.b.value)
```

    (-1, 150)
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.]


As you can see the attributes of the model are fully accessible from Python. The model has 3 layers. The first layer is an embedding and you can access its parameter `E` (where the embeddings are stored) like any other attribute of a Python object. Its shape contains a `-1` which indicates that this parameter is not fully specified yet. When we decide what data we will run through this network (very shortly) the shape will be the size of the input vocabulary. We also print the bias term in the last layer. Bias terms are by default initialized to 0 (but there's also a way to change that).


## CNTK Configuration

To train and test a model in CNTK, we need to create a model and specify how to read data and perform training and testing. 

In order to train we need to specify:

* how to read the data 
* the model function, its inputs, and outputs
* hyper-parameters for the learner such as the learning rate

[comment]: <> (For testing ...)

### A Brief Look at Data and Data Reading

We already looked at the data.
But how do you generate this format?
For reading text, this tutorial uses the `CNTKTextFormatReader`. It expects the input data to be
in a specific format, as described [here](https://github.com/Microsoft/CNTK/wiki/CNTKTextFormat-Reader).

For this tutorial, we created the corpora by two steps:
* convert the raw data into a plain text file that contains of TAB-separated columns of space-separated text. For example:

  ```
  BOS show flights from burbank to st. louis on monday EOS (TAB) flight (TAB) O O O O B-fromloc.city_name O B-toloc.city_name I-toloc.city_name O B-depart_date.day_name O
  ```

  This is meant to be compatible with the output of the `paste` command.
* convert it to CNTK Text Format (CTF) with the following command:

  ```
  python [CNTK root]/Scripts/txt2ctf.py --map query.wl intent.wl slots.wl --annotated True --input atis.test.txt --output atis.test.ctf
  ```
  where the three `.wl` files give the vocabulary as plain text files, one word per line.

In these CTF files, our columns are labeled `S0`, `S1`, and `S2`.
These are connected to the actual network inputs by the corresponding lines in the reader definition:


```python
def create_reader(path, is_training):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
         query         = StreamDef(field='S0', shape=vocab_size,  is_sparse=True),
         intent_unused = StreamDef(field='S1', shape=num_intents, is_sparse=True),  
         slot_labels   = StreamDef(field='S2', shape=num_labels,  is_sparse=True)
     )), randomize=is_training, max_sweeps = INFINITELY_REPEAT if is_training else 1)
```


```python
# peek
reader = create_reader(data['train']['file'], is_training=True)
reader.streams.keys()
```




    dict_keys(['intent_unused', 'slot_labels', 'query'])



### Trainer

We also must define the training criterion (loss function), and also an error metric to track. Below we make extensive use of `placeholders`. Remember that the code we have been writing is not actually executing any heavy computation it is just specifying the function we want to compute on data during training/testing. And in the same way that it is convenient to have names for arguments when you write a regular function in a programming language, it is convenient to have placeholders that refer to arguments (or local computations that need to be reused). Eventually, some other code will replace these placeholders with other known quantities in the same way that in a programming language the function will be called with concrete values bound to its arguments. 


```python
def create_criterion_function(model):
    labels = C.placeholder(name='labels')
    ce   = cross_entropy_with_softmax(model, labels)
    errs = classification_error      (model, labels)
    return combine ([ce, errs]) # (features, labels) -> (loss, metric)
```


```python
def train(reader, model, max_epochs=16):
    # criterion: (model args, labels) -> (loss, metric)
    #   here  (query, slot_labels) -> (ce, errs)
    criterion = create_criterion_function(model)

    criterion.replace_placeholders({criterion.placeholders[0]: C.sequence.input(vocab_size),
                                    criterion.placeholders[1]: C.sequence.input(num_labels)})

    # training config
    epoch_size = 18000        # 18000 samples is half the dataset size 
    minibatch_size = 70
    
    # LR schedule over epochs 
    # In CNTK, an epoch is how often we get out of the minibatch loop to
    # do other stuff (e.g. checkpointing, adjust learning rate, etc.)
    # (we don't run this many epochs, but if we did, these are good values)
    lr_per_sample = [0.003]*4+[0.0015]*24+[0.0003]
    lr_per_minibatch = [x * minibatch_size for x in lr_per_sample]
    lr_schedule = learning_rate_schedule(lr_per_minibatch, UnitType.minibatch, epoch_size)
    
    # Momentum
    momentum_as_time_constant = momentum_as_time_constant_schedule(700)
    
    # We use a variant of the FSAdaGrad optimizer which is known to work well on this dataset
    # Feel free to try other optimizers from 
    # https://www.cntk.ai/pythondocs/cntk.learner.html#module-cntk.learner
    learner = fsadagrad(criterion.parameters,
                        lr=lr_schedule, momentum=momentum_as_time_constant,
                        gradient_clipping_threshold_per_sample=15, gradient_clipping_with_truncation=True)

    # trainer
    progress_printer = ProgressPrinter(tag='Training', num_epochs=max_epochs)
    #progress_printer = ProgressPrinter(freq=100, first=10, tag='Training', num_epochs=max_epochs) # more detailed logging
    trainer = Trainer(model, criterion, learner, progress_printer)

    # process minibatches and perform model training
    log_number_of_parameters(model)

    t = 0
    for epoch in range(max_epochs):         # loop over epochs
        epoch_end = (epoch+1) * epoch_size
        while t < epoch_end:                # loop over minibatches on the epoch
            data = reader.next_minibatch(minibatch_size, input_map={  # fetch minibatch
                criterion.arguments[0]: reader.streams.query,
                criterion.arguments[1]: reader.streams.slot_labels
            })
            trainer.train_minibatch(data)                                     # update model with it
            t += data[criterion.arguments[1]].num_samples                     # samples so far
        trainer.summarize_training_progress()

```

### Running it

You can find the complete recipe below.


```python
def do_train():
    global model
    model = create_model()
    reader = create_reader(data['train']['file'], is_training=True)
    train(reader, model)
do_train()
```

    Training 721479 parameters in 6 parameter tensors.
    Finished Epoch[1 of 16]: [Training] loss = 1.097468 * 18010, metric = 20.61% * 18010 7.884s (2284.4 samples/s);
    Finished Epoch[2 of 16]: [Training] loss = 0.443539 * 18051, metric = 9.76% * 18051 2.985s (6047.2 samples/s);
    Finished Epoch[3 of 16]: [Training] loss = 0.294534 * 17941, metric = 6.16% * 17941 3.040s (5901.6 samples/s);
    Finished Epoch[4 of 16]: [Training] loss = 0.213083 * 18059, metric = 4.70% * 18059 3.168s (5700.4 samples/s);
    Finished Epoch[5 of 16]: [Training] loss = 0.158128 * 17957, metric = 3.52% * 17957 3.027s (5932.3 samples/s);
    Finished Epoch[6 of 16]: [Training] loss = 0.143503 * 18021, metric = 3.19% * 18021 3.073s (5864.3 samples/s);
    Finished Epoch[7 of 16]: [Training] loss = 0.117660 * 17980, metric = 2.55% * 17980 3.019s (5955.6 samples/s);
    Finished Epoch[8 of 16]: [Training] loss = 0.121787 * 18025, metric = 2.63% * 18025 3.062s (5886.7 samples/s);
    Finished Epoch[9 of 16]: [Training] loss = 0.082948 * 17956, metric = 1.88% * 17956 3.070s (5848.9 samples/s);
    Finished Epoch[10 of 16]: [Training] loss = 0.084909 * 18039, metric = 1.93% * 18039 2.989s (6035.1 samples/s);
    Finished Epoch[11 of 16]: [Training] loss = 0.091183 * 17966, metric = 2.13% * 17966 3.045s (5900.2 samples/s);
    Finished Epoch[12 of 16]: [Training] loss = 0.065457 * 18041, metric = 1.45% * 18041 3.032s (5950.2 samples/s);
    Finished Epoch[13 of 16]: [Training] loss = 0.069261 * 17984, metric = 1.49% * 17984 3.121s (5762.3 samples/s);
    Finished Epoch[14 of 16]: [Training] loss = 0.069089 * 17976, metric = 1.53% * 17976 3.113s (5774.5 samples/s);
    Finished Epoch[15 of 16]: [Training] loss = 0.061216 * 18030, metric = 1.35% * 18030 3.059s (5894.1 samples/s);
    Finished Epoch[16 of 16]: [Training] loss = 0.052744 * 18014, metric = 1.10% * 18014 3.144s (5729.6 samples/s);


This shows how learning proceeds over epochs (passes through the data).
For example, after four epochs, the loss, which is the cross-entropy criterion, has reached 0.22 as measured on the ~18000 samples of this epoch,
and that the error rate is 5.0% on those same 18000 training samples.

The epoch size is the number of samples--counted as *word tokens*, not sentences--to
process between model checkpoints.

Once the training has completed (a little less than 2 minutes on a Titan-X or a Surface Book),
you will see an output like this
```
Finished Epoch [16]: [Training] loss = 0.058111 * 18014, metric = 1.3% * 18014
```
which is the loss (cross entropy) and the metric (classification error) averaged over the final epoch.

On a CPU-only machine, it can be 4 or more times slower. You can try setting
```python
emb_dim    = 50 
hidden_dim = 100
```
to reduce the time it takes to run on a CPU, but the model will not fit as well as when the 
hidden and embedding dimension are larger. 

### Evaluating the model

Like the train() function, we also define a function to measure accuracy on a test set by computing the error over multiple minibatches of test data. For evaluating on a small sample read from a file, you can set a minibatch size reflecting the sample size and run the test_minibatch on that instance of data. To see how to evaluate a single sequence, we provide an instance later in the tutorial. 


```python
def evaluate(reader, model):
    criterion = create_criterion_function(model)
    criterion.replace_placeholders({criterion.placeholders[0]: C.sequence.input(num_labels)})

    # process minibatches and perform evaluation
    lr_schedule = learning_rate_schedule(1, UnitType.minibatch)
    momentum_as_time_constant = momentum_as_time_constant_schedule(0)
    dummy_learner = fsadagrad(criterion.parameters, 
                              lr=lr_schedule, momentum=momentum_as_time_constant)
    progress_printer = ProgressPrinter(tag='Evaluation', num_epochs=0)
    evaluator = Trainer(model, criterion, dummy_learner, progress_printer)

    while True:
        minibatch_size = 500
        data = reader.next_minibatch(minibatch_size, input_map={  # fetch minibatch
            criterion.arguments[0]: reader.streams.query,
            criterion.arguments[1]: reader.streams.slot_labels
        })
        if not data:                                 # until we hit the end
            break
        evaluator.test_minibatch(data)
    evaluator.summarize_test_progress()

```

Now we can measure the model accuracy by going through all the examples in the test set and using the ``test_minibatch`` method of the trainer created inside the evaluate function defined above. At the moment (when this tutorial was written) the Trainer constructor requires a learner (even if it is only used to perform ``test_minibatch``) so we have to specify a dummy learner. In the future it will be allowed to construct a Trainer without specifying a learner as long as the trainer only calls ``test_minibatch``


```python
def do_test():
    reader = create_reader(data['test']['file'], is_training=False)
    evaluate(reader, model)
do_test()
model.classify.b.value
```

    Finished Evaluation [1]: Minibatch[1-23]: metric = 2.75% * 10984;





    array([ -6.17747232e-02,   1.67661682e-01,   2.67170876e-01,
            -1.74592167e-01,  -8.95598158e-02,  -1.41288474e-01,
            -1.65669933e-01,  -3.37033480e-01,  -2.43424982e-01,
            -4.46879327e-01,  -1.68832123e-01,  -3.02033037e-01,
            -1.76276967e-01,  -2.74571359e-01,   1.39392167e-01,
             8.47950578e-02,  -5.46203852e-01,  -1.13420404e-01,
             2.31410816e-01,  -6.03151739e-01,   1.89564109e-01,
             1.24402538e-01,  -2.14792952e-01,  -4.18093294e-01,
            -3.74919444e-01,   1.71167210e-01,  -2.51746386e-01,
            -1.43728361e-01,  -9.00753140e-02,  -3.50904651e-02,
            -1.18319981e-01,  -1.28747910e-01,   5.05145304e-02,
            -1.58300754e-02,  -2.74066269e-01,   8.33014965e-01,
             2.71680862e-01,   2.87652820e-01,   5.84618114e-02,
            -8.77581835e-02,  -5.99530458e-01,   5.89623414e-02,
             4.13840562e-01,   2.55903274e-01,   5.06808162e-01,
            -1.23129576e-03,  -9.60508063e-02,  -1.53484032e-01,
             2.22164318e-01,  -1.28024980e-01,  -1.65894985e-01,
            -1.12369619e-01,  -1.55051336e-01,   2.41953552e-01,
             1.22186407e-01,  -4.66029614e-01,   9.18036923e-02,
            -2.58443445e-01,  -2.13418126e-01,  -1.93984509e-01,
            -4.68383223e-01,  -3.08870345e-01,  -2.98281580e-01,
            -4.93319243e-01,  -4.19607759e-01,  -4.19408023e-01,
             2.76399225e-01,  -2.33146802e-01,  -4.22037512e-01,
            -5.73692262e-01,  -5.13007700e-01,  -4.26108599e-01,
            -2.96079546e-01,  -4.26696450e-01,  -5.30181348e-01,
            -4.90620762e-01,  -1.50063828e-01,  -8.53933841e-02,
            -2.88960665e-01,  -3.32533330e-01,  -1.82914272e-01,
            -6.08624257e-02,   1.39558390e-01,  -3.55813771e-01,
             2.61407048e-02,  -3.05180550e-01,  -2.52452403e-01,
            -3.59835267e-01,  -3.46008331e-01,  -7.74581134e-02,
            -4.68599766e-01,  -4.85256277e-02,   8.09188373e-03,
            -1.18216984e-01,  -2.91808486e-01,  -3.23002934e-01,
            -5.39601818e-02,  -2.29025468e-01,  -2.62914360e-01,
            -2.39585891e-01,   3.86749119e-01,  -4.06216919e-01,
            -7.60370567e-02,  -1.73274606e-01,  -4.33509350e-01,
            -1.12799868e-01,  -5.64482331e-01,  -8.78847912e-02,
            -1.08817555e-01,   1.01803645e-01,  -1.71953231e-01,
            -1.97624683e-01,  -3.14228624e-01,  -5.24510860e-01,
            -4.37267184e-01,  -2.05273196e-01,  -4.08936590e-01,
            -4.90692347e-01,  -4.09333646e-01,   5.79493642e-02,
            -6.00475848e-01,  -2.65847147e-01,  -5.08210421e-01,
            -3.80596787e-01,   7.19665957e-04,  -3.00037086e-01,
            -2.07646459e-01,  -1.52841210e-01,   4.92503822e-01], dtype=float32)



The following block of code illustrates how to evaluate a single sequence. Additionally we show how one can pass in the information using NumPy arrays.


```python
# load dictionaries
query_wl = [line.rstrip('\n') for line in open(data['query']['file'])]
slots_wl = [line.rstrip('\n') for line in open(data['slots']['file'])]
query_dict = {query_wl[i]:i for i in range(len(query_wl))}
slots_dict = {slots_wl[i]:i for i in range(len(slots_wl))}

# let's run a sequence through
seq = 'BOS flights from new york to seattle EOS'
w = [query_dict[w] for w in seq.split()] # convert to word indices
print(w)
onehot = np.zeros([len(w),len(query_dict)], np.float32)
for t in range(len(w)):
    onehot[t,w[t]] = 1
pred = model.eval({model.arguments[0]:[onehot]})[0]
print(pred.shape)
best = np.argmax(pred,axis=1)
print(best)
list(zip(seq.split(),[slots_wl[s] for s in best]))
```

    [178, 429, 444, 619, 937, 851, 752, 179]
    (8, 129)
    [128 128 128  48 110 128  78 128]





    [('BOS', 'O'),
     ('flights', 'O'),
     ('from', 'O'),
     ('new', 'B-fromloc.city_name'),
     ('york', 'I-fromloc.city_name'),
     ('to', 'O'),
     ('seattle', 'B-toloc.city_name'),
     ('EOS', 'O')]



## Modifying the Model

In the following, you will be given tasks to practice modifying CNTK configurations.
The solutions are given at the end of this document... but please try without!

### A Word About [`Sequential()`](https://www.cntk.ai/pythondocs/layerref.html#sequential)

Before jumping to the tasks, let's have a look again at the model we just ran.
The model is described in what we call *function-composition style*.
```python
        Sequential([
            Embedding(emb_dim),
            Recurrence(LSTM(hidden_dim), go_backwards=False),
            Dense(num_labels)
        ])
```
You may be familiar with the "sequential" notation from other neural-network toolkits.
If not, [`Sequential()`](https://www.cntk.ai/pythondocs/layerref.html#sequential) is a powerful operation that,
in a nutshell, allows to compactly express a very common situation in neural networks
where an input is processed by propagating it through a progression of layers.
`Sequential()` takes an list of functions as its argument,
and returns a *new* function that invokes these functions in order,
each time passing the output of one to the next.
For example,
```python
	FGH = Sequential ([F,G,H])
    y = FGH (x)
```
means the same as
```
    y = H(G(F(x))) 
```
This is known as ["function composition"](https://en.wikipedia.org/wiki/Function_composition),
and is especially convenient for expressing neural networks, which often have this form:

         +-------+   +-------+   +-------+
    x -->|   F   |-->|   G   |-->|   H   |--> y
         +-------+   +-------+   +-------+

Coming back to our model at hand, the `Sequential` expression simply
says that our model has this form:

         +-----------+   +----------------+   +------------+
    x -->| Embedding |-->| Recurrent LSTM |-->| DenseLayer |--> y
         +-----------+   +----------------+   +------------+

### Task 1: Add Batch Normalization

We now want to add new layers to the model, specifically batch normalization.

Batch normalization is a popular technique for speeding up convergence.
It is often used for image-processing setups, for example our other [hands-on lab on image
recognition](./Hands-On-Labs-Image-Recognition).
But could it work for recurrent models, too?

> Note: training with Batch Normalization is currently only supported on GPU.

So your task will be to insert batch-normalization layers before and after the recurrent LSTM layer.
If you have completed the [hands-on labs on image processing](https://github.com/Microsoft/CNTK/blob/v2.0.rc1/Tutorials/CNTK_201B_CIFAR-10_ImageHandsOn.ipynb),
you may remember that the [batch-normalization layer](https://www.cntk.ai/pythondocs/layerref.html#batchnormalization-layernormalization-stabilizer) has this form:
```
    BatchNormalization()
```
So please go ahead and modify the configuration and see what happens.

If everything went right, you will notice improved convergence speed (`loss` and `metric`)
compared to the previous configuration.


```python
# Your task: Add batch normalization
def create_model():
    with default_options(initial_state=0.1):
        return Sequential([
            Embedding(emb_dim),
            Recurrence(LSTM(hidden_dim), go_backwards=False),
            Dense(num_labels)
        ])

# Enable these when done:
#do_train()
#do_test()
```

### Task 2: Add a Lookahead 

Our recurrent model suffers from a structural deficit:
Since the recurrence runs from left to right, the decision for a slot label
has no information about upcoming words. The model is a bit lopsided.
Your task will be to modify the model such that
the input to the recurrence consists not only of the current word, but also of the next one
(lookahead).

Your solution should be in function-composition style.
Hence, you will need to write a Python function that does the following:

* takes no input arguments
* creates a placeholder (sequence) variable
* computes the "next value" in this sequence using the `sequence.future_value()` operation and
* concatenates the current and the next value into a vector of twice the embedding dimension using `splice()`

and then insert this function into `Sequential()`'s list right after the embedding layer.


```python
# Your task: Add lookahead
def create_model():
    with default_options(initial_state=0.1):
        return Sequential([
            Embedding(emb_dim),
            Recurrence(LSTM(hidden_dim), go_backwards=False),
            Dense(num_labels)
        ])
    
# Enable these when done:
#do_train()
#do_test()
```

### Task 3: Bidirectional Recurrent Model

Aha, knowledge of future words help. So instead of a one-word lookahead,
why not look ahead until all the way to the end of the sentence, through a backward recurrence?
Let us create a bidirectional model!

Your task is to implement a new layer that
performs both a forward and a backward recursion over the data, and
concatenates the output vectors.

Note, however, that this differs from the previous task in that
the bidirectional layer contains learnable model parameters.
In function-composition style,
the pattern to implement a layer with model parameters is to write a *factory function*
that creates a *function object*.

A function object, also known as [*functor*](https://en.wikipedia.org/wiki/Function_object), is an object that is both a function and an object.
Which means nothing else that it contains data yet still can be invoked as if it was a function.

For example, `Dense(outDim)` is a factory function that returns a function object that contains
a weight matrix `W`, a bias `b`, and another function to compute 
`input @ W + b.` (This is using 
[Python 3.5 notation for matrix multiplication](https://docs.python.org/3/whatsnew/3.5.html#whatsnew-pep-465).
In Numpy syntax it is `input.dot(W) + b`).
E.g. saying `Dense(1024)` will create this function object, which can then be used
like any other function, also immediately: `Dense(1024)(x)`. 

Let's look at an example for further clarity: Let us implement a new layer that combines
a linear layer with a subsequent batch normalization. 
To allow function composition, the layer needs to be realized as a factory function,
which could look like this:

```python
def DenseLayerWithBN(dim):
    F = Dense(dim)
    G = BatchNormalization()
    x = placeholder()
    apply_x = G(F(x))
    return apply_x
```

Invoking this factory function will create `F`, `G`, `x`, and `apply_x`. In this example, `F` and `G` are function objects themselves, and `apply_x` is the function to be applied to the data.
Thus, e.g. calling `DenseLayerWithBN(1024)` will
create an object containing a linear-layer function object called `F`, a batch-normalization function object `G`,
and `apply_x` which is the function that implements the actual operation of this layer
using `F` and `G`. It will then return `apply_x`. To the outside, `apply_x` looks and behaves
like a function. Under the hood, however, `apply_x` retains access to its specific instances of `F` and `G`.

Now back to our task at hand. You will now need to create a factory function,
very much like the example above.
You shall create a factory function
that creates two recurrent layer instances (one forward, one backward), and then defines an `apply_x` function
which applies both layer instances to the same `x` and concatenate the two results.

Alright, give it a try! To know how to realize a backward recursion in CNTK,
please take a hint from how the forward recursion is done.
Please also do the following:
* remove the one-word lookahead you added in the previous task, which we aim to replace; and
* make sure each LSTM is using `hidden_dim//2` outputs to keep the total number of model parameters limited.


```python
# Your task: Add bidirectional recurrence
def create_model():
    with default_options(initial_state=0.1):  
        return Sequential([
            Embedding(emb_dim),
            Recurrence(LSTM(hidden_dim), go_backwards=False),
            Dense(num_labels)
        ])

# Enable these when done:
#do_train()
#do_test()
```

Works like a charm! This model achieves 2.1%, a tiny bit better than the lookahead model above.
The bidirectional model has 40% less parameters than the lookahead one. However, if you go back and look closely
you may find that the lookahead one trained about 30% faster.
This is because the lookahead model has both less horizontal dependencies (one instead of two
recurrences) and larger matrix products, and can thus achieve higher parallelism.

### Solution 1: Adding Batch Normalization


```python
def create_model():
    with default_options(initial_state=0.1):
        return Sequential([
            Embedding(emb_dim),
            BatchNormalization(),
            Recurrence(LSTM(hidden_dim), go_backwards=False),
            BatchNormalization(),
            Dense(num_labels)
        ])

do_train()
do_test()
```

    Training 722379 parameters in 10 parameter tensors.
    Finished Epoch[1 of 16]: [Training] loss = 0.442525 * 18010, metric = 8.04% * 18010 18.515s (972.7 samples/s);
    Finished Epoch[2 of 16]: [Training] loss = 0.163139 * 18051, metric = 3.58% * 18051 3.352s (5385.1 samples/s);
    Finished Epoch[3 of 16]: [Training] loss = 0.117635 * 17941, metric = 2.55% * 17941 3.204s (5599.6 samples/s);
    Finished Epoch[4 of 16]: [Training] loss = 0.090393 * 18059, metric = 2.14% * 18059 3.409s (5297.4 samples/s);
    Finished Epoch[5 of 16]: [Training] loss = 0.048607 * 17957, metric = 1.25% * 17957 3.362s (5341.2 samples/s);
    Finished Epoch[6 of 16]: [Training] loss = 0.048540 * 18021, metric = 1.27% * 18021 3.280s (5494.2 samples/s);
    Finished Epoch[7 of 16]: [Training] loss = 0.044776 * 17980, metric = 1.14% * 17980 3.329s (5401.0 samples/s);
    Finished Epoch[8 of 16]: [Training] loss = 0.036818 * 18025, metric = 1.04% * 18025 3.379s (5334.4 samples/s);
    Finished Epoch[9 of 16]: [Training] loss = 0.025719 * 17956, metric = 0.80% * 17956 3.300s (5441.2 samples/s);
    Finished Epoch[10 of 16]: [Training] loss = 0.027097 * 18039, metric = 0.81% * 18039 3.410s (5290.0 samples/s);
    Finished Epoch[11 of 16]: [Training] loss = 0.027286 * 17966, metric = 0.76% * 17966 3.310s (5427.8 samples/s);
    Finished Epoch[12 of 16]: [Training] loss = 0.018633 * 18041, metric = 0.55% * 18041 3.236s (5575.1 samples/s);
    Finished Epoch[13 of 16]: [Training] loss = 0.020923 * 17984, metric = 0.72% * 17984 3.453s (5208.2 samples/s);
    Finished Epoch[14 of 16]: [Training] loss = 0.019829 * 17976, metric = 0.65% * 17976 3.289s (5465.5 samples/s);
    Finished Epoch[15 of 16]: [Training] loss = 0.019536 * 18030, metric = 0.56% * 18030 3.360s (5366.1 samples/s);
    Finished Epoch[16 of 16]: [Training] loss = 0.013894 * 18014, metric = 0.43% * 18014 3.359s (5362.9 samples/s);
    Finished Evaluation [1]: Minibatch[1-23]: metric = 2.05% * 10984;


### Solution 2: Add a Lookahead


```python
def OneWordLookahead():
    x = C.placeholder()
    apply_x = splice (x, sequence.future_value(x))
    return apply_x

def create_model():
    with default_options(initial_state=0.1):
        return Sequential([
            Embedding(emb_dim),
            OneWordLookahead(),
            Recurrence(LSTM(hidden_dim), go_backwards=False),
            Dense(num_labels)        
        ])

do_train()
do_test()
```

    Training 901479 parameters in 6 parameter tensors.
    Finished Epoch[1 of 16]: [Training] loss = 1.042147 * 18010, metric = 19.54% * 18010 3.392s (5309.6 samples/s);
    Finished Epoch[2 of 16]: [Training] loss = 0.367830 * 18051, metric = 8.32% * 18051 3.157s (5717.8 samples/s);
    Finished Epoch[3 of 16]: [Training] loss = 0.240927 * 17941, metric = 5.22% * 17941 3.199s (5608.3 samples/s);
    Finished Epoch[4 of 16]: [Training] loss = 0.162869 * 18059, metric = 3.72% * 18059 3.288s (5492.4 samples/s);
    Finished Epoch[5 of 16]: [Training] loss = 0.117314 * 17957, metric = 2.48% * 17957 3.180s (5646.9 samples/s);
    Finished Epoch[6 of 16]: [Training] loss = 0.104019 * 18021, metric = 2.21% * 18021 3.168s (5688.4 samples/s);
    Finished Epoch[7 of 16]: [Training] loss = 0.091837 * 17980, metric = 2.11% * 17980 3.157s (5695.3 samples/s);
    Finished Epoch[8 of 16]: [Training] loss = 0.085473 * 18025, metric = 1.90% * 18025 3.310s (5445.6 samples/s);
    Finished Epoch[9 of 16]: [Training] loss = 0.055418 * 17956, metric = 1.17% * 17956 3.246s (5531.7 samples/s);
    Finished Epoch[10 of 16]: [Training] loss = 0.056879 * 18039, metric = 1.23% * 18039 3.199s (5638.9 samples/s);
    Finished Epoch[11 of 16]: [Training] loss = 0.059522 * 17966, metric = 1.31% * 17966 3.206s (5603.9 samples/s);
    Finished Epoch[12 of 16]: [Training] loss = 0.038848 * 18041, metric = 0.84% * 18041 3.216s (5609.8 samples/s);
    Finished Epoch[13 of 16]: [Training] loss = 0.042280 * 17984, metric = 0.96% * 17984 3.286s (5472.9 samples/s);
    Finished Epoch[14 of 16]: [Training] loss = 0.046721 * 17976, metric = 1.02% * 17976 3.201s (5615.7 samples/s);
    Finished Epoch[15 of 16]: [Training] loss = 0.037370 * 18030, metric = 0.82% * 18030 3.322s (5427.5 samples/s);
    Finished Epoch[16 of 16]: [Training] loss = 0.034206 * 18014, metric = 0.77% * 18014 3.193s (5641.7 samples/s);
    Finished Evaluation [1]: Minibatch[1-23]: metric = 2.18% * 10984;


### Solution 3: Bidirectional Recurrent Model


```python
def BiRecurrence(fwd, bwd):
    F = Recurrence(fwd)
    G = Recurrence(bwd, go_backwards=True)
    x = C.placeholder()
    apply_x = splice (F(x), G(x))
    return apply_x 

def create_model():
    with default_options(initial_state=0.1):
        return Sequential([
            Embedding(emb_dim),
            BiRecurrence(LSTM(hidden_dim//2), LSTM(hidden_dim//2)),
            Dense(num_labels)
        ])

do_train()
do_test()
```

    Training 541479 parameters in 9 parameter tensors.
    Finished Epoch[1 of 16]: [Training] loss = 1.066098 * 18010, metric = 19.96% * 18010 3.879s (4642.9 samples/s);
    Finished Epoch[2 of 16]: [Training] loss = 0.398866 * 18051, metric = 8.70% * 18051 3.787s (4766.6 samples/s);
    Finished Epoch[3 of 16]: [Training] loss = 0.256761 * 17941, metric = 5.60% * 17941 3.692s (4859.4 samples/s);
    Finished Epoch[4 of 16]: [Training] loss = 0.179482 * 18059, metric = 4.22% * 18059 3.781s (4776.2 samples/s);
    Finished Epoch[5 of 16]: [Training] loss = 0.130943 * 17957, metric = 2.92% * 17957 3.776s (4755.6 samples/s);
    Finished Epoch[6 of 16]: [Training] loss = 0.115195 * 18021, metric = 2.54% * 18021 3.716s (4849.6 samples/s);
    Finished Epoch[7 of 16]: [Training] loss = 0.095113 * 17980, metric = 2.07% * 17980 3.665s (4905.9 samples/s);
    Finished Epoch[8 of 16]: [Training] loss = 0.094233 * 18025, metric = 2.08% * 18025 3.732s (4829.8 samples/s);
    Finished Epoch[9 of 16]: [Training] loss = 0.062660 * 17956, metric = 1.30% * 17956 3.753s (4784.4 samples/s);
    Finished Epoch[10 of 16]: [Training] loss = 0.063548 * 18039, metric = 1.40% * 18039 3.804s (4742.1 samples/s);
    Finished Epoch[11 of 16]: [Training] loss = 0.063781 * 17966, metric = 1.32% * 17966 3.617s (4967.1 samples/s);
    Finished Epoch[12 of 16]: [Training] loss = 0.046256 * 18041, metric = 1.06% * 18041 3.592s (5022.6 samples/s);
    Finished Epoch[13 of 16]: [Training] loss = 0.048336 * 17984, metric = 1.02% * 17984 3.918s (4590.1 samples/s);
    Finished Epoch[14 of 16]: [Training] loss = 0.054140 * 17976, metric = 1.19% * 17976 3.691s (4870.2 samples/s);
    Finished Epoch[15 of 16]: [Training] loss = 0.041836 * 18030, metric = 0.94% * 18030 3.815s (4726.1 samples/s);
    Finished Epoch[16 of 16]: [Training] loss = 0.044179 * 18014, metric = 0.98% * 18014 3.691s (4880.5 samples/s);
    Finished Evaluation [1]: Minibatch[1-23]: metric = 2.01% * 10984;



```python

```
