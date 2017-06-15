from __future__ import print_function
import numpy as np
import collections
import os
from cntk import Trainer, Axis
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT
from cntk.learners import momentum_sgd, fsadagrad, momentum_as_time_constant_schedule, learning_rate_schedule, UnitType
from cntk import input, cross_entropy_with_softmax, classification_error, sequence, \
                element_select, alias, hardmax, placeholder_variable, combine, parameter, times, plus
from cntk.ops.functions import CloneMethod, load_model, Function
from cntk.initializer import glorot_uniform
from cntk.logging import log_number_of_parameters, ProgressPrinter
from cntk.logging.graph import plot
from cntk.layers import *
from cntk.layers.sequence import *
from cntk.layers.models.attention import *
from cntk.layers.typing import *

# variables you may want to change
## common values for prepare.py and main.py
vocabulary_file_path  = "./data/vocab.txt"
training_ctf_file_path = "./data/training.ctf"
validation_ctf_file_path = "./data/validation.ctf"
test_ctf_file_path = "./data/test.ctf"
tiny_ctf_file_path = "./data/tiny.ctf"
seq_start = "__{__" # has to be a palindrome
seq_end = "__}__" # has to be another palindrome
## values that are only needed in main.py


# configure CNTK to use CPU or GPU
vm_config="cpu" # default value
if 'TEST_DEVICE' in os.environ:
    vm_config = os.environ['TEST_DEVICE']
else:
    print("TEST_DEVICE environment variable was not found, using %s as a default value"%vm_config)

print("vm_config=%s"%vm_config)

# Select the right target device when this notebook is being tested:
if vm_config == "gpu":
    import cntk
    print("using GPU")
    cntk.device.set_default_device(cntk.device.gpu(0))
else:
    import cntk
    print("using CPU")
    cntk.device.try_set_default_device(cntk.device.cpu())

# Because of randomization in training we set a fixed random seed to ensure repeatable outputs
from _cntk_py import set_fixed_random_seed
set_fixed_random_seed(34)

# Helper function to load the model vocabulary file
def get_vocab(path):
    # get the vocab for printing output sequences in plaintext
    vocab = [w.strip() for w in open(path).readlines()]
    i2w = { i:w for i,w in enumerate(vocab) }
    w2i = { w:i for i,w in enumerate(vocab) }
    
    return (vocab, i2w, w2i)

# Read vocabulary data and generate their corresponding indices
vocab, i2w, w2i = get_vocab(vocabulary_file_path)

def create_reader(path, is_training):
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        features = StreamDef(field='S0', shape=input_vocab_dim, is_sparse=True),
        labels   = StreamDef(field='S1', shape=label_vocab_dim, is_sparse=True)
    )), randomize = is_training, max_sweeps = INFINITELY_REPEAT if is_training else 1)

input_vocab_dim = label_vocab_dim = len(vocab)

# Print vocab and the correspoding mapping to the phonemes
print("Vocabulary size is", len(vocab))
print("First 15 words are:")
print(vocab[:15])
print()

# Train data reader
train_reader = create_reader(training_ctf_file_path, True)

# Validation data reader
valid_reader = create_reader(validation_ctf_file_path, True)

# model hyperparameters
hidden_dim = 512
num_layers = 2
attention_dim = 128
attention_span = 20
attention_axis = -3
use_attention = True
use_embedding = True
embedding_dim = 200
vocab = ([w.strip() for w in open(vocabulary_file_path).readlines()]) # all lines of VOCAB_FILE in a list
length_increase = 1.5

# 
sentence_start = Constant(np.array([w==seq_start for w in vocab], dtype=np.float32))
sentence_end_index = vocab.index(seq_end)

# Source and target inputs to the model
inputAxis = Axis('inputAxis')
labelAxis = Axis('labelAxis')
InputSequence = SequenceOver[inputAxis]
LabelSequence = SequenceOver[labelAxis]

# define the model 
# create the s2s model
def create_model(): # :: (history*, input*) -> logP(w)*
    
    # Embedding: (input*) --> embedded_input*
    embed = Embedding(embedding_dim, name='embed') if use_embedding else identity
    
    # Encoder: (input*) --> (h0, c0)
    # Create multiple layers of LSTMs by passing the output of the i-th layer
    # to the (i+1)th layer as its input
    # Note: We go_backwards for the plain model, but forward for the attention model.
    with default_options(enable_self_stabilization=True, go_backwards=not use_attention):
        LastRecurrence = Fold if not use_attention else Recurrence
        encode = Sequential([
            embed,
            Stabilizer(),
            For(range(num_layers-1), lambda:
                Recurrence(LSTM(hidden_dim))),
            LastRecurrence(LSTM(hidden_dim), return_full_state=True),
            (Label('encoded_h'), Label('encoded_c')),
        ])

    # Decoder: (history*, input*) --> unnormalized_word_logp*
    # where history is one of these, delayed by 1 step and <s> prepended:
    #  - training: labels
    #  - testing:  its own output hardmax(z) (greedy decoder)
    with default_options(enable_self_stabilization=True):
        # sub-layers
        stab_in = Stabilizer()
        rec_blocks = [LSTM(hidden_dim) for i in range(num_layers)]
        stab_out = Stabilizer()
        proj_out = Dense(label_vocab_dim, name='out_proj')
        # attention model
        if use_attention: # maps a decoder hidden state and all the encoder states into an augmented state
            attention_model = AttentionModel(attention_dim, 
                                             attention_span, 
                                             attention_axis, 
                                             name='attention_model') # :: (h_enc*, h_dec) -> (h_dec augmented)
        # layer function
        @Function
        def decode(history, input):
            encoded_input = encode(input)
            r = history
            r = embed(r)
            r = stab_in(r)
            for i in range(num_layers):
                rec_block = rec_blocks[i]   # LSTM(hidden_dim)  # :: (dh, dc, x) -> (h, c)
                if use_attention:
                    if i == 0:
                        @Function
                        def lstm_with_attention(dh, dc, x):
                            h_att = attention_model(encoded_input.outputs[0], dh)
                            x = splice(x, h_att)
                            return rec_block(dh, dc, x)
                        r = Recurrence(lstm_with_attention)(r)
                    else:
                        r = Recurrence(rec_block)(r)
                else:
                    # unlike Recurrence(), the RecurrenceFrom() layer takes the initial hidden state as a data input
                    r = RecurrenceFrom(rec_block)(*(encoded_input.outputs + (r,))) # :: h0, c0, r -> h                    
            r = stab_out(r)
            r = proj_out(r)
            r = Label('out_proj_out')(r)
            return r

    return decode

def create_model_train(s2smodel):
    # model used in training (history is known from labels)
    # note: the labels must NOT contain the initial <s>
    @Function
    def model_train(input, labels): # (input*, labels*) --> (word_logp*)

        # The input to the decoder always starts with the special label sequence start token.
        # Then, use the previous value of the label sequence (for training) or the output (for execution).
        past_labels = Delay(initial_state=sentence_start)(labels)
        return s2smodel(past_labels, input)
    return model_train

def create_model_greedy(s2smodel):
    # model used in (greedy) decoding (history is decoder's own output)
    @Function
    @Signature(InputSequence[Tensor[input_vocab_dim]])
    def model_greedy(input): # (input*) --> (word_sequence*)

        # Decoding is an unfold() operation starting from sentence_start.
        # We must transform s2smodel (history*, input* -> word_logp*) into a generator (history* -> output*)
        # which holds 'input' in its closure.
        unfold = UnfoldFrom(lambda history: s2smodel(history, input) >> hardmax,
                            # stop once sentence_end_index was max-scoring output
                            until_predicate=lambda w: w[...,sentence_end_index],
                            length_increase=length_increase)
        
        return unfold(initial_state=sentence_start, dynamic_axes_like=input)
    return model_greedy

def create_criterion_function(model):
    @Function
    @Signature(input = InputSequence[Tensor[input_vocab_dim]], labels = LabelSequence[Tensor[label_vocab_dim]])
    def criterion(input, labels):
        # criterion function must drop the <s> from the labels
        postprocessed_labels = sequence.slice(labels, 1, 0) # <s> A B C </s> --> A B C </s>
        z = model(input, postprocessed_labels)
        ce   = cross_entropy_with_softmax(z, postprocessed_labels)
        errs = classification_error      (z, postprocessed_labels)
        return (ce, errs)

    return criterion

def train(train_reader, valid_reader, vocab, i2w, s2smodel, max_epochs, epoch_size):

    # create the training wrapper for the s2smodel, as well as the criterion function
    model_train = create_model_train(s2smodel)
    criterion = create_criterion_function(model_train)

    # also wire in a greedy decoder so that we can properly log progress on a validation example
    # This is not used for the actual training process.
    model_greedy = create_model_greedy(s2smodel)

    # Instantiate the trainer object to drive the model training
    minibatch_size = 72
    lr = 0.001 if use_attention else 0.005
    learner = fsadagrad(model_train.parameters,
                        lr       = learning_rate_schedule([lr]*2+[lr/2]*3+[lr/4], UnitType.sample, epoch_size),
                        momentum = momentum_as_time_constant_schedule(1100),
                        gradient_clipping_threshold_per_sample=2.3,
                        gradient_clipping_with_truncation=True)
    trainer = Trainer(None, criterion, learner)

    # Get minibatches of sequences to train with and perform model training
    total_samples = 0
    mbs = 0
    eval_freq = 100

    # print out some useful training information
    log_number_of_parameters(model_train) ; print()
    progress_printer = ProgressPrinter(freq=30, tag='Training')    

    # a hack to allow us to print sparse vectors
    sparse_to_dense = create_sparse_to_dense(input_vocab_dim)

    for epoch in range(max_epochs):
        while total_samples < (epoch+1) * epoch_size:
            # get next minibatch of training data
            mb_train = train_reader.next_minibatch(minibatch_size)
            
            # do the training
            trainer.train_minibatch({criterion.arguments[0]: mb_train[train_reader.streams.features], 
                                     criterion.arguments[1]: mb_train[train_reader.streams.labels]})

            progress_printer.update_with_trainer(trainer, with_metric=True) # log progress

            # every N MBs evaluate on a test sequence to visually show how we're doing
            if mbs % eval_freq == 0:
                mb_valid = valid_reader.next_minibatch(1)

                # run an eval on the decoder output model (i.e. don't use the groundtruth)
                e = model_greedy(mb_valid[valid_reader.streams.features])
                print(format_sequences(sparse_to_dense(mb_valid[valid_reader.streams.features]), i2w))
                print("->")
                print(format_sequences(e, i2w))

                # visualizing attention window
                if use_attention:
                    debug_attention(model_greedy, mb_valid[valid_reader.streams.features])

            total_samples += mb_train[train_reader.streams.labels].num_samples
            mbs += 1

        # log a summary of the stats for the epoch
        progress_printer.epoch_summary(with_metric=True)

    # done: save the final model
    model_path = "model_%d.cmf" % epoch
    print("Saving final model to '%s'" % model_path)
    s2smodel.save(model_path)
    print("%d epochs complete." % max_epochs)

# dummy for printing the input sequence below. Currently needed because input is sparse.
def create_sparse_to_dense(input_vocab_dim):
    I = Constant(np.eye(input_vocab_dim))
    @Function
    @Signature(InputSequence[SparseTensor[input_vocab_dim]])
    def no_op(input):
        return times(input, I)
    return no_op

# Given a vocab and tensor, print the output
def format_sequences(sequences, i2w):
    return [" ".join([i2w[np.argmax(w)] for w in s]) for s in sequences]

# to help debug the attention window
def debug_attention(model, input):
    q = combine([model, model.attention_model.attention_weights])
    #words, p = q(input) # Python 3
    words_p = q(input)
    words = words_p[0]
    p     = words_p[1]
    seq_len = words[0].shape[attention_axis-1]
    span = 7 #attention_span  #7 # test sentence is 7 tokens long
    p_sq = np.squeeze(p[0][:seq_len,:span,0,:]) # (batch, len, attention_span, 1, vector_dim)
    opts = np.get_printoptions()
    np.set_printoptions(precision=5)
    print(p_sq)
    np.set_printoptions(**opts)

model = create_model()
train(train_reader, valid_reader, vocab, i2w, model, max_epochs=1, epoch_size=25000)

# Uncomment the line below to train the model for a full epoch
#train(train_reader, valid_reader, vocab, i2w, model, max_epochs=1, epoch_size=908241)

# load the model for epoch 0
model_path = "model_0.cmf"
model = Function.load(model_path)

# create a reader pointing at our testing data
test_reader = create_reader(test_ctf_file_path, False)

# This decodes the test set and counts the string error rate.
def evaluate_decoding(reader, s2smodel, i2w):
    
    model_decoding = create_model_greedy(s2smodel) # wrap the greedy decoder around the model

    progress_printer = ProgressPrinter(tag='Evaluation')

    sparse_to_dense = create_sparse_to_dense(input_vocab_dim)

    minibatch_size = 512
    num_total = 0
    num_wrong = 0
    while True:
        mb = reader.next_minibatch(minibatch_size)
        if not mb: # finish when end of test set reached
            break
        e = model_decoding(mb[reader.streams.features])
        outputs = format_sequences(e, i2w)
        labels  = format_sequences(sparse_to_dense(mb[reader.streams.labels]), i2w)
        # prepend sentence start for comparison
        outputs = [seq_start + " " + output for output in outputs]

        num_total += len(outputs)
        num_wrong += sum([label != output for output, label in zip(outputs, labels)])
        
    rate = num_wrong / num_total
    print("string error rate of {:.1f}% in {} samples".format(100 * rate, num_total))
    return rate

# print the string error rate
evaluate_decoding(test_reader, model, i2w)

# This decodes the test set and counts the string error rate.
def evaluate_decoding(reader, s2smodel, i2w):
    
    model_decoding = create_model_greedy(s2smodel) # wrap the greedy decoder around the model

    progress_printer = ProgressPrinter(tag='Evaluation')

    sparse_to_dense = create_sparse_to_dense(input_vocab_dim)

    minibatch_size = 512
    num_total = 0
    num_wrong = 0
    while True:
        mb = reader.next_minibatch(minibatch_size)
        if not mb: # finish when end of test set reached
            break
        e = model_decoding(mb[reader.streams.features])
        outputs = format_sequences(e, i2w)
        labels  = format_sequences(sparse_to_dense(mb[reader.streams.labels]), i2w)
        # prepend sentence start for comparison
        outputs = ["<s> " + output for output in outputs]
        
        for s in range(len(labels)):
            for w in range(len(labels[s])):
                num_total += 1
                if w < len(outputs[s]): # in case the prediction is longer than the label
                    if outputs[s][w] != labels[s][w]:
                        num_wrong += 1
                
    rate = num_wrong / num_total
    print("{:.1f}".format(100 * rate))
    return rate

# print the phoneme error rate
test_reader = create_reader(test_ctf_file_path, False)
evaluate_decoding(test_reader, model, i2w)

