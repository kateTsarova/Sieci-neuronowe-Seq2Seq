import io
import re

import tensorflow as tf
import tensorflow_addons as tfa

import os
import time

from preprocessing import NMTDataset

import unicodedata
from sklearn.model_selection import train_test_split

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.debugging.set_log_device_placement(True)

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
#tf.debugging.set_log_device_placement(True)
# Create some tensors
'''
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)
'''

BUFFER_SIZE = 32000
BATCH_SIZE = 16
# Let's limit the #training examples for faster training
num_examples = 30000

dataset_creator = NMTDataset()
train_dataset, val_dataset, inp_tok, targ_tok = dataset_creator.call(num_examples, BUFFER_SIZE, BATCH_SIZE, "code\\*.txt")
example_input_batch, example_target_batch = next(iter(train_dataset))
print(example_input_batch.shape, example_target_batch.shape)

vocab_inp_size = len(inp_tok.word_index) + 1
vocab_targ_size = len(targ_tok.word_index) + 1
max_length_input = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]

embedding_dim = 256
units = 512  # 1024
steps_per_epoch = num_examples // BATCH_SIZE

print("max_length_spanish, max_length_english, vocab_size_spanish, vocab_size_english")
print(max_length_input, max_length_output, vocab_inp_size, vocab_targ_size)


#####

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        ##________ LSTM layer in Encoder ------- ##
        self.lstm_layer = tf.keras.layers.LSTM(self.enc_units,
                                               return_sequences=True,
                                               return_state=True,
                                               recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, h, c = self.lstm_layer(x, initial_state=hidden)
        return output, h, c

    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]


## Test Encoder Stack

# vocab_inp_size = 16
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_h, sample_c = encoder(example_input_batch, sample_hidden)
print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print('Encoder h vecotr shape: (batch size, units) {}'.format(sample_h.shape))
print('Encoder c vector shape: (batch size, units) {}'.format(sample_c.shape))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, attention_type='luong'):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.attention_type = attention_type

        # Embedding Layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # Final Dense layer on which softmax will be applied
        self.fc = tf.keras.layers.Dense(vocab_size)

        # Define the fundamental cell for decoder recurrent structure
        self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)

        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(self.dec_units,
                                                                  None, self.batch_sz * [max_length_input],
                                                                  self.attention_type)

        # Wrap attention mechanism with the fundamental rnn cell of decoder
        self.rnn_cell = self.build_rnn_cell(batch_sz)

        # Define the decoder with respect to fundamental rnn cell
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)

    def build_rnn_cell(self, batch_sz):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell,
                                                self.attention_mechanism, attention_layer_size=self.dec_units)
        return rnn_cell

    def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type='luong'):
        # ------------- #
        # typ: Which sort of attention (Bahdanau, Luong)
        # dec_units: final dimension of attention outputs
        # memory: encoder hidden states of shape (batch_size, max_length_input, enc_units)
        # memory_sequence_length: 1d array of shape (batch_size) with every element set to max_length_input (for masking purpose)

        if (attention_type == 'bahdanau'):
            return tfa.seq2seq.BahdanauAttention(units=dec_units, memory=memory,
                                                 memory_sequence_length=memory_sequence_length)
        else:
            return tfa.seq2seq.LuongAttention(units=dec_units, memory=memory,
                                              memory_sequence_length=memory_sequence_length)

    def build_initial_state(self, batch_sz, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_sz, dtype=Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state

    def call(self, inputs, initial_state):
        x = self.embedding(inputs)
        outputs, _, _ = self.decoder(x, initial_state=initial_state,
                                     sequence_length=self.batch_sz * [max_length_output - 1])
        return outputs


# Test decoder stack

decoder = Decoder(vocab_targ_size, embedding_dim, units, BATCH_SIZE, 'luong')
sample_x = tf.random.uniform((BATCH_SIZE, max_length_output))
decoder.attention_mechanism.setup_memory(sample_output)
initial_state = decoder.build_initial_state(BATCH_SIZE, [sample_h, sample_c], tf.float32)

sample_decoder_outputs = decoder(sample_x, initial_state)

print("Decoder Outputs Shape: ", sample_decoder_outputs.rnn_output.shape)

optimizer = tf.keras.optimizers.Adam()


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))







def evaluate_code(code):
    # sentence = dataset_creator.call_predict(sentence)

    inputs = [inp_tok.word_index[i] for i in code.split(' ') if i != '']
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_input, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    inference_batch_size = inputs.shape[0]
    result = ''

    enc_start_state = [tf.zeros((inference_batch_size, units)), tf.zeros((inference_batch_size, units))]
    enc_out, enc_h, enc_c = encoder(inputs, enc_start_state)

    dec_h = enc_h
    dec_c = enc_c

    start_tokens = tf.fill([inference_batch_size], targ_tok.word_index['<sof>'])
    end_token = targ_tok.word_index['<eof>']

    greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

    # # Instantiate BasicDecoder object
    decoder_instance = tfa.seq2seq.BasicDecoder(cell=decoder.rnn_cell, sampler=greedy_sampler, output_layer=decoder.fc, maximum_iterations=1000)
    # # Setup Memory in decoder stack
    decoder.attention_mechanism.setup_memory(enc_out)

    # # set decoder_initial_state
    decoder_initial_state = decoder.build_initial_state(inference_batch_size, [enc_h, enc_c], tf.float32)

    # ### Since the BasicDecoder wraps around Decoder's rnn cell only, you have to ensure that the inputs to BasicDecoder
    ## decoding step is output of embedding layer. tfa.seq2seq.GreedyEmbeddingSampler() takes care of this.
    # ### You only need to get the weights of embedding layer, which can be done by decoder.embedding.variables[0] and pass this callabble to BasicDecoder's call() function

    decoder_embedding_matrix = decoder.embedding.variables[0]

    # print('before')
    # nie generuje eof
    # sprawdzic output
    outputs, _, _ = decoder_instance(decoder_embedding_matrix, start_tokens=start_tokens, end_token=end_token, initial_state=decoder_initial_state)
    # outputs, _, _ = decoder(decoder_embedding_matrix, initial_state=decoder_initial_state)
    # print('after')
    return outputs.sample_id.numpy()


def translate_one_line(code):
    result = evaluate_code(code)
    print(result)
    result = targ_tok.sequences_to_texts(result)
    print('Input: for(i=0;i<n;i++);')
    print('Output: {}'.format(result))

def translate(path):
    code = dataset_creator.preprocess_code(path)
    for line in code:
        result = evaluate_code(line)
        # print(result)
        result = targ_tok.sequences_to_texts(result)
        print('Output: {}'.format(result))


# test = ['0 ~', '#include', '<stdio.h>', '1 ~', '#include', '<stdlib.h>', '2 ~', 'void', '1@', '(', ')', ')', '{', '3 ~', '}', '4 ~', 'int', 'main', '(', ')', '{', '5 ~', 'long', 'int', '2@', ',', '3@', ',', '4@', ',', '5@', ';', '6 ~', 'int', '6@', ',', '7@', ',', '8@', '=', '_<number>_# ', ',', '9@', '=', '_<number>_# ', ';', '7 ~', 'scanf', '(', '_<string>_', ',', '&', '2@', ',', '&', '3@', ',', '&', '4@', ',', '&', '5@', ')', ';', '8 ~', 'return', '_<number>_# ', ';', '9 ~', '}']

translate("test/test.txt")

# ['~ #include <stdio.h> ~ #include <stdlib.h> ~ int main ( ) { ~ int 1@ , 2@ , 3@ , 4@ , 5@ , 6@ , 7@ ; ~ scanf ( _<string>_ , & 1@ ) ; ~ int 8@ [ 1@ ] ; ~ for ( 3@ = _<number>_# ; 3@ < 1@ ; 3@ ++ ) ~ { ~ scanf ( _<string>_ , & 8@ [ 3@ ] ) ; ~ } ~ scanf ( _<string>_ , & 2@ ) ; ~ for ( 3@ = _<number>_# ; 3@ < 2@ ; 3@ ++ ) ~ { ~ scanf ( _<string>_ , & 8@ [ 3@ ] ) ; ~ } ~ for ( 3@ = _<number>_# ; 3@ < 1@ ; 3@ ++ ) ~ { ~ for ( 4@ = 3@ + _<number>_# ; 4@ < 1@ ; 4@ ++ ) ~ { ~ if ( 8@ [ 3@ ] > 8@ [ 4@ ] ) ~ { ~ 5@ = 8@ [ 3@ ] ; ~ 8@ [ 3@ ] = 8@ [ 4@ ] ; ~ 8@ [ 4@ ] = 5@ ; ~ } ~ } ~ } ~ for ( 3@ = _<number>_# ; 3@ < 1@ ; 3@ ++ ) ~ { ~ printf ( _<string>_ , 8@ [ 3@ ] ) ; ~ } ~ return _<number>_# ; ~ } <eof>']
