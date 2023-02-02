# imports
import pandas as pd
import os
from keras.preprocessing.text import Tokenizer
import tensorflow as tf

from model import Translator


# load datasets
train_raw_path = os.path.join( 'data', "train_raw")
train_raw = tf.data.Dataset.load(train_raw_path)

val_raw_path = os.path.join('data', "val_raw")
val_raw = tf.data.Dataset.load(val_raw_path)

# define constants
max_vocab_size = 5000
UNITS = 256


def tf_lower_and_split_punct(text):
  # Split accented characters.
  text = tf.strings.lower(text)
  # Keep space, a to z, and select punctuation.
  text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
  # Add spaces around punctuation.
  text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
  # Strip whitespace.
  text = tf.strings.strip(text)


# tokenization
context_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size,
    ragged=True)

context_text_processor.adapt(train_raw.map(lambda context, target: context))

target_text_processor = tf.keras.layers.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size,
    ragged=True)

target_text_processor.adapt(train_raw.map(lambda context, target: target))


def process_text(context, target):
  context = context_text_processor(context).to_tensor()
  target = target_text_processor(target)
  targ_in = target[:,:-1].to_tensor()
  targ_out = target[:,1:].to_tensor()
  return (context, targ_in), targ_out

train_ds = train_raw.map(process_text, tf.data.AUTOTUNE)
val_ds = val_raw.map(process_text, tf.data.AUTOTUNE)

# create model
model = Translator(UNITS, context_text_processor, target_text_processor)

model.compile(optimizer='adam')



# train model
history = model.fit(
    train_ds.repeat(), 
    epochs=100,
    steps_per_epoch = 100,
    validation_data=val_ds,
    validation_steps = 20,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3)])

# save model to file
class Export(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
  def translate(self, inputs):
    return self.model.translate(inputs)

export = Export(model)

tf.saved_model.save(export, 'translator', signatures={'serving_default': export.translate})