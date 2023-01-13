from flask import Flask, render_template, request
from translate import id_to_ch, ch_to_id, id_to_text
import tensorflow as tf
from OneStep import OneStep
from MyModel import MyModel
import os
app = Flask(__name__)

@app.route("/",methods=['GET', 'POST'])
def lyrics():
    if request.method == "GET":
        return render_template("lyrics.html")
    elif request.method == "POST":
        lyr = get_lyrics(1000).split("\n")
        return render_template('lyrics.html',lyrics=lyr)
    
def get_lyrics(chars):
    model = MyModel(
        vocab_size=97,
        embedding_dim=256,
        rnn_units=1024)
    model.load_weights('./ckpt_20')
    one_step = OneStep(model, id_to_ch, ch_to_id)
    states = None
    next_char = tf.constant(['['])
    result = [next_char]

    for n in range(chars):
        next_char, states = one_step.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    return result[0].numpy().decode('utf-8')

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


if __name__ == "__main__":
  app.run(debug=False, port=int(os.environ.get("PORT")))