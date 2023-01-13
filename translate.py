import tensorflow as tf

r = open('./chars.txt', 'r')
chars = r.read()

ch_to_id = tf.keras.layers.StringLookup(vocabulary=list(chars), mask_token=None)
id_to_ch = tf.keras.layers.StringLookup(vocabulary=list(chars), invert=True, mask_token=None)
def id_to_text(ids):
    return tf.strings.reduce_join(id_to_ch(ids), axis=-1)

r.close()