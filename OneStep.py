import tensorflow as tf

class OneStep(tf.keras.Model):
    def __init__(self, model, id_to_ch, ch_to_id, temperature=1.3):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.id_to_ch = id_to_ch
        self.ch_to_id = ch_to_id

    def generate_one_step(self, inputs, states=None):
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ch_to_id(input_chars).to_tensor()

        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                              return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature

        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        predicted_chars = self.id_to_ch(predicted_ids)

        return predicted_chars, states