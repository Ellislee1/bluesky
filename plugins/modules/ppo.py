from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow as tf

LEARNING_RATE = 1e-4
HIDDEN_SIZE = 50
CLIPPING = 0.2
LOSS = 1e-5


# PPO loss function
def PPO_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        r = prob/(old_prob + 1e-10)

        fcn = -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - CLIPPING,
                                                      max_value=1 + CLIPPING) * advantage) + LOSS * -(prob * K.log(prob + 1e-10)))

        return fcn

    return loss


class PPO:
    def __init__(self, statesize, num_intruders, actionsize, valuesize, checkpoint):
        self.statesize = statesize
        self.num_intruders = num_intruders
        self.actionsize = actionsize
        self.valuesize = valuesize
        self.checkpoint_path = "models/training/"+checkpoint

        self.model = self.__build_linear__()

    def __build_linear__(self):
        # Input of the aircraft of focus
        _input = keras.layers.Input(
            shape=(self.statesize,), name='input_state')

        # This is the input for the n_closest aircraft
        _input_context = keras.layers.Input(
            shape=(self.num_intruders, 10), name='input_context')

        # Empty layer
        empty = keras.layers.Input(shape=(HIDDEN_SIZE,), name='empty')

        # Input for advantages
        advantage = keras.layers.Input(shape=(1,), name="advantage")

        # Input old prediction
        old_prediction = keras.layers.Input(
            shape=(self.actionsize,), name='old_predictions')

        # Flatten the context layer (As context is passed as an n*m tensor)
        flatten_context = keras.layers.Flatten()(_input_context)

        # Hidden Layers

        # 1st hidden applies to the context only
        h1 = keras.layers.Dense(
            HIDDEN_SIZE, activation='relu')(flatten_context)

        # Combine the input and the context
        combine = keras.layers.concatenate([_input, h1], axis=1)

        # Hidden layers 2 & 3 apply to all inputs
        h2 = keras.layers.Dense(512, activation='relu')(combine)
        h3 = keras.layers.Dense(512, activation='relu')(h2)

        # Output layer
        out = keras.layers.Dense(self.actionsize+1, activation=None)(h3)

        # Policy and value layer processing
        policy = keras.layers.Lambda(
            lambda x: x[:, :self.actionsize], output_shape=(self.actionsize,))(out)
        value = keras.layers.Lambda(
            lambda x: x[:, self.actionsize:], output_shape=(self.valuesize,))(out)

        # Policy and value outputs
        policy_out = keras.layers.Activation(
            'softmax', name='policy_out')(policy)
        value_out = keras.layers.Activation(
            'linear', name='value_out')(value)

        # Optimizer
        optimizer = keras.optimizers.Adam(lr=LEARNING_RATE)

        # Produce the model
        model = keras.models.Model(inputs=[
                                   _input, _input_context, empty, advantage, old_prediction], outputs=[policy_out, value_out])

        self.estimator = keras.models.Model(
            inputs=[_input, _input_context, empty], outputs=[policy_out, value_out])

        # Compile the model

        model.compile(optimizer=optimizer, loss={'policy_out': PPO_loss(
            advantage=advantage, old_prediction=old_prediction), 'value_out': 'mse'})

        print(model.summary())
        return model
