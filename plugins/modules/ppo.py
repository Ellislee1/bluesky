from tensorflow.keras import layers, optimizers, models
import tensorflow.keras.backend as K


class PPO():
    def __init__(self, state_size, action_size, value_size, hidden_size=45, lr=1e-4, loss_clipping=0.2, entropy_loss=1e-4):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.value_size = value_size
        self.lr = lr
        self.loss_clipping = loss_clipping
        self.entropy_loss = entropy_loss

    def build_model(self):
        # Inputs
        _input = layers.Input(
            shape=(self.state_size,), name='input_states')

        _input_context = layers.Input(
            shape=(None, 6), name='context_input')
        empty = layers.Input(shape=(self.hidden_size,), name='empty')

        advantage = layers.Input(shape=(1,), name='A')
        old_prediction = layers.Input(
            shape=(self.action_size,), name='old_pred')

        # LSTM
        h1 = layers.LSTM(self.hidden_size, activation='tanh')(
            _input_context, initial_state=[empty, empty])

        # Combine inputs
        combined = layers.concatenate([_input, h1], axis=1)

        # Hidden Layers
        h2 = layers.Dense(256, activation='relu')(combined)
        h3 = layers.Dense(256, activation='relu')(h2)
        h4 = layers.Dense(256, activation='relu')(h3)

        # Output
        output = layers.Dense(self.action_size+1, activation=None)(h4)

        # Policy and value
        policy = layers.Lambda(
            lambda x: x[:, :self.action_size], output_shape=(self.action_size,))(output)
        value = layers.Lambda(
            lambda x: x[:, self.action_size:], output_shape=(self.value_size,))(output)

        # applied activation
        policy_out = layers.Activation(
            'softmax', name='policy_out')(policy)
        value_out = layers.Activation(
            'linear', name='value_out')(value)

        # optimiser
        opt = optimizers.Adam(lr=self.lr)

        model = models.Model(inputs=[
            _input, _input_context, empty, advantage, old_prediction], outputs=[policy_out, value_out])

        self.predictor = models.Model(
            inputs=[_input, _input_context, empty], outputs=[policy_out, value_out])

        model.compile(optimizer=opt, loss={'policy_out': self.ppo_loss(
            advantage=advantage,
            old_prediction=old_prediction), 'value_out': 'mse'})

        print(model.summary())

        return model

    def ppo_loss(self, advantage, old_prediction):
        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            r = prob/(old_prob + 1e-10)
            return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - self.loss_clipping, max_value=1 + self.loss_clipping) * advantage) + self.entropy_loss * -(prob * K.log(prob + 1e-10)))
        return loss
