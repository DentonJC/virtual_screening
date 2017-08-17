import keras
from keras.engine import InputSpec
from keras.layers import Wrapper


class residual_block(Wrapper):
    def build(self, input_shape):
        output_shape = input_shape
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        self.input_spec = [InputSpec(shape=input_shape)]
        super(residual_block, self).build()


    def call(self, x, mask=None):
        layer_output = self.layer.call(x, mask)
        output = keras.layers.Add()([x, layer_output])
        return output
