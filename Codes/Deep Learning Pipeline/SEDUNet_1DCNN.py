# Import Necessary Libraries
import numpy as np
import tensorflow as tf


def Conv_Block(inputs, model_width, kernel, multiplier):
    # 1D Convolutional Block
    x = tf.keras.layers.Conv1D(model_width * multiplier, kernel, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def trans_conv1D(inputs, model_width, multiplier):
    # 1D Transposed Convolutional Block, used instead of UpSampling
    x = tf.keras.layers.Conv1DTranspose(model_width * multiplier, 2, strides=2, padding='same')(inputs)  # Stride = 2, Kernel Size = 2
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def Concat_Block(input1, *argv):
    # Concatenation Block from the KERAS Library
    cat = input1
    for arg in range(0, len(argv)):
        cat = tf.keras.layers.concatenate([cat, argv[arg]], axis=-1)

    return cat


def upConv_Block(inputs):
    # 1D UpSampling Block
    up = tf.keras.layers.UpSampling1D(size=2)(inputs)

    return up


def Feature_Extraction_Block(inputs, model_width, feature_number):
    # Feature Extraction Block for the AutoEncoder Mode
    shape = inputs.shape
    latent = tf.keras.layers.Flatten()(inputs)
    latent = tf.keras.layers.Dense(feature_number, name='features')(latent)
    latent = tf.keras.layers.Dense(model_width * shape[1])(latent)
    latent = tf.keras.layers.Reshape((shape[1], model_width))(latent)

    return latent


def Attention_Block(skip_connection, gating_signal, num_filters, multiplier):
    # Attention Block
    conv1x1_1 = tf.keras.layers.Conv1D(num_filters*multiplier, 1, strides=2)(skip_connection)
    conv1x1_1 = tf.keras.layers.BatchNormalization()(conv1x1_1)
    conv1x1_2 = tf.keras.layers.Conv1D(num_filters*multiplier, 1, strides=1)(gating_signal)
    conv1x1_2 = tf.keras.layers.BatchNormalization()(conv1x1_2)
    conv1_2 = tf.keras.layers.add([conv1x1_1, conv1x1_2])
    conv1_2 = tf.keras.layers.Activation('relu')(conv1_2)
    conv1_2 = tf.keras.layers.Conv1D(1, 1, strides=1)(conv1_2)
    conv1_2 = tf.keras.layers.BatchNormalization()(conv1_2)
    conv1_2 = tf.keras.layers.Activation('sigmoid')(conv1_2)
    resampler1 = upConv_Block(conv1_2)
    resampler2 = trans_conv1D(conv1_2, 1, 1)
    resampler = tf.keras.layers.add([resampler1, resampler2])
    out = skip_connection * resampler

    return out


def dense_block(x, num_filters, kernel_size, multiplier, num_layers):
    for _ in range(num_layers):
        cb = Conv_Block(x, num_filters, kernel_size, multiplier)
        cb = Conv_Block(cb, num_filters, kernel_size, multiplier)
        x = tf.keras.layers.concatenate([x, cb], axis=-1)

    return x


def SqueezeExcite(x, ratio=16):
    nb_chan = tf.keras.backend.int_shape(x)[-1]
    y = tf.keras.layers.GlobalAveragePooling1D()(x)
    y = tf.keras.layers.Dense(nb_chan // ratio, activation='relu')(y)
    y = tf.keras.layers.Dense(nb_chan, activation='sigmoid')(y)
    y = tf.keras.layers.Multiply()([x, y])

    return y


class SEDUNet:
    def __init__(self, length, model_depth, num_channel, model_width, kernel_size, problem_type='Regression',
                 output_nums=1, ds=1, ae=0, ag=0, lstm=0, dense_loop=1, se_ratio=16, feature_number=1024, is_transconv=True):
        # length: Input Signal Length
        # model_depth: Depth of the Model
        # model_width: Width of the Input Layer of the Model
        # num_channel: Number of Channels allowed by the Model
        # kernel_size: Kernel or Filter Size of the Convolutional Layers
        # problem_type: Classification (Binary or Multiclass) or Regression
        # output_nums: Output Classes (Classification Mode) or Features (Regression Mode)
        # ds: Checks where Deep Supervision is active or not, either 0 or 1 [Default value set as 0]
        # ae: Enables or diables the AutoEncoder Mode, either 0 or 1 [Default value set as 0]
        # ag: Checks where Attention Guided is active or not, either 0 or 1 [Default value set as 0]
        # lstm: Checks where Bidirectional LSTM is active or not, either 0 or 1 [Default value set as 0]
        # dense_loop: Number of Dense Block in the most bottom layers (1 and 3 are defaults for the BCDUNet's latent layer)
        # se_ratio: Squeeze and Excite Ratio
        # feature_number: Number of Features or Embeddings to be extracted from the AutoEncoder in the A_E Mode
        # is_transconv: (TRUE - Transposed Convolution, FALSE - UpSampling) in the Encoder Layer
        self.length = length
        self.model_depth = model_depth
        self.num_channel = num_channel
        self.model_width = model_width
        self.kernel_size = kernel_size
        self.problem_type = problem_type
        self.output_nums = output_nums
        self.D_S = ds
        self.A_E = ae
        self.A_G = ag
        self.LSTM = lstm
        self.dense_loop = dense_loop
        self.se_ratio = se_ratio
        self.feature_number = feature_number
        self.is_transconv = is_transconv


    def SEDUNet(self):
        # Variable 1DBCDUNet Model Design
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            raise ValueError("Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            conv = Conv_Block(pool, self.model_width, self.kernel_size, 2 ** (i - 1))
            conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** (i - 1))
            pool = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
            convs["conv%s" % i] = conv

        conv = dense_block(pool, self.model_width, self.kernel_size, 2 ** self.model_depth, self.dense_loop - 1)
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            conv = Feature_Extraction_Block(conv, self.model_width, self.feature_number)
        conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)
        conv = Conv_Block(conv, self.model_width, self.kernel_size, 2 ** self.model_depth)

        # Decoding
        deconv = conv
        convs_list = list(convs.values())

        for j in range(0, self.model_depth):
            skip_connection = convs_list[self.model_depth - j - 1]
            if self.A_G == 1:
                skip_connection = Attention_Block(convs_list[self.model_depth - j - 1], deconv, self.model_width, 2 ** (self.model_depth - j - 1))
            if self.D_S == 1:
                # For Deep Supervision
                level = tf.keras.layers.Conv1D(1, 1, name=f'level{self.model_depth - j}')(deconv)
                levels.append(level)
            if self.is_transconv:
                deconv = trans_conv1D(deconv, self.model_width, 2 ** (self.model_depth - j - 1))
            elif not self.is_transconv:
                deconv = upConv_Block(deconv)
            deconv = SqueezeExcite(deconv, ratio=self.se_ratio)
            deconv = tf.keras.layers.BatchNormalization()(deconv)
            deconv = tf.keras.layers.Activation('relu')(deconv)
            if self.LSTM == 1:
                x1 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** (self.model_depth - j - 1)), np.int32(self.model_width * (2 ** (self.model_depth - j - 1)))))(skip_connection)
                x2 = tf.keras.layers.Reshape(target_shape=(1, np.int32(self.length / 2 ** (self.model_depth - j - 1)), np.int32(self.model_width * (2 ** (self.model_depth - j - 1)))))(deconv)
                merge = tf.keras.layers.concatenate([x1, x2], axis=-1)
                deconv = tf.keras.layers.ConvLSTM1D(filters=np.int32(self.model_width * (2 ** (self.model_depth - j - 2))), kernel_size=3, padding='same', return_sequences=False, go_backwards=True, kernel_initializer='he_normal')(merge)
            deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1))
            deconv = SqueezeExcite(deconv, ratio=self.se_ratio)
            deconv = Conv_Block(deconv, self.model_width, self.kernel_size, 2 ** (self.model_depth - j - 1))

        # Output
        outputs = []
        if self.problem_type == 'Classification':
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='softmax', name="out")(deconv)
        elif self.problem_type == 'Regression':
            outputs = tf.keras.layers.Conv1D(self.output_nums, 1, activation='linear', name="out")(deconv)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        if self.D_S == 1:
            levels.append(outputs)
            levels.reverse()
            model = tf.keras.Model(inputs=[inputs], outputs=levels)

        return model


if __name__ == '__main__':
    # Configurations
    length = 1024  # Length of each Segment
    model_name = 'SEDUNet'  # UNet or UNetPP
    model_depth = 5  # Number of Level in the CNN Model
    model_width = 64  # Width of the Initial Layer, subsequent layers start from here
    kernel_size = 3  # Size of the Kernels/Filter
    num_channel = 1  # Number of Channels in the Model
    D_S = 1  # Turn on Deep Supervision
    A_E = 0  # Turn on AutoEncoder Mode for Feature Extraction
    A_G = 1  # Turn on for Guided Attention
    LSTM = 1 # Turn on for LSTM
    num_dense_loop = 3 # Number of Dense Blocks in the BottleNeck Layer
    se_ratio = 16 # Branches of the Squeeze and Excite Block
    problem_type = 'Regression'
    output_nums = 1  # Number of Class for Classification Problems, always '1' for Regression Problems
    is_transconv = True # True: Transposed Convolution, False: UpSampling
    '''Only required if the AutoEncoder Mode is turned on'''
    feature_number = 1024  # Number of Features to be Extracted
    #
    Model = SEDUNet(length, model_depth, num_channel, model_width, kernel_size, problem_type=problem_type, output_nums=output_nums,
                    ds=D_S, ae=A_E, ag=A_G, lstm=LSTM, dense_loop=num_dense_loop, se_ratio=se_ratio, is_transconv=is_transconv).SEDUNet()
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss=tf.keras.losses.MeanAbsoluteError(), metrics=tf.keras.metrics.MeanSquaredError())
    Model.summary()
