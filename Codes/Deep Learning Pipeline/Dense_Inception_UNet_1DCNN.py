# Import Necessary Libraries
import numpy as np
import tensorflow as tf


def Conv_Block(inputs, model_width, kernel_size, strides, multiplier, is_batchnorm=True, is_relu=True):
    # 1D Convolutional Block
    x = tf.keras.layers.Conv1D(model_width * multiplier, kernel_size, strides=strides, padding='same')(inputs)
    if is_batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    if is_relu:
        x = tf.keras.layers.Activation('relu')(x)

    return x


def trans_conv1D(inputs, model_width, kernel_size, strides, multiplier, is_batchnorm=True, is_relu=True):
    # 1D Transposed Convolutional Block, used instead of UpSampling
    x = tf.keras.layers.Conv1DTranspose(model_width * multiplier, kernel_size, strides=strides, padding='same')(inputs)  # Stride = 2, Kernel Size = 2
    if is_batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    if is_relu:
        x = tf.keras.layers.Activation('relu')(x)

    return x


def upConv_Block(inputs):
    # 1D UpSampling Block
    up = tf.keras.layers.UpSampling1D(size=2)(inputs)

    return up


def Concat_Block(input1, *argv):
    # Concatenation Block from the KERAS Library
    cat = input1
    for arg in range(0, len(argv)):
        cat = tf.keras.layers.concatenate([cat, argv[arg]], axis=-1)

    return cat


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
    resampler2 = trans_conv1D(conv1_2, 1, 2, 2, 1)
    resampler = tf.keras.layers.add([resampler1, resampler2])
    out = skip_connection*resampler

    return out


def Downsampling_Block(inputs, model_width, multiplier):
    # Downsampling Block
    pool = tf.keras.layers.MaxPooling1D(pool_size=2)(inputs)
    #
    conv1x1 = Conv_Block(inputs, model_width, 1, 1, multiplier, is_batchnorm=False, is_relu=False)
    conv3x3 = Conv_Block(conv1x1, model_width, 3, 2, multiplier)
    #
    conv1x1_dbl = Conv_Block(inputs, model_width, 1, 1, multiplier, is_batchnorm=False, is_relu=False)
    conv3x3_dbl = Conv_Block(conv1x1_dbl, model_width, 3, 1, multiplier, is_batchnorm=False, is_relu=False)
    conv3x3_dbl_2 = Conv_Block(conv3x3_dbl, model_width, 3, 2, multiplier)
    #
    branch_concat = tf.keras.layers.concatenate([pool, conv3x3, conv3x3_dbl_2], axis=-1)
    out = Conv_Block(branch_concat, model_width, 1, 1, multiplier)

    return out


def Upsampling_Block(inputs, model_width, multiplier):
    # Upsampling Block
    upconv1D = upConv_Block(inputs)
    #
    transconv1x1 = trans_conv1D(inputs, model_width, 1, 1, multiplier, is_batchnorm=False, is_relu=False)
    transconv3x3 = trans_conv1D(transconv1x1, model_width, 3, 2, multiplier)
    #
    transconv1x1_dbl = trans_conv1D(inputs, model_width, 1, 1, multiplier, is_batchnorm=False, is_relu=False)
    transconv3x3_dbl = trans_conv1D(transconv1x1_dbl, model_width, 3, 1, multiplier)
    transconv3x3_dbl_2 = trans_conv1D(transconv3x3_dbl, model_width, 3, 2, multiplier)
    #
    branch_concat = tf.keras.layers.concatenate([upconv1D, transconv3x3, transconv3x3_dbl_2], axis=-1)
    out = trans_conv1D(branch_concat, model_width, 1, 1, multiplier)

    return out

def Inception_Res_Block(inputs, model_width, multiplier):
    # Inception Residual Block
    conv1x1_BN = Conv_Block(inputs, model_width, 1, 1, multiplier)
    #
    conv1x1 = Conv_Block(inputs, model_width, 1, 1, multiplier, is_batchnorm=False, is_relu=False)
    conv3x3 = Conv_Block(conv1x1, model_width, 3, 1, multiplier)
    #
    conv1x1_dbl = Conv_Block(inputs, model_width, 1, 1, multiplier, is_batchnorm=False, is_relu=False)
    conv3x3_dbl = Conv_Block(conv1x1_dbl, model_width, 3, 1, multiplier)
    conv3x3_trpl = Conv_Block(conv3x3_dbl, model_width, 3, 1, multiplier)
    #
    branch_concat = tf.keras.layers.concatenate([conv1x1_BN, conv3x3, conv3x3_trpl], axis=-1)
    branch_conv = Conv_Block(branch_concat, model_width, 1, 1, multiplier)
    #
    out = tf.keras.layers.concatenate([inputs, branch_conv], axis=-1)

    return out


def Inception_Res_Block_2(inputs, model_width, multiplier):
    # Inception Residual Block
    conv1x1 = Conv_Block(inputs, model_width, 1, 1, multiplier)
    #
    pool = tf.keras.layers.MaxPooling1D(pool_size=1)(inputs)
    conv1x1_2 = Conv_Block(pool, model_width, 1, 1, multiplier, is_batchnorm=False, is_relu=False)
    #
    conv1x1_dbl = Conv_Block(inputs, model_width, 1, 1, multiplier, is_batchnorm=False, is_relu=False)
    conv3x3_dbl = Conv_Block(conv1x1_dbl, model_width, 3, 1, multiplier)
    conv3x3_trpl = Conv_Block(conv3x3_dbl, model_width, 3, 1, multiplier)
    #
    branch_concat = tf.keras.layers.concatenate([conv1x1, conv1x1_2, conv3x3_trpl], axis=-1)
    branch_conv = Conv_Block(branch_concat, model_width, 1, 1, multiplier)
    #
    out = tf.keras.layers.concatenate([inputs, branch_conv], axis=-1)

    return out


def Dense_Inception_Block(x, model_width, multiplier, num_dense_loop):
    for _ in range(0, num_dense_loop):
        IRU = Inception_Res_Block_2(x, model_width, multiplier)
        x = tf.keras.layers.concatenate([x, IRU], axis=-1)

    return x


class Dense_Inception_UNet:
    def __init__(self, length, model_depth, num_channel, model_width, kernel_size, problem_type='Regression',
                 output_nums=1, num_dense_loop=0, ds=0, ae=0, ag=0, feature_number=1024):
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
        # dense_loop: Number of Dense Block in the most bottom layers (1 and 3 are defaults for the BCDUNet's latent layer)
        # feature_number: Number of Features or Embeddings to be extracted from the AutoEncoder in the A_E Mode
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
        self.num_dense_loop = num_dense_loop
        self.feature_number = feature_number

    def Dense_Inception_UNet(self):
        """Variable Dense_Inception_UNet Model Design"""
        if self.length == 0 or self.model_depth == 0 or self.model_width == 0 or self.num_channel == 0 or self.kernel_size == 0:
            print("ERROR: Please Check the Values of the Input Parameters!")

        convs = {}
        levels = []

        # Encoding
        inputs = tf.keras.Input((self.length, self.num_channel))
        pool = inputs

        for i in range(1, (self.model_depth + 1)):
            if i == self.model_depth:
                conv = Dense_Inception_Block(pool, self.model_width, 2 ** (i - 1), self.num_dense_loop)
                pool = Downsampling_Block(conv, self.model_width, 2 ** (i - 1))
                convs["conv%s" % i] = conv
                continue
            conv = Inception_Res_Block(pool, self.model_width, 2 ** (i - 1))
            pool = Downsampling_Block(conv, self.model_width, 2 ** (i - 1))
            convs["conv%s" % i] = conv

        # Collect Latent Features or Embeddings from AutoEncoders
        if self.A_E == 1:
            # Collect Latent Features or Embeddings from AutoEncoders
            pool = Feature_Extraction_Block(pool, self.model_width, self.feature_number)
        conv = Dense_Inception_Block(pool, self.model_width, 2 ** self.model_depth, self.num_dense_loop)

        # Decoding
        deconv = conv
        convs_list = list(convs.values())

        for j in range(0, self.model_depth):
            layer_num = self.model_depth - j
            skip_connection = convs_list[layer_num - 1]
            if self.A_G == 1:
                skip_connection = Attention_Block(convs_list[layer_num - 1], deconv, self.model_width, 2 ** (layer_num - 1))
            if self.D_S == 1:
                level = tf.keras.layers.Conv1D(1, 1, name=f'level{layer_num}')(deconv)
                levels.append(level)
            deconv = Upsampling_Block(deconv, self.model_width, 2 ** (layer_num - 1))
            deconv = Concat_Block(deconv, skip_connection)
            if layer_num == self.model_depth:
                deconv = Dense_Inception_Block(deconv, self.model_width, 2 ** (layer_num - 1), self.num_dense_loop)
            else:
                deconv = Inception_Res_Block(deconv, self.model_width, 2 ** (layer_num - 1))

        deconv = Inception_Res_Block(deconv, self.model_width, 0.5)
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
    signal_length = 1024  # Length of each Segment
    model_depth = 5  # Number of Level in the CNN Model
    model_width = 64  # Width of the Initial Layer, subsequent layers start from here
    kernel_size = 3  # Size of the Kernels/Filter
    num_channel = 1  # Number of Channels in the Model
    D_S = 1  # Turn on Deep Supervision
    A_E = 0  # Turn on AutoEncoder Mode for Feature Extraction
    A_G = 0  # Turn on for Guided Attention
    num_dense_loop = 2
    problem_type = 'Regression'
    output_nums = 1  # Number of Class for Classification Problems, always '1' for Regression Problems
    feature_number = 1024  # Number of Features to be Extracted, only required if the AutoEncoder Mode is turned on
    model_name = 'Dense_Inception_UNet'  # Dense_Inception_UNet
    #
    Model = Dense_Inception_UNet(signal_length, model_depth, num_channel, model_width, kernel_size, problem_type=problem_type,
                                 output_nums=output_nums, num_dense_loop =num_dense_loop, ds=D_S, ae=A_E, ag=A_G).Dense_Inception_UNet()
    Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss=tf.keras.losses.MeanAbsoluteError(), metrics=tf.keras.metrics.MeanSquaredError())
    Model.summary()
