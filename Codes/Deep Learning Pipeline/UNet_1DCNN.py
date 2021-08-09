# Import Necessary Libraries
from keras.models import Model
from keras.layers import Input, Reshape, Flatten, Dense, Add, concatenate, BatchNormalization, Activation
from keras.layers import Conv1D, UpSampling1D, MaxPooling1D, Conv1DTranspose


def Conv_Block(inputs, model_width, kernel, multiplier):
    # 1D Convolutional Block
    conv = Conv1D(model_width * multiplier, kernel, padding='same')(inputs)
    batch_norm = BatchNormalization()(conv)
    activate = Activation('relu')(batch_norm)

    return activate


def trans_conv1D(inputs, model_width, kernel, multiplier):
    # 1D Transposed Convolutional Block, instead of UpSampling
    transposed_conv = Conv1DTranspose(model_width * multiplier, kernel, strides=2, padding='same')(inputs)  # Stride = 2
    batch_norm = BatchNormalization()(transposed_conv)
    activate = Activation('relu')(batch_norm)

    return activate


def Concat_Block(input1, *argv):
    # Concatenation Block from the KERAS Library
    cat = input1
    for arg in range(0, len(argv)):
        cat = concatenate([cat, argv[arg]], axis=-1)

    return cat


def upConv_Block(inputs):
    # 1D UpSampling Block
    up = UpSampling1D(size=2)(inputs)

    return up


def Feature_Extraction_Block(inputs, model_width, Dim2, feature_number):
    # Feature Extraction Block for the AutoEncoder Mode
    latent = Flatten()(inputs)
    latent = Dense(feature_number, name='features')(latent)
    latent = Dense(model_width * Dim2)(latent)
    latent = Reshape((Dim2, model_width))(latent)

    return latent


'''Version 2 (v2) use Transposed Convolution instead of UpSampling'''
def UNet(length, model_depth, num_channel, model_width, kernel_size, problem_type='Regression', output_nums=1, D_S=0, A_E=0, *argv):
    """Variable UNet Model Design"""
    # length: Input Signal Length
    # model_depth: Depth of the Model
    # model_width: Width of the Input Layer of the Model
    # num_channel: Number of Channels allowed by the Model
    # kernel_size: Kernel or Filter Size of the Convolutional Layers
    # problem_type: Classification (Binary or Multiclass) or Regression
    # output_nums: Output Classes (Classification Mode) or Features (Regression Mode)
    # ds: Checks where Deep Supervision is active or not, either 0 or 1 [Default value set as 0]
    # ae: Enables or diables the AutoEncoder Mode, either 0 or 1 [Default value set as 0]
    # feature_number: Number of Features or Embeddings to be extracted from the AutoEncoder in the A_E Mode
    """Variable UNet Model Design"""
    if length == 0 or model_depth == 0 or model_width == 0 or num_channel == 0 or kernel_size == 0:
        print("ERROR: Please Check the Values of the Input Parameters!")

    convs = {}
    levels = []
    i = 1

    # Encoding
    inputs = Input((length, num_channel))
    conv = Conv_Block(inputs, model_width, kernel_size, 2 ** 0)
    conv = Conv_Block(conv, model_width, kernel_size, 2 ** 0)
    pool = MaxPooling1D(pool_size=2)(conv)
    convs["conv%s" % i] = conv

    for i in range(2, (model_depth + 1)):
        conv = Conv_Block(pool, model_width, kernel_size, 2 ** (i - 1))
        conv = Conv_Block(conv, model_width, kernel_size, 2 ** (i - 1))
        pool = MaxPooling1D(pool_size=2)(conv)
        convs["conv%s" % i] = conv

    # Collect Latent Features or Embeddings from AutoEncoders
    if (A_E == 0) and (D_S == 0):
        conv = Conv_Block(pool, model_width, kernel_size, 2 ** model_depth)
        conv = Conv_Block(conv, model_width, kernel_size, 2 ** model_depth)
    elif (A_E == 0) and (D_S == 1):
        conv = Conv_Block(pool, model_width, kernel_size, 2 ** model_depth)
        conv = Conv_Block(conv, model_width, kernel_size, 2 ** model_depth)
        level0 = Conv1D(1, 1, name=f'level{model_depth}')(conv)
        levels.append(level0)
    elif (A_E == 1) and (D_S == 0):
        latent = Feature_Extraction_Block(pool, model_width, int(length / (2 ** model_depth)), feature_number)
        conv = Conv_Block(latent, model_width, kernel_size, 2 ** model_depth)
        conv = Conv_Block(conv, model_width, kernel_size, 2 ** model_depth)
    elif (A_E == 1) and (D_S == 1):
        latent = Feature_Extraction_Block(pool, model_width, int(length / (2 ** model_depth)), feature_number)
        conv = Conv_Block(latent, model_width, kernel_size, 2 ** model_depth)
        conv = Conv_Block(conv, model_width, kernel_size, 2 ** model_depth)
        level0 = Conv1D(1, 1, name=f'level{model_depth}')(conv)
        levels.append(level0)
    else:
        print("ERROR: Please Check the Values of the Input Parameters!")

    # Decoding
    convs_list = list(convs.values())
    deconv = Conv_Block(Concat_Block(upConv_Block(conv), convs_list[model_depth - 1]), model_width, kernel_size, 2 ** (model_depth - 1))
    deconv = Conv_Block(deconv, model_width, kernel_size, 2 ** (model_depth - 1))

    for j in range(1, .model_depth):
        if D_S == 0:
            deconv = Conv_Block(Concat_Block(upConv_Block(deconv), convs_list[model_depth - j - 1]), model_width, kernel_size, 2 ** (model_depth - j - 1))
            deconv = Conv_Block(deconv, model_width, kernel_size, 2 ** (model_depth - j - 1))
        elif D_S == 1:
            level = Conv1D(1, 1, name=f'level{model_depth - j}')(deconv)
            levels.append(level)
            deconv = Conv_Block(Concat_Block(upConv_Block(deconv), convs_list[model_depth - j - 1]), model_width, kernel_size, 2 ** (model_depth - j - 1))
            deconv = Conv_Block(deconv, model_width, kernel_size, 2 ** (model_depth - j - 1))
        else:
            print("ERROR: Please Check the Values of the Input Parameters!")

    # Output
    if problem_type == 'Classification':
        outputs = Conv1D(output_nums, 1, activation='softmax', name="out")(deconv)
    elif problem_type == 'Regression':
        outputs = Conv1D(output_nums, 1, activation='linear', name="out")(deconv)

    model = Model(inputs=[inputs], outputs=[outputs])

    if D_S == 1:
        levels.append(outputs)
        levels.reverse()
        model = Model(inputs=[inputs], outputs=levels)

    return model
