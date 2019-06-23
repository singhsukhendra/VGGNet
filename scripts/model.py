# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/6/17 20:35
   desc: the project
"""
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Model
from keras.optimizers import SGD


def VGG13(input_shape=(224, 224, 3), n_classes=1000):
    """
    实现VGG13的网络结构
    :param input_shape: 输入图片(H, W, C)尊重设计者这里使用224输入
    :return:
    """
    # input layer
    input_layer = Input(shape=input_shape)
    # block1
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = Conv2D(64, 3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # block2
    x = Conv2D(128, 3, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(128, 3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # block3
    x = Conv2D(256, 3, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # block4
    x = Conv2D(512, 3, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # block5
    x = Conv2D(512, 3, strides=1, padding='same', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, 2, padding='same')(x)
    # fc
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    output_layer = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    sgd = SGD(momentum=0.9, lr=0.01, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    VGG_13 = VGG13()
    print(VGG_13.summary())
