import keras


def build_model():
    input_layer = keras.layers.Input(shape=((8,8)),name='input_layer')
    flatten_layer = keras.layers.Flatten()(input_layer)
    hid_layer1 = keras.layers.Dense(128,activation='relu')(flatten_layer)
    hid_layer2 = keras.layers.Dense(64,activation='relu')(hid_layer1)
    output_layer = keras.layers.Dense(4,activation='softmax')(hid_layer2)
    model = keras.models.Model(inputs=input_layer,outputs=output_layer)
    
    return model


model = build_model()
