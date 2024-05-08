import keras
import numpy as np



class NeuralNetwork:
    def __init__(self,output=4):
        self.model = None
        self.output = output
        input_layer = keras.layers.Input(shape=(8,8))
        flatten_layer = keras.layers.Flatten()(input_layer)
        hid_layer1 = keras.layers.Dense(128,activation='relu')(flatten_layer)
        hid_layer2 = keras.layers.Dense(64,activation='relu')(hid_layer1)
        output_layer = keras.layers.Dense(self.output,activation='softmax')(hid_layer2)
        self.model = keras.models.Model(inputs=input_layer,outputs=output_layer)
            
    def forward_pass(self,X_test:np.ndarray):
        ''''
            it takes the X_test which is 2d matrix(Board)
            model predict the given data and return the prediction
        '''
        y_pred = self.model.predict(X_test)
        y_pred = np.ceil(y_pred)
        return y_pred
    
    #get the weights of the model after prediction to optimize it
    def get_weights(self):
        weights = self.model.get_weights()
        return weights

    #save model weights to h5 file
    def save_weights(self,path:str):
        self.model.save_weights(f'{path}.h5')
    
    #laod weights from h5 file
    def load_weights(self,path:str):
        self.model.load_weights(f'{path}.h5')


#testing
array = [[
    #first initaliztion
    [0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [2, 0, 2, 0, 2, 0, 2, 0],
    [0, 2, 0, 2, 0, 2, 0, 2],
    [2, 0, 2, 0, 2, 0, 2, 0]
    #0 for empty cell
    #1 for player 1
    #2 for player 2
    #3 for player 1 that is king
    #4 for player 2 that is king
]]

array = np.array(array)

#how to make instance and model from it
nn = NeuralNetwork()
pred = nn.forward_pass(array)
print(pred)
# weights = model.get_weights()
# print(weights)
