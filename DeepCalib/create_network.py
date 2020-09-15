def create_deep_learning_network(
    input_shape=(20, 50),
    lstm_layers_dimensions=(1000, 250, 50),
    number_of_outputs=1) :
    """Creates and compiles a deep learning network.
    
    Inputs:    
    input_shape: Should be the same size of the input trajectory []
    lstm_layers_dimensions: number of neurons in each LSTM layer [tuple of positive integers]
        
    Output:
    network: deep learning network
    """    

    from keras import models, layers, optimizers
    
    ### INITIALIZE DEEP LEARNING NETWORK
    network = models.Sequential()

    ### CONVOLUTIONAL BASIS
    for lstm_layer_number, lstm_layer_dimension in zip(range(len(lstm_layers_dimensions)), lstm_layers_dimensions):

        # add LSTM layer
        lstm_layer_name = 'lstm_' + str(lstm_layer_number + 1)
        if lstm_layer_number + 1 < len(lstm_layers_dimensions): # All layers but last
            lstm_layer = layers.LSTM(lstm_layer_dimension,
                                     return_sequences=True,
                                     dropout=0,
                                     recurrent_dropout=0,
                                     input_shape=input_shape,
                                     name=lstm_layer_name)
        else: # Last layer
            lstm_layer = layers.LSTM(lstm_layer_dimension,
                                     return_sequences=False,
                                     dropout=0,
                                     recurrent_dropout=0,
                                     input_shape=input_shape,
                                     name=lstm_layer_name)

        network.add(lstm_layer)
    # OUTPUT LAYER
    output_layer = layers.Dense(number_of_outputs, name='output')
    network.add(output_layer)
    
    network.compile(optimizer=optimizers.Adam(lr=1e-3), loss='mse', metrics=['mse', 'mae'])
    
    return network
