''' DeepCalib 1.0
Parameter estimation of force fields with Deep Learning
version 1.0 - 27 April 2020
© Aykut Argun, Tobias Thalheim, Stefano Bo, Frank Cichos & Giovanni Volpe
http://www.softmatterlab.org
'''

class trajectory:
  def __init__(self, names, values, scalings, scaled_values):
    self.names = names
    self.values = values
    self.scalings = scalings
    self.scaled_values = scaled_values
 

class targets:
  def __init__(self, names, values, scalings, scaled_values):
    self.names = names
    self.values = values
    self.scalings = scalings
    self.scaled_values = scaled_values

    
def load_data(data_name):
    import scipy.io as sci
    x, y = sci.loadmat(data_name)['x'], sci.loadmat(data_name)['y']
    x = x.reshape(x.shape[1],)
    y = y.reshape(y.shape[1],)
    return x,y
    
    
def plot_sample_data(x,y):
    
    import numpy as np
    import matplotlib.pyplot as plt
    
   
    
    ### Plot the data 
    
    fig = plt.figure(figsize=(15, 10))

    fntsize = 20
    
    plt.axes([.1, .59, .9, .41])
    plt.plot(np.arange(len(x))*0.01,x)
    plt.ylabel('$x$ [\u03BCm]',fontsize=fntsize)
    plt.xlabel('$t$ [s]',fontsize=fntsize)
    plt.title('$x$ vs $t$',fontsize=fntsize+2)
    plt.axes([.1, .05, .9, .41])
    plt.plot(np.arange(len(y))*0.01,y)
    plt.ylabel('$y$ [\u03BCm]',fontsize=fntsize)
    plt.xlabel('$t$ [s]',fontsize=fntsize)
    plt.title('$y$ vs $t$',fontsize=fntsize+2)
        
        
def plot_sample_trajectories(simulate_trajectory, number_of_trajectories_to_show):
    """
    
    Inputs:   
    
    simulate_trajectory:                 trajectroy generator function
    number_of_trajectories_to_show:      number of trajectories to be plotted as sample  
        
    Output:
    
    This function does not return any outputs, it only plots some sample trajectories to preview the generator function
   
    """ 
    import matplotlib.pyplot as plt
    inputs, targets = simulate_trajectory(number_of_trajectories_to_show)
    
    if len(inputs.names)>1:
        for scaled_values, scaled_targets, target_values in zip(inputs.scaled_values, targets.scaled_values, targets.values):
            plt.figure(figsize=(20, 5))    
            number_subplots = len(inputs.names)
            for subplot in range(number_subplots):
                plt.subplot(1, number_subplots, subplot+1)
                plt.plot(scaled_values[subplot])
                plt.xlabel('timestep', fontsize=18)
                plt.ylabel(inputs.scalings[subplot], fontsize=18)   

            title_text = 'Parameters: ' 
            for parameter in range(len(target_values)):
                title_text +=  ', ' + targets.names[parameter] + ' = ' + '%1.2e' % target_values[parameter]
            plt.title(title_text, fontsize=18)  

    else:
        for scaled_values, scaled_targets, target_values in zip(inputs.scaled_values, targets.scaled_values, targets.values):
            plt.figure(figsize=(20, 5))
            plt.plot(scaled_values)
            plt.xlabel('timestep', fontsize=18)
            plt.ylabel(inputs.scalings[0], fontsize=18)
            title_text = 'Parameters: ' 
            if len(targets.names)>1:
                for parameter in range(len(targets.names)):
                    title_text +=  ', ' + targets.names[parameter] + ' = ' + '%1.2e' % target_values[parameter]
                plt.title(title_text, fontsize=18)  
            else:
                plt.title(title_text + ' ' + targets.names[0] + ' = ' + '%1.2e' % target_values , fontsize=18)  
    
            
            
        

      
  
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


def train_deep_learning_network(
    network,
    simulate_trajectory,
    sample_sizes = (32, 128, 512, 2048),
    iteration_numbers = (3001, 2001, 1001, 101),
    verbose=True):
    """Train a deep learning network.
    
    Input:
    network: deep learning network
    simulate_trajectory: trajectory generator function
    sample_sizes: sizes of the batches of trajectories used in the training [tuple of positive integers]
    iteration_numbers: numbers of batches used in the training [tuple of positive integers]
    verbose: frequency of the update messages [number between 0 and 1]
        
    Output:
    training_history: dictionary with training history
    """  
    
    import numpy as np
    from time import time
     
    training_history = {}
    training_history['Sample Size'] = []
    training_history['Iteration Number'] = []
    training_history['Iteration Time'] = []
    training_history['MSE'] = []
    training_history['MAE'] = []
    
    for sample_size, iteration_number in zip(sample_sizes, iteration_numbers):
        for iteration in range(iteration_number):
            
            # meaure initial time for iteration
            initial_time = time()

            # generate trajectories and targets
            trajectory_shape = network.get_layer(index=0).get_config()['batch_input_shape'][1:]
            input_shape = (sample_size, ) + tuple(trajectory_shape)
            trajectories = np.zeros(input_shape)
                        
            number_of_outputs = network.get_layer(index=-1).get_config()['units']
            output_shape = (sample_size, number_of_outputs)
            targets = np.zeros(output_shape)
            
            
            
            trajectory, target = simulate_trajectory(sample_size)
            trajectory = trajectory.scaled_values
            trajectories = np.array(trajectory).reshape(sample_size,trajectory_shape[0],trajectory_shape[1])
            targets = target.scaled_values
                
                

            # training
            history = network.fit(trajectories,
                                targets,
                                epochs=1, 
                                batch_size=sample_size,
                                verbose=False)
                        
            # measure elapsed time during iteration
            iteration_time = time() - initial_time

            # record training history
            mse = history.history['mse'][0]
            mae = history.history['mae'][0]
                        
            training_history['Sample Size'].append(sample_size)
            training_history['Iteration Number'].append(iteration)
            training_history['Iteration Time'].append(iteration_time)
            training_history['MSE'].append(mse)
            training_history['MAE'].append(mae)

            if not(iteration%int(verbose**-1)):
                print('Sample size %6d   iteration number %6d   MSE %10.4f   MAE %10.4f   Time %10f ms' % (sample_size, iteration + 1, mse, mae, iteration_time * 1000))
                
    return training_history


def plot_learning_performance(training_history, number_of_timesteps_for_average = 100, figsize=(20,20)):
    """Plot the learning performance of the deep learning network.
    
    Input:
    training_history: dictionary with training history, typically obtained from train_deep_learning_network()
    number_of_timesteps_for_average: length of the average [positive integer number]
    figsize: figure size [list of two positive numbers]
        
    Output: none
    """    

    import matplotlib.pyplot as plt
    from numpy import convolve, ones
    
    plt.figure(figsize=figsize)

    plt.subplot(5, 1, 1)
    plt.semilogy(training_history['MSE'], 'k')
    plt.semilogy(convolve(training_history['MSE'], ones(number_of_timesteps_for_average) / number_of_timesteps_for_average, mode='valid'), 'r')
    plt.ylabel('MSE', fontsize=24)
    plt.xlabel('Epochs', fontsize=24)

    plt.subplot(5, 1, 2)
    plt.semilogy(training_history['MAE'], 'k')
    plt.semilogy(convolve(training_history['MAE'], ones(number_of_timesteps_for_average) / number_of_timesteps_for_average, mode='valid'), 'r')
    plt.ylabel('MAE', fontsize=24)
    plt.xlabel('Epochs', fontsize=24)
    plt.show()





def predict(network, trajectory):
    """ Predict parameters of the force field from the trajectory using the deep learnign network.
    
    Inputs:
    network: deep learning network
    image: trajectroy [numpy array of real numbers]
    
    Output:
    predicted_targets: predicted parameters of the calibrated force field [1D numpy array containing outputs]
    """
    
    from numpy import reshape
    
    trajectory_shape = network.get_layer(index=0).get_config()['batch_input_shape'][1:]
    input_shape = (1, ) + tuple(trajectory_shape)

    predicted_targets = network.predict(reshape(trajectory, input_shape))   
        
    return predicted_targets


def plot_test_performance(simulate_trajectory, network, rescale_targets, number_of_predictions_to_show=100, dt = 1e-1):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    trajectory_shape = network.get_layer(index=0).get_config()['batch_input_shape'][1:]


    predictions_scaled = []
    predictions_physical = []

    trajectory, targets = simulate_trajectory(number_of_predictions_to_show)
    targets_physical = list(targets.values)
    targets_scaled = list(targets.scaled_values)
    trajectory = trajectory.scaled_values
    trajectories = np.array(trajectory).reshape(number_of_predictions_to_show,trajectory_shape[0],trajectory_shape[1])


    for i in range(number_of_predictions_to_show):
        predictions = predict(network, trajectories[i].reshape(trajectory_shape))


        predictions_scaled.append(predictions[0])
        predictions_physical.append(rescale_targets(*predictions[0]))

    number_of_outputs = network.get_layer(index=-1).get_config()['units']    

    targets_physical = np.array(targets_physical).transpose()
    targets_scaled = np.array(targets_scaled).transpose()
    predictions_scaled = np.array(predictions_scaled).transpose()
    predictions_physical = np.array(predictions_physical).transpose()

    # Do not show results at the edges of the training range 

    if number_of_outputs>1:

        ind = np.isfinite(targets_scaled[0])
        for target_number in range(number_of_outputs):
            target_max = .9 * np.max(targets_scaled[target_number]) + .1 * np.min(targets_scaled[target_number])
            target_min = .1 * np.max(targets_scaled[target_number]) + .9 * np.min(targets_scaled[target_number])
            ind = np.logical_and(ind, targets_scaled[target_number] < target_max)
            ind = np.logical_and(ind, targets_scaled[target_number] > target_min)
    else:
        target_max = .9 * np.max(targets_scaled) + .1 * np.min(targets_scaled)
        target_min = .1 * np.max(targets_scaled) + .9 * np.min(targets_scaled)
        ind = np.logical_and(targets_scaled < target_max, targets_scaled > target_min)



    if number_of_outputs>1:

        for target_number in range(number_of_outputs):
            plt.figure(figsize=(20, 10))

            plt.subplot(121)
            plt.plot(targets_scaled[target_number],
                     predictions_scaled[target_number],
                     '.')
            plt.xlabel(targets.scalings[target_number], fontsize=18)
            plt.ylabel('Predicted ' + targets.scalings[target_number], fontsize=18)
            plt.axis('square')
            plt.title('Prediction performance in scaled units', fontsize=18)

            plt.subplot(122)
            plt.plot(targets_physical[target_number],
                     predictions_physical[target_number],
                    '.')
            plt.xlabel(targets.names[target_number], fontsize=18)
            plt.ylabel('Predicted ' + targets.names[target_number], fontsize=18)
            plt.axis('square')
            plt.title('Prediction performance in real units', fontsize=18)


    else: 
        plt.figure(figsize=(20, 10))

        plt.subplot(121)
        plt.plot(targets_scaled[ind],
                 predictions_scaled.transpose()[ind],
                 '.')
        plt.xlabel(targets.scalings[0], fontsize=18)
        plt.ylabel('Predicted ' + targets.scalings[0], fontsize=18)
        plt.axis('square')
        plt.title('Prediction performance in scaled units', fontsize=18)

        plt.subplot(122)
        plt.plot(targets_physical[ind],
                 predictions_physical.transpose()[ind],
                '.')
        plt.xlabel(targets.names[0], fontsize=18)
        plt.ylabel('Predicted ' + targets.names[0], fontsize=18)
        plt.axis('square')
        plt.title('Prediction performance in real units', fontsize=18)


        
    
     
    