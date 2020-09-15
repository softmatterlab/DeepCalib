def predict(network, trajectory):
    """ Predict parameters of the force field from the trajectory using the deep learnign network.
    
    Inputs:
    network: deep learning network
    image: trajectroy [numpy array of real numbers]
    
    Output:
    predicted_targets: predicted parameters of the calibrated force field [1D numpy array containing outputs]
    """
    
    from numpy import reshape
    
    network_blocksize = network.get_layer(index=0).get_config()['batch_input_shape'][1:][1]
    predicted_targets = network.predict(reshape(trajectory, [1,round(trajectory.size/network_blocksize),network_blocksize]))   
        
    return predicted_targets