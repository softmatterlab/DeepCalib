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