from .create_network import *
from .plot_learning_performance import *
from .plot_sample_trajectories import * 
from .plot_test_performance import *
from .predict import * 
from .train_network import *


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
        
        
    
     