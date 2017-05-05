from tensorflow.python import pywrap_tensorflow  
import os
model_dir = "save/srGAN1/"
checkpoint_path = os.path.join(model_dir, "srGAN")  
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  
var_to_shape_map = reader.get_variable_to_shape_map()  
for key in var_to_shape_map:  
    print("tensor_name: ", key)  
