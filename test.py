import tensorflow as tf
import numpy as np
interpreter = tf.lite.Interpreter(model_path="tflteMovenet/movenet.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set input tensor to float32
input_shape = input_details[0]['shape']
input_tensor = np.zeros(input_shape, dtype=np.int32)

interpreter.set_tensor(input_details[0]['index'], input_tensor)

# Set output tensor to float32
output_shape = output_details[0]['shape']
output_tensor = np.zeros(output_shape, dtype=np.float32)
interpreter.set_tensor(output_details[0]['index'], output_tensor)

interpreter.invoke()