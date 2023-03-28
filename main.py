import tensorflow as tf

# Carrega o modelo MoveNet Multi-Person no formato TF2 SavedModel
model = tf.saved_model.load('movenet_singlepose_lightning_4/')


converter = tf.lite.TFLiteConverter.from_saved_model('movenet_singlepose_lightning_4/')
converter.optimizations = [tf.lite.Optimize.DEFAULT]


tflite_model = converter.convert()

# Salva o modelo TensorFlow Lite em um arquivo
with open('tflteMovenet/movenet.tflite', 'wb') as f:
  f.write(tflite_model)