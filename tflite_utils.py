import tensorflow as tf
import time
import numpy as np

def convert_to_tflite(model, filename='tflite_model.tflite'):
    """
    Convert a Keras model to TensorFlow Lite format.

    Args:
        model (keras.Model): The Keras model to convert.
        filename (str): The filename to save the TensorFlow Lite model.

    Returns:
        None
    """
    # Convert the Keras model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save the TensorFlow Lite model to a file
    with open(filename, 'wb') as f:
        f.write(tflite_model)

def tflite_prediction(filepath='converted_model_optimization.tflite', testX=[], testY=[]):
    """
    Perform predictions using a TensorFlow Lite model.

    Args:
        filepath (str): The file path of the TensorFlow Lite model.
        testX (list): List of input images for prediction.
        testY (list): List of true labels (optional, not used).

    Returns:
        list: Predicted labels
    """
    # Initialize TensorFlow Lite interpreter
    interpreter = tf.lite.Interpreter(model_path=filepath)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Perform predictions
    predictions = []
    for image in testX:
        interpreter.set_tensor(input_details[0]['index'], [image])
        interpreter.invoke()
        predictions.append(interpreter.get_tensor(output_details[0]['index']))
    
    # Convert predictions to labels
    predicted_labels = [np.argmax(prediction) for prediction in predictions]
    return predicted_labels

def tflite_inference(filepath='converted_model_optimization.tflite', testX=[]):
    """
    Measure inference time of a TensorFlow Lite model.

    Args:
        filepath (str): The file path of the TensorFlow Lite model.
        testX (list): List of input images for inference.

    Returns:
        float: Average inference time per image (in seconds).
    """
    # Initialize TensorFlow Lite interpreter
    interpreter = tf.lite.Interpreter(model_path=filepath)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    
    # Perform inference and measure time
    start_time = time.time()
    for image in testX:
        interpreter.set_tensor(input_details[0]['index'], [image])
        interpreter.invoke()
    end_time = time.time()
    
    # Calculate average inference time per image
    inference_time = (end_time - start_time) / len(testX)
    return inference_time
