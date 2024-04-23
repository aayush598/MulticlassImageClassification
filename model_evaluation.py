import keras
import time

def measure_inference_time(model_path='Models/final.h5', testX=[]):
    """
    Measure inference time of a model.

    Args:
        model_path (str): Path to the model file.
        testX (list): List of input test data.

    Returns:
        float: Average inference time per sample.
    """
    # Load the model
    model = keras.models.load_model(model_path)
    start_time = time.time()
    # Make predictions
    model.predict(testX, batch_size=8)
    # Calculate inference time per sample
    inference_time_per_sample = (time.time() - start_time) / len(testX)
    return inference_time_per_sample

def evaluate_model(model, test_images=[]):
    results = model.evaluate(test_images, verbose=0)
    print("    Test Loss: {:.5f}".format(results[0]))
    print("Test Accuracy: {:.2f}%".format(results[1] * 100))