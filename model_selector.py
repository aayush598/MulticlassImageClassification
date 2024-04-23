import keras
from keras.layers import Input

def load_keras_application_model(model_name):
    """
    Load a Keras application model by name.

    Args:
        model_name (str): Name of the Keras application model.

    Returns:
        keras.Model or None: The loaded Keras application model if found, otherwise None.
    """
    # Dictionary mapping model names to their corresponding Keras application classes
    models = {
        'VGG16': keras.applications.VGG16,
        'VGG19': keras.applications.VGG19,
        'ResNet50': keras.applications.ResNet50,
        'ResNet101': keras.applications.ResNet101,
        'ResNet152': keras.applications.ResNet152,
        'InceptionV3': keras.applications.InceptionV3,
        'Xception': keras.applications.Xception,
        'MobileNet': keras.applications.MobileNet,
        'MobileNetV2': keras.applications.MobileNetV2,
        'DenseNet121': keras.applications.DenseNet121,
        'DenseNet169': keras.applications.DenseNet169,
        'DenseNet201': keras.applications.DenseNet201,
        'NASNetMobile': keras.applications.NASNetMobile,
        'NASNetLarge': keras.applications.NASNetLarge
    }
    
    # Check if the provided model_name exists in the models dictionary
    if model_name in models:
        # Instantiate the selected model with default parameters
        return models[model_name](weights="imagenet", include_top=False, input_shape=(224, 224, 3), input_tensor=Input(shape=(224, 224, 3)))
    else:
        # Return None if the provided model_name does not exist
        return None
