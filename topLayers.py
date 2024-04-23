from keras.layers import AveragePooling2D, Flatten, Dense, Dropout

top_models = {
    "top_model_1": {
        "pool_size": (4, 4),
        "dense_units": 64,
        "dropout_rate": 0.5,
        "output_activation": "softmax"
    },
    "top_model_2": {
        "pool_size": (7, 7),
        "dense_units": 256,
        "dropout_rate": 0.5,
        "output_activation": "softmax"
    },
    "top_model_3": {
        "pool_size": (7,7),
        "dense_units": 128,
        "dropout_rate": 0.5,
        "output_activation": "softmax"
    },
    "top_model_4": {
        "pool_size": (7,7),
        "dense_units": 128,
        "dropout_rate": 0.5,
        "output_activation": "softmax"
    },
    # Define 18 more top models with different sets of layers
    "top_model_20": {
        "pool_size": (3, 3),
        "dense_units": 128,
        "dropout_rate": 0.5,
        "output_activation": "softmax"
    }
}

def create_top_model(base_model, top_model_name,categories=2):
    top_model = base_model.output
    top_model = AveragePooling2D(pool_size=top_models[top_model_name]["pool_size"])(top_model)
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(top_models[top_model_name]["dense_units"], activation="relu")(top_model)
    top_model = Dropout(top_models[top_model_name]["dropout_rate"])(top_model)
    top_model = Dense(int(categories), activation=top_models[top_model_name]["output_activation"])(top_model)
    return top_model
