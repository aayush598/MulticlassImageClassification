from keras.optimizers.schedules import ExponentialDecay, InverseTimeDecay, PolynomialDecay, PiecewiseConstantDecay
from keras.optimizers import Adam, Adadelta, Adagrad, RMSprop, SGD

def set_trainable(model, value=False):
    """
    Sets the trainable attribute of all layers in the model.

    Args:
        model (keras.Model): The Keras model.
        value (bool): The value to set for the trainable attribute.
    """
    for layer in model.layers:
        layer.trainable = value

def select_learning_rate_scheduler(scheduler_name, lr_rate=0.01, decay_steps=10000, decay_rate=0.9):
    """
    Selects a learning rate scheduler based on the provided name and parameters.

    Args:
        scheduler_name (str): Name of the learning rate scheduler.
        lr_rate (float): Initial learning rate.
        decay_steps (int): Number of steps for decay.
        decay_rate (float): Rate of decay.

    Returns:
        keras.optimizers.schedules.LearningRateSchedule: The selected learning rate scheduler.
    
    Raises:
        ValueError: If the provided scheduler name is invalid.
    """
    if scheduler_name == 'exponential_decay':
        return ExponentialDecay(initial_learning_rate=lr_rate, decay_steps=decay_steps, decay_rate=decay_rate)
    elif scheduler_name == 'inverse_time_decay':
        return InverseTimeDecay(initial_learning_rate=lr_rate, decay_steps=decay_steps, decay_rate=decay_rate)
    elif scheduler_name == 'polynomial_decay':
        return PolynomialDecay(initial_learning_rate=lr_rate, decay_steps=decay_steps, decay_rate=decay_rate)
    elif scheduler_name == 'piecewise_constant_decay':
        return PiecewiseConstantDecay(boundaries=[5000, 10000], values=[0.01, 0.001, 0.0001])
    else:
        raise ValueError(f"Invalid learning rate scheduler name: {scheduler_name}")

def select_optimizer(optimizer_name, learning_rate=0.01):
    """
    Selects an optimizer based on the provided name and learning rate.

    Args:
        optimizer_name (str): Name of the optimizer.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        keras.optimizers.Optimizer: The selected optimizer.
    
    Raises:
        ValueError: If the provided optimizer name is invalid.
    """
    if optimizer_name == 'Adam':
        return Adam(learning_rate=learning_rate)
    elif optimizer_name == 'Adadelta':
        return Adadelta(learning_rate=learning_rate)
    elif optimizer_name == 'Adagrad':
        return Adagrad(learning_rate=learning_rate)
    elif optimizer_name == 'RMSprop':
        return RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'SGD':
        return SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    else:
        raise ValueError('Invalid optimizer name')
    
def train_model(model, training_data_generator, validation_data, batch_size=8, epochs=20, steps_per_epoch=None):
    """
    Train the model.

    Args:
        model (keras.Model): The Keras model to train.
        training_data_generator (keras.utils.Sequence): The data generator for training data.
        validation_data (tuple): Tuple containing validation data (testX, testY).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs.
        steps_per_epoch (int): Number of steps (batches) to yield from the generator during each epoch. 
                               If `None`, the length of `training_data_generator` will be used.

    Returns:
        History: The training history.
    """
    # Train the model
    history = model.fit(
        training_data_generator,
        validation_data=validation_data,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch
    )
    return history