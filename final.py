
from keras.models import Model
from sklearn.model_selection import train_test_split
from topLayers import *
from model_selector import load_keras_application_model
from data_utils import rename_images, convert_path_to_df, data_augmentation
from model_utils import set_trainable, select_learning_rate_scheduler, select_optimizer, train_model
from model_evaluation import evaluate_model
import os

def training(combination=[]):
	
	dataset, categories, filename_rename, model_name, top_model_name, scheduler_name, optimizer, loss_function, batch_size, epochs, model_file_name, tflite_file_name = combination
	if (categories==[]):
		categories = os.listdir(dataset)
	try:
		rename_images(dataset=dataset, categories=categories, filename=filename_rename)
	except:
		print("Renamed file already exist")

	# Example usage
	image_df = convert_path_to_df(dataset, categories=categories)

	train_df, test_df = train_test_split(image_df, test_size=0.2, shuffle=True, random_state=42)
	
	train_images, val_images, test_images = data_augmentation(train_df=train_df, test_df=test_df)

	baseModel = load_keras_application_model(model_name=model_name)

	set_trainable(model=baseModel,value=False)

	topModel = create_top_model(base_model=baseModel, top_model_name=top_model_name, categories=len(categories))

	model = Model(inputs=baseModel.input, outputs=topModel)

	lr_schedule = select_learning_rate_scheduler(scheduler_name=scheduler_name)

	opt = select_optimizer(optimizer_name=optimizer, learning_rate=lr_schedule)

	model.compile(loss=loss_function, optimizer=opt,metrics=["accuracy"])

	history = train_model(
		model=model,
    	training_data_generator=train_images,
    	validation_data=val_images,
		batch_size = batch_size,
    	epochs=epochs
	)
	model.save(model_file_name, 'h5')

	# inference_time_model = measure_inference_time(model_path = model_file_name,testX=testX)

	# convert_to_tflite(model=model, filename=tflite_file_name)

	# accuracy = tflite_prediction(testX=testX, testY=testY)

	# inference_time_tflite_model = tflite_inference(testX=testX)
 
	evaluate_model(model=model, test_images=test_images)
	