
# importing required packages
import cv2
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


#initialize image data generator
trainData_gen = ImageDataGenerator(rescale=1./255)
validationData_gen = ImageDataGenerator(rescale=1./255)


#preprocess all test images
train_generator = trainData_gen.flow_from_directory(
    '../data/train',
    target_size=(48,48),
    batch_size=32,
    color_mode="grayscale",
    class_mode='categorical'
)

#preprocess all test images
validation_generator = validationData_gen.flow_from_directory(
    '../data/test',
    target_size=(48,48),
    batch_size=32,
    color_mode="grayscale",
    class_mode='categorical'
)

# create model structure
emotion_model = Sequential()

emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(4, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

# Train the neural network/model
emotion_model_info: object = emotion_model.fit(
        train_generator,
        steps_per_epoch=21006 // 32,
        epochs=80,
        validation_data=validation_generator,
        validation_steps=5212 // 32)

# save model structure in jason file
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
emotion_model.save_weights('emotion_model.h5')