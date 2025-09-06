import os
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----- Paths -----
BASE_DIR  = r"D:\py2\kaggle_cataract"  # update if different
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR  = os.path.join(BASE_DIR, "test")
VAL_DIR   = os.path.join(BASE_DIR, "val") if os.path.exists(os.path.join(BASE_DIR, "val")) else TEST_DIR

# ----- Data -----
train_gen = ImageDataGenerator(rescale=1./255)
val_gen   = ImageDataGenerator(rescale=1./255)
test_gen  = ImageDataGenerator(rescale=1./255)

train_generator = train_gen.flow_from_directory(
    TRAIN_DIR, target_size=(224, 224), batch_size=32, class_mode='binary'
)
validation_generator = val_gen.flow_from_directory(
    VAL_DIR, target_size=(224, 224), batch_size=32, class_mode='binary'
)
test_generator = test_gen.flow_from_directory(
    TEST_DIR, target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False
)

# ----- Model -----
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ----- Train -----
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# ----- Evaluate -----
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print("Test loss:", test_loss, "Test accuracy:", test_acc)

# ----- Save -----
save_path = r"D:\py2\saved_model\cataract_cnn.keras"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.save(save_path)
print("✅ Model kaydedildi:", save_path)