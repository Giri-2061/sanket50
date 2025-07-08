# train_model.py
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

def train_model(data_dir, model_save_path, label_map_path):
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(label_map_path), exist_ok=True)
    
    # Stronger data augmentation
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # Use ResNet preprocessing
        validation_split=0.2,
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        brightness_range=[0.7, 1.3],
        channel_shift_range=30.0,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Training and validation generators
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    # Save label mappings
    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump(train_generator.class_indices, f, ensure_ascii=False, indent=4)
    print("Class indices saved to", label_map_path)
    
    # Improved model: ResNet50 backbone + custom head
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(128, 128, 3)
    )
    base_model.trainable = False  # Freeze base for transfer learning

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(train_generator.num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Callbacks for better training
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=30,
        callbacks=callbacks
    )
    
    # Optionally unfreeze some layers and fine-tune
    base_model.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    history_finetune = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=10,
        callbacks=callbacks
    )

    # Accuracy Plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✅ Model training complete. Best model saved to {model_save_path}")

if __name__ == "__main__":
    data_dir = "data/captured_signs"  # NEW path to your own dataset
    model_save_path = "models/sign_language_model.h5"
    label_map_path = "models/class_indices.json"
    
    train_model(data_dir, model_save_path, label_map_path)