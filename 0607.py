import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import random

#1
# 資料路徑
base_dir = r"C:\Users\Yu Shan\Downloads\base_dir"

# 影像參數
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# 資料增強與標準化
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.7,1.3],
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    channel_shift_range=20
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# 建立資料產生器
train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    os.path.join(base_dir, 'validation'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(base_dir, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)



#2
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[early_stop]
)



#3# 準確率曲線
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 損失函數曲線
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#4
# 評估測試集
test_loss, test_acc = model.evaluate(test_generator)
print(f"測試集準確率: {test_acc*100:.2f}%")

# 混淆矩陣
y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=train_generator.class_indices,
            yticklabels=train_generator.class_indices)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# 類別報告
target_names = list(train_generator.class_indices.keys())
report = classification_report(y_true, y_pred_classes, target_names=target_names, output_dict=True)
import pandas as pd
report_df = pd.DataFrame(report).transpose()
print(report_df)

#5
# 顯示隨機正確與錯誤案例各 5
filenames = test_generator.filenames
correct = np.where(y_true == y_pred_classes)[0]
incorrect = np.where(y_true != y_pred_classes)[0]

def show_samples(indices, title):
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(indices[:5]):
        img_path = os.path.join(test_generator.directory, filenames[idx])
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.title(f"Pred: {target_names[y_pred_classes[idx]]}\nTrue: {target_names[y_true[idx]]}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

show_samples(random.sample(list(correct), 5), "Correct Predictions")
show_samples(random.sample(list(incorrect), 5), "Incorrect Predictions")
