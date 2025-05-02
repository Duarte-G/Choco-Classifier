import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# configurações gerais
dataset_dir = 'dataset_processado'
img_size = (86, 86)
batch_size = 32
num_classes = len(next(os.walk(dataset_dir))[1])

# carregar datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

# pré-processamento
autotune = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(autotune)
val_ds = val_ds.cache().prefetch(autotune)

# construir modelo CNN com transfer learning
def build_model():
    base = ResNet50V2(include_top=False, weights='imagenet', input_shape=(*img_size, 3))
    for layer in base.layers[:-30]:
        layer.trainable = False
    x = Input(shape=(*img_size, 3))
    y = base(x)
    y = GlobalAveragePooling2D()(y)
    y = BatchNormalization()(y)
    y = Dense(512, activation='relu')(y)
    y = Dropout(0.5)(y)
    outputs = Dense(num_classes, activation='softmax')(y)
    model = Model(inputs=x, outputs=outputs)
    return model

model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# callbacks
os.makedirs('checkpoints', exist_ok=True)
callbacks = [
    ModelCheckpoint('checkpoints/best_model.keras', save_best_only=True, monitor='val_accuracy'),
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
]

# treinar
history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks)

# avaliar
val_images, val_labels = [], []
for imgs, labs in val_ds:
    val_images.append(imgs.numpy())
    val_labels.append(labs.numpy())
val_images = np.concatenate(val_images)
val_labels = np.concatenate(val_labels)

pred_probs = model.predict(val_images)
pred_labels = np.argmax(pred_probs, axis=1)

# matriz de confusão e relatório
cm = confusion_matrix(val_labels, pred_labels)
print('Confusion Matrix:\n', cm)
print(classification_report(val_labels, pred_labels))

# curvas ROC (one-vs-rest)
fpr = dict(); tpr = dict(); roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve((val_labels==i).astype(int), pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# plot AUC média
plt.figure()
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()
