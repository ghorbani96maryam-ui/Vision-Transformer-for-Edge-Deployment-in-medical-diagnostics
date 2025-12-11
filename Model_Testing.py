import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score

test_dir = "/path/to/test"
img_size = 224
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)

test_data = datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=False
)

def eval_tflite(interpreter, data):
    input_idx = interpreter.get_input_details()[0]['index']
    output_idx = interpreter.get_output_details()[0]['index']

    preds, trues = [], []

    for images, labels in data:
        for i in range(images.shape[0]):
            img = images[i][np.newaxis, ...]
            interpreter.set_tensor(input_idx, img.astype(np.float32))
            interpreter.invoke()
            out = interpreter.get_tensor(output_idx)[0]

            preds.append(np.argmax(out))
            trues.append(np.argmax(labels[i]))

        if len(trues) >= data.samples:
            break

    acc = np.mean(np.array(preds) == np.array(trues))
    f1 = f1_score(trues, preds, average='weighted')
    return acc, f1

interpreter = tf.lite.Interpreter(model_path="/path/to/model.tflite")
interpreter.allocate_tensors()

acc, f1 = eval_tflite(interpreter, test_data)
print("Accuracy:", acc)
print("F1 Score:", f1)