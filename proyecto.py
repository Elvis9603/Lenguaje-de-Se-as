import cv2
import numpy as np
import tensorflow as tf
from collections import deque

model = tf.keras.models.load_model('proyecto.h5')

CLASSES = ['a', 'b', 'c', 'd', 'del', 'e', 'f', 'g', 'h', 'i',
           'j', 'k', 'l', 'm', 'n', 'nothing', 'o', 'p', 'q',
           'r', 's', 'space', 't', 'u', 'v', 'w', 'x', 'y', 'z']

IMG_SIZE = 64
cap = cv2.VideoCapture(0)

letter_buffer = deque(maxlen=15)
current_text = ""

def preprocess_image(roi):
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi = roi.astype('float32') / 255.0
    roi = np.expand_dims(roi, axis=0)
    return roi

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Aquí cambiamos el tamaño del cuadro ROI a 400x400 píxeles
    # Por ejemplo, desde w-400 a w-0 y de 50 a 450 (ajusta según necesites)
    x1, y1, x2, y2 = w - 400, 50, w - 0, 400

    # Puedes quitar el rectángulo si no quieres que se vea
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    roi = frame[y1:y2, x1:x2]

    if roi.shape[0] > 0 and roi.shape[1] > 0:
        processed_roi = preprocess_image(roi)
        prediction = model.predict(processed_roi, verbose=0)
        class_index = np.argmax(prediction)
        predicted_letter = CLASSES[class_index]

        cv2.putText(frame, f"Letra: {predicted_letter}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        if predicted_letter == "space":
            current_text += " "
        elif predicted_letter == "del":
            current_text = current_text[:-1]
        elif predicted_letter != "nothing":
            letter_buffer.append(predicted_letter)

        if len(letter_buffer) == letter_buffer.maxlen:
            if all(l == letter_buffer[0] for l in letter_buffer):
                current_text += letter_buffer[0]
                letter_buffer.clear()

    cv2.putText(frame, f"Palabra: {current_text}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Reconocimiento de señas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
