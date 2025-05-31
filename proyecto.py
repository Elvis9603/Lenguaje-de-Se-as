import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('proyecto.h5')

labels = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
          'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U',
          'V', 'W', 'X', 'Y', 'Z']

cap = cv2.VideoCapture(0)

word = ""
current_letter = ""
prev_letter = ""
confidence_threshold = 0.90

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    x1, y1, x2, y2 = 400, 100, 600, 300
    roi = frame[y1:y2, x1:x2]

    roi_resized = cv2.resize(roi, (64, 64))
    roi_normalized = roi_resized / 255.0
    roi_input = np.expand_dims(roi_normalized, axis=0)

    prediction = model.predict(roi_input)
    max_prob = np.max(prediction[0])
    max_index = np.argmax(prediction[0])
    predicted_label = labels[max_index]

    # Solo si la predicción tiene suficiente confianza
    if max_prob > confidence_threshold and predicted_label != 'nothing':
        current_letter = predicted_label

        if current_letter != prev_letter:
            prev_letter = current_letter

            if current_letter == 'space':
                word += ' '
            elif current_letter == 'del':
                word = word[:-1]
            else:
                word += current_letter

    # Dibujar ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Mostrar letra reconocida
    cv2.putText(frame, f'Letra: {current_letter.upper()}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    # Mostrar palabra construida
    cv2.putText(frame, f'Palabra: {word}', (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Reconocimiento de señas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
