import os
import cv2
import numpy as np
import datetime
import sqlite3
from sklearn.neighbors import KNeighborsClassifier
import PySimpleGUI as sg

def create_new_folder():
    folder_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(folder_name)
    return folder_name

def save_image(image, folder, image_name):
    if folder is not None:
        file_name = os.path.join(folder, image_name)
        cv2.imwrite(file_name, image)
    else:
        print("Erro: Nome da pasta é None.")

def create_database():
    conn = sqlite3.connect('iris_recognition.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        dob TEXT,
        cpf TEXT,
        iris_features BLOB
    )''')
    conn.commit()
    conn.close()

def insert_user(name, dob, cpf, iris_features):
    conn = sqlite3.connect('iris_recognition.db')
    c = conn.cursor()
    c.execute('INSERT INTO users (name, dob, cpf, iris_features) VALUES (?, ?, ?, ?)', (name, dob, cpf, iris_features))
    conn.commit()
    conn.close()

def user_exists(name, cpf):
    conn = sqlite3.connect('iris_recognition.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE name = ? AND cpf = ?', (name, cpf))
    user = c.fetchone()
    conn.close()
    return user is not None

def get_user_info():
    layout = [
        [sg.Text('Nome:'), sg.InputText(key='-NAME-')],
        [sg.Text('Data de Nascimento (YYYY-MM-DD):'), sg.InputText(key='-DOB-')],
        [sg.Text('CPF:'), sg.InputText(key='-CPF-')],
        [sg.Button('Salvar'), sg.Button('Cancelar')]
    ]
    window = sg.Window('Cadastro de Usuário', layout)
    event, values = window.read()
    window.close()
    if event == 'Salvar':
        return values['-NAME-'], values['-DOB-'], values['-CPF-']
    return None, None, None

def extract_features(eye_image, size=(64, 64)):
    resized_eye = cv2.resize(eye_image, size)
    return resized_eye.flatten()

def confirm_recognition(prediction, features):
    return prediction

def recognize_user_from_image(eye_image):
    conn = sqlite3.connect('iris_recognition.db')
    c = conn.cursor()
    c.execute('SELECT name, iris_features FROM users')
    users = c.fetchall()
    conn.close()

    if not users:
        return None

    features = extract_features(eye_image)
    feature_vectors = [np.frombuffer(user[1], dtype=np.uint8) for user in users]
    labels = [user[0] for user in users]

    n_neighbors = min(3, len(feature_vectors))
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(feature_vectors, labels)

    prediction = classifier.predict([features])
    return prediction[0] if prediction.size > 0 else None

def get_user_id(name):
    conn = sqlite3.connect('iris_recognition.db')
    c = conn.cursor()
    c.execute('SELECT id FROM users WHERE name = ?', (name,))
    user_id = c.fetchone()
    conn.close()
    return user_id[0] if user_id else None

def draw_text_on_image(frame, texts, position, font_scale=0.8, color=(0, 255, 0), thickness=2):
    y0, dy = position
    for i, text in enumerate(texts):
        y = y0 + i * 30
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def draw_rectangle_around_head(frame, x, y, w, h):
    # Ajuste o retângulo para cobrir a cabeça toda
    padding = 20
    cv2.rectangle(frame, (x - padding, y - padding), (x + w + padding, y + h + padding), (0, 255, 0), 2)

if os.path.exists('iris_recognition.db'):
    os.remove('iris_recognition.db')

create_database()

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao inicializar a câmera.")
    exit()

folder_name = None
registering = False
user_info = {}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o frame.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (i, (x, y, w, h)) in enumerate(eyes):
        eye = gray_frame[y:y+h, x:x+w]

        if registering:
            if folder_name is None:
                folder_name = create_new_folder()
                print(f"Pasta criada: {folder_name}")

            image_name = f"iris_{i}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            save_image(eye, folder_name, image_name)

            if user_exists(user_info['name'], user_info['cpf']):
                print("Usuário já cadastrado. Por favor, registre um usuário diferente.")
            else:
                features = extract_features(eye)
                features_blob = features.tobytes()
                insert_user(user_info['name'], user_info['dob'], user_info['cpf'], features_blob)
                print("Usuário registrado com sucesso!")
            
            registering = False
            folder_name = None
            user_info = {}

        else:
            if cv2.waitKey(1) & 0xFF == ord('r'):
                name, dob, cpf = get_user_info()
                if name and dob and cpf:
                    user_info = {'name': name, 'dob': dob, 'cpf': cpf}
                    registering = True
                else:
                    print("Cadastro cancelado ou informações não fornecidas.")

        recognized_user = recognize_user_from_image(eye)

        if recognized_user:
            user_id = get_user_id(recognized_user)
            if user_id:
                draw_text_on_image(frame, [f"Nome: {recognized_user}"], (frame.shape[0] - 10, 5))
            else:
                draw_text_on_image(frame, ["Usuário não reconhecido"], (frame.shape[0] - 10, 5))
        else:
            draw_text_on_image(frame, ["Usuário não reconhecido"], (frame.shape[0] - 10, 5))

        draw_rectangle_around_head(frame, x, y, w, h)

    cv2.imshow('Reconhecimento de Íris', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
