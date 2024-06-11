import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QLineEdit, QListWidget, \
    QListWidgetItem, QDialog, QHBoxLayout, QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QIcon
import cv2
import sqlite3
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from datetime import datetime

# Conexión a la base de datos
conn = sqlite3.connect('Caras.db')
cursor = conn.cursor()
#Creado por CiberDosis. Derechos reservados.

# Crear la tabla si no existe
cursor.execute('''
    CREATE TABLE IF NOT EXISTS CaraPersonas (
        ID INTEGER PRIMARY KEY,
        nombre TEXT,
        imagen BLOB,
        fecha_hora TEXT
    )
''')

# Crear la tabla Pasos si no existe
cursor.execute('''
    CREATE TABLE IF NOT EXISTS Pasos (
        ID INTEGER PRIMARY KEY,
        nombre TEXT,
        fecha_hora TEXT
    )
''')


class ListaCaras(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Lista de Caras')
        self.setGeometry(100, 100, 480, 320)

        # Lista de caras
        self.list_widget = QListWidget(self)
        self.list_widget.itemClicked.connect(self.mostrar_imagen_seleccionada)  # Conectar la señal itemClicked

        # Botón para eliminar una cara
        self.btn_eliminar = QPushButton('Eliminar Cara', self)
        self.btn_eliminar.clicked.connect(self.eliminar_cara)

        # Botón para cerrar la ventana
        self.btn_cerrar = QPushButton('Cerrar', self)
        self.btn_cerrar.clicked.connect(self.close)


        # Diseño de la interfaz
        layout = QVBoxLayout()
        layout.addWidget(self.list_widget)
        layout.addWidget(self.btn_eliminar)
        layout.addWidget(self.btn_cerrar)
        self.setLayout(layout)

        # Mostrar la lista de caras
        self.mostrar_caras()

    def mostrar_caras(self):
        self.list_widget.clear()
        cursor.execute("SELECT * FROM CaraPersonas")
        rows = cursor.fetchall()

        for row in rows:
            nombre = row[1]
            fecha_hora = row[3]  # Obtener la fecha/hora de la base de datos
            item_text = f"{nombre} - Guardado el: {fecha_hora}"  # Agregar la fecha/hora al texto del elemento
            item = QListWidgetItem(item_text)
            self.list_widget.addItem(item)

    def eliminar_cara(self):
        if not self.list_widget.selectedItems():
            QMessageBox.warning(self, 'Error', 'Seleccione una cara para eliminar.')
            return

        respuesta = QMessageBox.question(self, 'Eliminar Cara', '¿Está seguro de eliminar esta cara?',
                                         QMessageBox.Yes | QMessageBox.No)
        if respuesta == QMessageBox.Yes:
            nombre = self.list_widget.currentItem().text().split('-')[0].strip()  # Obtener solo el nombre

            cursor.execute("DELETE FROM CaraPersonas WHERE nombre=?", (nombre,))
            conn.commit()

            self.list_widget.takeItem(self.list_widget.currentRow())
            QMessageBox.information(self, 'Cara Eliminada', 'La cara ha sido eliminada de la base de datos.')

    def mostrar_imagen_seleccionada(self, item):
        nombre = item.text().split('-')[0].strip()  # Obtener solo el nombre

        # Recuperar la imagen de la base de datos
        cursor.execute("SELECT imagen FROM CaraPersonas WHERE nombre=?", (nombre,))
        row = cursor.fetchone()

        if row:
            imagen_bytes = row[0]
            imagen_decodificada = cv2.imdecode(np.frombuffer(imagen_bytes, np.uint8), cv2.IMREAD_COLOR)

            # Mostrar la imagen en una ventana emergente
            cv2.imshow('Imagen Seleccionada', imagen_decodificada)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


class InterfazGrafica(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.captura_realizada = False  # Bandera para evitar múltiples registros

    def initUI(self):
        self.setWindowTitle('Administrador de Reconocimiento Facial')
        self.setGeometry(100, 100, 640, 480)

        # Botón para iniciar el reconocimiento facial
        self.btn_iniciar = QPushButton('Iniciar Reconocimiento Facial', self)
        self.btn_iniciar.clicked.connect(self.iniciar_reconocimiento)

        # Botón para detener el reconocimiento facial
        self.btn_detener = QPushButton('Detener Reconocimiento Facial', self)
        self.btn_detener.clicked.connect(self.detener_reconocimiento)

        # Botón para mostrar la lista de caras
        self.btn_mostrar_caras = QPushButton('Mostrar Caras Guardadas', self)
        self.btn_mostrar_caras.clicked.connect(self.mostrar_caras_guardadas)

        # Botón para mostrar la última imagen guardada
        self.btn_mostrar_ultima_imagen = QPushButton('Mostrar Última Imagen', self)
        self.btn_mostrar_ultima_imagen.clicked.connect(self.mostrar_ultima_imagen)

        # Etiqueta para mostrar la cámara
        self.label_camara = QLabel(self)

        # Etiqueta para mostrar el estado del reconocimiento
        self.label_estado = QLabel(self)
        self.label_estado.setAlignment(Qt.AlignCenter)

        # Campo de texto para mostrar el nombre detectado
        self.txt_nombre_detectado = QLineEdit(self)
        self.txt_nombre_detectado.setReadOnly(True)

        # Campo de texto para mostrar la fecha y hora del reconocimiento
        self.txt_fecha_hora = QLineEdit(self)
        self.txt_fecha_hora.setReadOnly(True)

        # Campo de texto para ingresar el nombre
        self.txt_nombre = QLineEdit(self)

        # Botón para guardar el nombre ingresado
        self.btn_guardar = QPushButton('Guardar Nombre', self)
        self.btn_guardar.clicked.connect(self.guardar_nombre)

       
        # Diseño de la interfaz
        layout = QVBoxLayout()
        layout.addWidget(self.btn_iniciar)
        layout.addWidget(self.btn_detener)
        layout.addWidget(self.btn_mostrar_caras)
        layout.addWidget(self.btn_mostrar_ultima_imagen)
        layout.addWidget(self.label_camara)
        layout.addWidget(self.label_estado)
        layout.addWidget(self.txt_nombre_detectado)
        layout.addWidget(self.txt_fecha_hora)
        layout.addWidget(self.txt_nombre)
        layout.addWidget(self.btn_guardar)
        
        self.setLayout(layout)

        # Inicializar la cámara
        self.cap = cv2.VideoCapture(0)
        self.timer = None

        # Inicializar el clasificador HaarCascade para detección de caras
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def iniciar_reconocimiento(self):
        # Iniciar la captura de imágenes
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.actualizar_camara)
        self.timer.start(100)  # Captura un fotograma cada 100 milisegundos

    def detener_reconocimiento(self):
        # Detener la captura de imágenes
        if self.timer is not None:
            self.timer.stop()

    def actualizar_camara(self):
        # Capturar un fotograma de la cámara
        ret, frame = self.cap.read()

        if ret:
            # Convertir el fotograma de OpenCV a un formato adecuado para mostrar en QLabel
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            pixmap = pixmap.scaled(self.label_camara.width(), self.label_camara.height(), aspectRatioMode=Qt.KeepAspectRatio)
            self.label_camara.setPixmap(pixmap)

            # Detección facial
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Verificar si las caras detectadas son conocidas o desconocidas
            for (x, y, w, h) in faces:
                roi_gray = gray_frame[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                match = self.comparar_cara(roi_gray)

                if match:
                    if not self.captura_realizada:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, "Conocido", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        nombre = self.obtener_nombre_conocido(roi_gray)
                        fecha_hora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                        self.txt_nombre_detectado.setText(nombre)
                        self.txt_fecha_hora.setText(fecha_hora)

                        self.registrar_paso(nombre, fecha_hora)

                        self.guardar_ultima_captura_conocida(roi_color, nombre)
            
                        self.captura_realizada = True
                else:
                    self.captura_realizada = False
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "Desconocido", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    self.txt_nombre_detectado.setText("Desconocido")

                    fecha_hora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    self.txt_fecha_hora.setText(fecha_hora)

                    self.registrar_paso("Desconocido", fecha_hora)

            cv2.imshow('Face Detection', frame)

    def obtener_nombre_conocido(self, cara):
        cursor.execute("SELECT nombre, imagen FROM CaraPersonas")
        rows = cursor.fetchall()

        for row in rows:
            nombre = row[0]
            imagen_bytes = row[1]
            imagen_decodificada = cv2.imdecode(np.frombuffer(imagen_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

            correlacion = cv2.matchTemplate(cara, imagen_decodificada, cv2.TM_CCOEFF_NORMED)

            if np.max(correlacion) > 0.8:
                return nombre

        return "Desconocido"

    def comparar_cara(self, cara):
        cursor.execute("SELECT * FROM CaraPersonas")
        rows = cursor.fetchall()

        for row in rows:
            imagen_bytes = row[2]
            imagen_decodificada = cv2.imdecode(np.frombuffer(imagen_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

            correlacion = cv2.matchTemplate(cara, imagen_decodificada, cv2.TM_CCOEFF_NORMED)

            if np.max(correlacion) > 0.8:
                return True

        return False

    def registrar_paso(self, nombre, fecha_hora):
        cursor.execute("INSERT INTO Pasos (nombre, fecha_hora) VALUES (?, ?)", (nombre, fecha_hora))
        conn.commit()

    def guardar_nombre(self):
        nombre = self.txt_nombre.text()
        print("Nombre ingresado:", nombre)

        ret, frame = self.cap.read()

        if ret:
            self.guardar_imagen_en_db(frame, nombre)
            QMessageBox.information(self, 'Registro Exitoso', 'Se ha registrado la cara correctamente.')

    def guardar_imagen_en_db(self, imagen, nombre):
        imagen_blob = cv2.imencode('.jpg', imagen)[1].tobytes()

        fecha_hora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute("INSERT INTO CaraPersonas (nombre, imagen, fecha_hora) VALUES (?, ?, ?)", (nombre, sqlite3.Binary(imagen_blob), fecha_hora))
        conn.commit()

    def mostrar_caras_guardadas(self):
        self.lista_caras = ListaCaras()
        self.lista_caras.exec_()

    def mostrar_ultima_imagen(self):
        cursor.execute("SELECT imagen FROM CaraPersonas ORDER BY ID DESC LIMIT 1")
        row = cursor.fetchone()

        if row:
            imagen_bytes = row[0]
            imagen_decodificada = cv2.imdecode(np.frombuffer(imagen_bytes, np.uint8), cv2.IMREAD_COLOR)

            cv2.imshow('Última Imagen Guardada', imagen_decodificada)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def guardar_ultima_captura_conocida(self, imagen, nombre):
        imagen_blob = cv2.imencode('.jpg', imagen)[1].tobytes()

        fecha_hora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute("INSERT INTO CaraPersonas (nombre, imagen, fecha_hora) VALUES (?, ?, ?)", (nombre, sqlite3.Binary(imagen_blob), fecha_hora))
        conn.commit()

    def ver_ultima_captura_conocida(self):
        cursor.execute("SELECT imagen FROM CaraPersonas ORDER BY ID DESC LIMIT 1")
        row = cursor.fetchone()

        if row:
            imagen_bytes = row[0]
            imagen_decodificada = cv2.imdecode(np.frombuffer(imagen_bytes, np.uint8), cv2.IMREAD_COLOR)

            cv2.imshow('Última Captura de Cara Conocida', imagen_decodificada)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            QMessageBox.warning(self, 'Aviso', 'No se ha detectado ninguna cara conocida durante esta ejecución del programa.')

   


    def closeEvent(self, event):
        # Cerrar la conexión a la base de datos al salir de la aplicación
        conn.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ventana = InterfazGrafica()
    ventana.show()
    sys.exit(app.exec_())
