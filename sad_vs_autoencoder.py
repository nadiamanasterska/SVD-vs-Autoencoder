import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense
from skimage import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
from tensorflow.keras.optimizers import Adam

# Załaduj dane MNIST
(x_train_data, y_train_data), (x_test_data, y_test_data) = mnist.load_data()

# Normalizacja pikseli obrazów do zakresu [0, 1], co przyspiesza i stabilizuje proces uczenia.
x_train_data = x_train_data.astype('float32') / 255.
x_test_data = x_test_data.astype('float32') / 255.

## Spłaszczenie obrazów 28x28 do jednowymiarowych wektorów o długości 784 (28*28), aby pasowały do modeli.
x_train_data = x_train_data.reshape((len(x_train_data), np.prod(x_train_data.shape[1:])))
x_test_data = x_test_data.reshape((len(x_test_data), np.prod(x_test_data.shape[1:])))

# --- SVD ---
# Zastosowanie TruncatedSVD do redukcji wymiarowości; Redukcja do 64 wymiarów przy zachowaniu maksymalnej ilości informacji.
svd_model = TruncatedSVD(n_components=64)
x_train_svd = svd_model.fit_transform(x_train_data)
x_test_svd = svd_model.transform(x_test_data)

# Trenowanie regresji logistycznej na danych zredukowanych za pomocą SVD.
# - max_iter=5000: większa liczba iteracji dla lepszej konwergencji.
# - Przewidywanie etykiet dla danych testowych.
svd_classifier = LogisticRegression(max_iter=5000)
svd_classifier.fit(x_train_svd, y_train_data)
y_pred_svd = svd_classifier.predict(x_test_svd)

svd_accuracy = accuracy_score(y_test_data, y_pred_svd)
print(f"Accuracy using SVD: {svd_accuracy:.4f}")

# --- Autoencoder ---
# Definicja części autoenkodera odpowiedzialnej za kodowanie:
# - Warstwa wejściowa: wektor o długości 784,
# - Warstwa ukryta: redukcja do 64 wymiarów z funkcją aktywacji ReLU.
input_layer = Input(shape=(784,))
encoded_layer = Dense(64, activation='relu')(input_layer)
encoder_model = Model(input_layer, encoded_layer)

# Definicja części autoenkodera odpowiedzialnej za dekodowanie:
# - Warstwa wejściowa: wektor o długości 64,
# - Warstwa wyjściowa: rekonstrukcja oryginalnego obrazu (784 wartości) z funkcją sigmoidalną.
decoder_input = Input(shape=(64,))
decoded_layer = Dense(784, activation='sigmoid')(decoder_input)
decoder_model = Model(decoder_input, decoded_layer)

# Łączenie enkodera i dekodera w pełny autoenkoder.
autoencoder_model = Model(input_layer, decoder_model(encoder_model(input_layer)))

# Kompilowanie autoenkodera:
# - Optymalizator Adam: szybka i efektywna optymalizacja,
# - Funkcja straty: średni błąd kwadratowy (rekonstrukcja obrazów).

custom_optimizer = Adam(learning_rate=0.001)

autoencoder_model.compile(optimizer=custom_optimizer, loss='mean_squared_error')

# Listy do przechowywania wartości strat
train_losses = []
validation_losses = []

epoch_count = 45

# Pętla treningowa, która zbiera straty dla treningu i walidacji przez 25 epok
for epoch_num in range(epoch_count):
    start_time = time.time()  # Rozpoczęcie liczenia czasu epoki

    # Trenowanie modelu przez jedną epokę
    epoch_history = autoencoder_model.fit(x_train_data, x_train_data,
                                           batch_size=512,
                                           shuffle=True,
                                           validation_data=(x_test_data, x_test_data),
                                           epochs=1,
                                           verbose=0)  # Wyłącz domyślną wypowiedź TensorFlow

    # Zbieranie strat dla danych treningowych i walidacyjnych
    train_loss = epoch_history.history['loss'][0]
    val_loss = epoch_history.history['val_loss'][0]
    train_losses.append(train_loss)
    validation_losses.append(val_loss)

    # Pomiar czasu trwania epoki
    epoch_time = time.time() - start_time  # Czas trwania epoki

    # Wyświetlanie postępu
    print(f"Epoch {epoch_num + 1}/{epoch_count}  time: {epoch_time:.2f}s  training loss: {train_loss:.4f}  test loss: {val_loss:.4f}")

# Pobranie zakodowanej reprezentacji danych testowych
encoded_train_data = encoder_model.predict(x_train_data)
encoded_test_data = encoder_model.predict(x_test_data)

# Trenowanie klasyfikatora regresji logistycznej na zakodowanych danych
autoencoder_classifier = LogisticRegression(max_iter=5000)
autoencoder_classifier.fit(encoded_train_data, y_train_data)
y_pred_autoencoder = autoencoder_classifier.predict(encoded_test_data)

autoencoder_accuracy = accuracy_score(y_test_data, y_pred_autoencoder)
print(f"Accuracy using Autoencoder: {autoencoder_accuracy:.4f}")

# --- Wykresy ---
# Wykres strat dla autoenkodera
plt.plot(train_losses, label="Training loss")
plt.plot(validation_losses, label="Validation loss")
plt.title('Training and Validation Loss of Autoencoder')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Porównanie dokładności SVD i autoenkodera
print(f"Accuracy comparison: \nSVD: {svd_accuracy:.4f}\nAutoencoder: {autoencoder_accuracy:.4f}")

# --- Wyświetlanie błędnie sklasyfikowanych przykładów ---
# Porównanie wyników
wrong_predictions = np.where(y_pred_autoencoder != y_test_data)[0]

# Liczba przykładów, które chcemy wyświetlić (np. 5 pierwszych błędów)
num_wrong_examples = 5

# Tworzenie wykresów dla błędnie sklasyfikowanych obrazów
for i in range(min(num_wrong_examples, len(wrong_predictions))):
    # Indeks błędnie sklasyfikowanego obrazu
    idx = wrong_predictions[i]

    # Pobranie obrazu
    img = x_test_data[idx]

    # Pobranie prawdziwej i przewidywanej etykiety
    true_label = y_test_data[idx]
    predicted_label = y_pred_autoencoder[idx]

    # Wyświetlenie obrazu
    plt.figure(figsize=(3, 3))
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title(f"True Label: {true_label}, Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()
