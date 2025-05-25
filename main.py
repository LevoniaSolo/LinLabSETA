import numpy as np
import time

# СТАНДАРТНАЯ СИГМОИДА (АПРОКСИМИРОВАННАЯ НЕ ДОСТИГАЛА ТЕХ ЖЕ РЕЗУЛЬТАТОВ)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# функция потерь (кросс-энтропия)
def cross_entropy_loss(y_true, y_pred):
    m = len(y_true)
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)  # с учётом log(0)
    loss = - (1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# Генерация данных
def generate_data(m=1000000, n=30):
    np.random.seed(42)
    X = np.random.normal(0, 1, size=(m, n))
    true_w = np.random.normal(0, 0.5, size=n)
    true_b = 0.0

    z = np.dot(X, true_w) + true_b

    y = (z > 0).astype(int) # метки по знаку z (вместо сигмоиды как раньше)
    # Переворачивание меток для начальной точности ~60–70%
    flip_prob = 0.1
    flip_mask = np.random.rand(m) < flip_prob
    y[flip_mask] = 1 - y[flip_mask]
    return X, y

# разделение данных на обучающую и тестовую выборки с 80% на 20%
def train_test_split(X, y, test_size=0.2):
    m = X.shape[0]
    indices = np.random.permutation(m)
    test_size = int(test_size * m)
    train_idx, test_idx = indices[test_size:], indices[:test_size]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test

# точность
def accuracy(y_true, y_pred):
    return np.mean(y_true == (y_pred > 0.5).astype(int))

# обучение
def train_perceptron(X_train, y_train, X_test, y_test, n_epochs=100, alpha=0.1):
    m, n = X_train.shape
    w = np.random.randn(n) * 0.1
    b = 0.0
    history = {'loss': [], 'accuracy': []}

    for epoch in range(n_epochs):
        # прямое распространение
        z = np.dot(X_train, w) + b
        y_pred = sigmoid(z)
        
        loss = cross_entropy_loss(y_train, y_pred)
        
        # градиенты
        error = y_pred - y_train
        dw = (1/m) * np.dot(X_train.T, error)
        db = (1/m) * np.sum(error)
        
        # новые веса и смещения
        w -= alpha * dw
        b -= alpha * db
        
        # оценка на тестовой выборке
        z_test = np.dot(X_test, w) + b
        y_pred_test = sigmoid(z_test)
        acc = accuracy(y_test, y_pred_test)
        
        history['loss'].append(loss)
        history['accuracy'].append(acc)
        
        if epoch % 10 == 0:
            print(f"Эпоха {epoch}: Потери = {loss:.4f}, Точность = {acc:.4f}")
    
    return w, b, history

def main():
    start_time = time.time()
    
    print("Генерация данных...")
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    print("Обучение перцептрона...")
    w, b, history = train_perceptron(X_train, y_train, X_test, y_test)
    
    z_test = np.dot(X_test, w) + b
    y_pred_test = sigmoid(z_test)
    final_accuracy = accuracy(y_test, y_pred_test)
    
    print(f"\nФинальная точность на тестовой выборке: {final_accuracy * 100:.2f}%")
    print(f"Время выполнения: {time.time() - start_time:.2f} секунд")
    
    if final_accuracy >= 0.85:
        print("Требуемая точность (>= 85%) достигнута!")
    else:
        print("Требуемая точность (>= 85%) не достигнута.")

if __name__ == "__main__":
    main()
