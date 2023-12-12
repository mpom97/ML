import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback, EarlyStopping
import matplotlib.pyplot as plt

# 데이터 로드 및 전처리
normal_df = pd.read_csv("normal.csv")
abnormal_df = pd.read_csv("abnormal.csv")
df = pd.concat([normal_df, abnormal_df], ignore_index=True)

# 훈련 데이터와 테스트 데이터 분할
train_df = df[df['motor_id'].between(1, 36)]
test_df = df[df['motor_id'].between(37, 48)]
X_train = train_df.drop(['class', 'motor_id'], axis=1)
y_train = train_df['class']
X_test = test_df.drop(['class', 'motor_id'], axis=1)
y_test = test_df['class']

# 모델 정의
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),  # 드롭아웃 추가
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),  # 드롭아웃 추가
    layers.Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ROC AUC를 계산하는 커스텀 콜백 정의
class RocAucCallback(Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.roc_aucs = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.x_val)
        roc_auc = roc_auc_score(self.y_val, y_pred)
        self.roc_aucs.append(roc_auc)
        print(f' - ROC AUC: {roc_auc:.4f}')

# 조기 종료 콜백 초기화
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# ROC AUC 콜백 초기화
roc_auc_callback = RocAucCallback(training_data=(X_train, y_train), validation_data=(X_test, y_test))

# 모델 학습
history = model.fit(
    X_train,
    y_train,
    epochs=450,
    batch_size=15,
    verbose=1,
    validation_data=(X_test, y_test),
    callbacks=[roc_auc_callback, early_stopping]
)

# 최대 ROC AUC 값과 해당 epoch 찾기
max_auc_index = np.argmax(roc_auc_callback.roc_aucs)
max_auc_value = roc_auc_callback.roc_aucs[max_auc_index]
print(f"최대 ROC AUC는 {max_auc_value:.4f}이며, 해당하는 epoch는 {max_auc_index+1}입니다.")

# 학습 과정 시각화
plt.figure(figsize=(12, 5))

# 손실 시각화
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# ROC AUC 시각화
plt.subplot(1, 2, 2)
epochs = len(roc_auc_callback.roc_aucs)
plt.plot(range(1, epochs + 1), roc_auc_callback.roc_aucs, label='ROC AUC', color='orange')
plt.title('ROC AUC')
plt.xlabel('Epoch')
plt.ylabel('ROC AUC')
plt.legend()

plt.tight_layout()
plt.show()