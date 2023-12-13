import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# normal.csv 파일의 데이터를 읽어온다.
normal_df = pd.read_csv("normal.csv")

# abnormal.csv 파일의 데이터를 읽어온다.
abnormal_df = pd.read_csv("abnormal.csv")

# 두 데이터프레임을 병합한다.
df = pd.concat([normal_df, abnormal_df], ignore_index=True)

# motor_id를 기준으로 훈련 데이터와 테스트 데이터를 분할한다.
train_df = df[df["motor_id"].isin(range(1, 36))]
test_df = df[df["motor_id"].isin(range(37, 48))]

# 데이터를 학습 데이터로 분할한다.
X_train, y_train = train_df.drop(["class", "motor_id"], axis=1), train_df["class"]

# 데이터를 정규화한다.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Create DataFrame with scaled data and column names
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# GridSearchCV를 사용하여 랜덤 포레스트 모델의 하이퍼파라미터 튜닝을 수행한다.
param_grid = {
    'n_estimators': [500],
    'max_depth': [100],
    'min_samples_split': [10],
    'min_samples_leaf': [6],
}

rf_model = RandomForestClassifier()
grid_search = GridSearchCV(rf_model, param_grid=param_grid, scoring='roc_auc', cv=5)
grid_search.fit(X_train_scaled_df, y_train)

# GridSearchCV에서 최적의 모델을 얻는다.
best_rf_model = grid_search.best_estimator_

# 최적의 RandomForest 모델을 기반으로 AdaBoost 모델을 훈련한다.
ada_param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1, 0.2],
}
ada_model = AdaBoostClassifier(base_estimator=best_rf_model)
ada_grid_search = GridSearchCV(ada_model, param_grid=ada_param_grid, scoring='roc_auc', cv=5)
ada_grid_search.fit(X_train_scaled_df, y_train)

# 테스트 데이터를 정규화한다.
X_test, y_test = test_df.drop(["class", "motor_id"], axis=1), test_df["class"]
X_test_scaled = scaler.transform(X_test)

# Create DataFrame with scaled test data and column names
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# AdaBoost 모델을 평가한다.
ada_score = roc_auc_score(y_test, ada_grid_search.predict(X_test_scaled_df))

# 특성 중요도 추출
feature_importances = best_rf_model.feature_importances_

# 특성 중요도를 내림차순으로 정렬
indices = feature_importances.argsort()[::-1]

# 특성 중요도 시각화
plt.figure(figsize=(10, 6))
plt.bar(range(X_train_scaled_df.shape[1]), feature_importances[indices])
plt.xticks(range(X_train_scaled_df.shape[1]), X_train_scaled_df.columns[indices], rotation=45)
plt.title("Feature Importances")
plt.show()

# 결과를 출력한다.
print("Random Forest GridSearchCV를 통한 최적 하이퍼파라미터:", grid_search.best_params_)
print("AdaBoost GridSearchCV를 통한 최적 하이퍼파라미터:", ada_grid_search.best_params_)
print("AdaBoost ROC_AUC 점수:", ada_score)
