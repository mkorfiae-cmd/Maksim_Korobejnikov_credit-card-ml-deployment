# Импорты
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Конфигурация
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "target"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "credit_card_processed.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_V1_PATH = MODELS_DIR / "model_v1.joblib"
MODEL_V2_PATH = MODELS_DIR / "model_v2.joblib"

#Загрузка данных
def load_data(data_path: Path):
    if not data_path.exists():
        raise FileNotFoundError(f"Файл с данными не найден: {data_path}")
    data = pd.read_csv(data_path)
    print("Данные загружены.")
    print("Путь:", data_path)
    print("Размер датасета:", data.shape)
    return data

# Разделение на признаки и target
def split_features_target(data: pd.DataFrame):
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"В датасете отсутствует целевая переменная: {TARGET_COLUMN}")
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]
    print("Данные разделены на X и y.")
    print("Размер X:", X.shape)
    print("Размер y:", y.shape)
    return X, y

# Определение типов признаков
def get_feature_groups(X: pd.DataFrame):
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE",]
    categorical_features = [
        column for column in categorical_features
        if column in X.columns
    ]
    numeric_features = [
        column for column in X.columns
        if column not in categorical_features
    ]
    print("Группы признаков определены.")
    print("Числовые признаки:", len(numeric_features))
    print("Категориальные признаки:", categorical_features)
    return numeric_features, categorical_features


# preprocessing pipeline
def build_preprocessor(numeric_features: list[str], categorical_features: list[str],):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


# Модели
def build_model_v1(preprocessor: ColumnTransformer):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def build_model_v2(preprocessor: ColumnTransformer):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


# Оценка качества модели
def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str,):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }    
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Основные метрики:")
    print("F1-score:", round(metrics["f1"], 4))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall:", round(metrics["recall"], 4))
    print("ROC-AUC:", round(metrics["roc_auc"], 4))
    return metrics

# Сохранение модели
def save_model(model: Pipeline, model_path: Path):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    
    
    
  
def main():
    data = load_data(DATA_PATH)
    X, y = split_features_target(data)
    numeric_features, categorical_features = get_feature_groups(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,)   
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    preprocessor_v1 = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    preprocessor_v2 = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    model_v1 = build_model_v1(preprocessor_v1)
    model_v2 = build_model_v2(preprocessor_v2)
    print("Обучение model_v1...")
    model_v1.fit(X_train, y_train)
    print("Обучение model_v2...")
    model_v2.fit(X_train, y_train)
    metrics_v1 = evaluate_model(model=model_v1, X_test=X_test, y_test=y_test, model_name="model_v1 LogisticRegression")
    metrics_v2 = evaluate_model(model=model_v2, X_test=X_test, y_test=y_test, model_name="model_v2 RandomForestClassifier")
    save_model(model_v1, MODEL_V1_PATH)
    save_model(model_v2, MODEL_V2_PATH)

    print("Итоговое сравнение моделей")
    print("model_v1 LogisticRegression:")
    print(metrics_v1)
    print("model_v2 RandomForestClassifier:")
    print(metrics_v2)
    print("Обучение завершено успешно.")


if __name__ == "__main__":
    main()