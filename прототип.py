import numpy as np
from catboost import CatBoostClassifier

def get_candidate_score(resume_text, financial_data):
    # 1. Текстовый анализ (Имитация BERT)
    # Превращаем текст резюме в вектор смыслов
    text_vector = model_bert.encode(resume_text) 
    
    # 2. Объединение данных
    # Конкатенируем текст с данными из БКИ и транзакций
    full_features = np.hstack([text_vector, financial_data])
    
    # 3. Предсказание (CatBoost)
    probability = model_decision.predict_proba(full_features)[0][1]
    return round(probability * 100, 2)

# Пример: Скоринг кандидата с доходом 150к и кредитным рейтингом 750
score = get_candidate_score("Опыт 5 лет на Python...", [5, 750, 150000])
print(f"Рейтинг кандидата: {score}/100")
