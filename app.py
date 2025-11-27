from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import requests

app = Flask(__name__)

# ------------------------------------------------
# 1) 모델 로드
# ------------------------------------------------
print("🔄 Loading stroke_model.pkl ...")
raw = joblib.load("stroke_model.pkl")

if isinstance(raw, dict) and "model" in raw:
    model = raw["model"]
else:
    model = raw

print("📌 Loaded object type:", type(raw))
print("📌 Final model type:", type(model))
print("📌 Keys:", raw.keys() if isinstance(raw, dict) else "none")

print("🔄 Loading scaler.pkl / kmeans.pkl ...")
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans.pkl")
print("✔ scaler / kmeans 로드 완료")

# 학습된 LightGBM 최적 threshold
THRESHOLD = 0.0297

# ------------------------------------------------
# 2) GROQ API 설정
# ------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("🔑 Loaded GROQ_API_KEY:", GROQ_API_KEY)


# ------------------------------------------------
# 3) LLM 조언 생성 함수
# ------------------------------------------------
def generate_advice(prob, user_info):
    if not GROQ_API_KEY:
        return "⚠️ AI 조언을 불러올 수 없습니다."

    prompt = f"""
    사용자의 건강 정보를 기반으로 의료 전문가의 관점에서 생활습관 개선 조언을 작성하세요.
    외국어 금지.
    한자 금지.
    기본 문장 부호(.,!? ) 외 특수기호 금지.
    지나치게 어려운 표현 금지.
    성별, 만 나이, BMI, 혈압, 혈당, 흡연, 음주 정보를 반영할 것.
    의사가 직접 설명해주는 말투로 5줄 이내로 작성하세요.
    
    사용자 정보:
    성별: {gender}
    나이: {age}
    BMI: {bmi}
    수축기 혈압: {sbp}
    이완기 혈압: {dbp}
    공복 혈당: {glucose}
    흡연 여부: {smoking}
    음주 빈도: {drinking}
    """


    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-8b-8192",   # ← 여기를 mixtral → llama 로 변경
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7
            }
        )

        res = r.json()

        if "choices" not in res:
            print("🔥 RAW LLM 응답:", res)
            return "AI 조언 생성 중 오류가 발생했습니다."

        return res["choices"][0]["message"]["content"]

    except Exception as e:
        print("❌ LLM 요청 실패:", e)
        return "AI 조언 생성 중 오류가 발생했습니다."

# ------------------------------------------------
# 4) Routing
# ------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        gender    = float(data["gender"])
        age       = float(data["age"])
        bmi       = float(data["bmi"])
        sbp       = float(data["sbp"])
        dbp       = float(data["dbp"])
        glucose   = float(data["glucose"])
        smoking   = float(data["smoking"])
        drinking  = float(data["drinking"])

        # ---- 추가 3개 Feature 계산 ----
        hypertension = 1 if sbp >= 140 else 0
        diabetes = 1 if glucose >= 126 else 0
        exercise = 0  # NHANES 모델과 동일하게 고정

        # ---- cluster 계산 ----
        arr12 = np.array([[age, gender, bmi, sbp, dbp, glucose,
                           smoking, drinking, hypertension, diabetes, exercise]])

        scaled = scaler.transform(arr12)
        cluster_value = int(kmeans.predict(scaled)[0])

        # ---- 최종 12 features + cluster = 13개 ----
        X = np.array([[gender, age, bmi, sbp, dbp, glucose,
                       smoking, drinking, hypertension, diabetes, exercise,
                       cluster_value]])

        proba = model.predict_proba(X)[0][1]
        prob_percent = round(proba * 100, 1)

        # ---- 위험군 분류 ----
        risk_text  = "고위험" if proba >= THRESHOLD else "저위험"
        risk_class = "result-high" if proba >= THRESHOLD else "result-low"

        # ---- LLM 조언 ----
        user_info = {
            "gender": gender,
            "age": age,
            "bmi": bmi,
            "sbp": sbp,
            "dbp": dbp,
            "glucose": glucose,
            "smoking": smoking,
            "drinking": drinking
        }

        advice = generate_advice(prob_percent, user_info)

        return jsonify({
            "prob": prob_percent,
            "risk_text": risk_text,
            "risk_class": risk_class,
            "advice": advice
        })

    except Exception as e:
        print("❌ 예측 오류:", e)
        return jsonify({"error": f"서버 오류: {str(e)}"})


if __name__ == "__main__":
    pass
