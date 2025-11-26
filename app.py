from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os
import requests

app = Flask(__name__)

# ------------------------------------------------
# 1) 모델 로드 (+ feature 리스트 포함 버전)
# ------------------------------------------------
print("🔄 Loading stroke_model.pkl ...")
bundle = joblib.load("stroke_model.pkl")   # 🔥 기존 model = joblib.load(...) 삭제
model = bundle["model"]
FEATURES = bundle["features"]
print("✅ 모델 로드 완료")
print("📌 Loaded FEATURES:", FEATURES)

THRESHOLD = 0.029698   # 기존 threshold 그대로 유지

# ------------------------------------------------
# 2) GROQ API 설정
# ------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_advice(prob):
    if not GROQ_API_KEY:
        return "AI 조언 생성이 활성화되지 않았습니다."

    prompt = f"""
    사용자의 뇌졸중 발병 확률은 {prob}% 입니다.

    한국 성인 기준으로 다음 항목을 중심으로,
    - 식습관
    - 운동
    - 혈압·혈당 관리
    - 위험 신호 체크
    - 금연/절주

    5줄 이내 한국어 문장으로만 작성하세요.
    절대로 외국어, *, 이모지, 일본어·중국어 등은 포함하지 마세요.
    """

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6
            },
            timeout=15
        )
        ans = r.json()
        return ans["choices"][0]["message"]["content"].strip()
    except Exception:
        return "AI 조언 생성 중 오류가 발생했습니다."


# ------------------------------------------------
# 3) 라우팅
# ------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # ------------------------------------------------
        # 🔥 Flask 입력 → FEATURE 리스트에 맞게 매핑
        # ------------------------------------------------
        user_input = {
            "Age": float(data["age"]),
            "Sex": float(data["gender"]),
            "BMI": float(data["bmi"]),
            "SBP_mean": float(data["sbp"]),
            "DBP_mean": float(data["dbp"]),
            "Glucose": float(data["glucose"]),
            "Smoking": float(data["smoking"]),
            "Alcohol": float(data["drinking"]),

            # 모델에서 필요한 추가 feature는 기본값(0 or mean)으로 채움
            "Hypertension": 0,
            "Diabetes": 0,
            "Exercise": 0,
            "cluster": 0,
        }

        # FEATURES 순서대로 DataFrame 생성
        X = pd.DataFrame([[user_input[f] for f in FEATURES]], columns=FEATURES)

        # 예측
        proba = model.predict_proba(X)[0][1]
        prob_percent = round(proba * 100, 1)

        # 위험도 분류
        risk_class = "result-low"
        risk_text  = "저위험"
        if proba >= THRESHOLD:
            risk_class = "result-high"
            risk_text  = "고위험"

        # AI 조언 생성
        advice = generate_advice(prob_percent)

        return jsonify({
            "prob": prob_percent,
            "risk_text": risk_text,
            "risk_class": risk_class,
            "advice": advice
        })

    except Exception as e:
        return jsonify({"error": f"서버 오류: {str(e)}"})


# ------------------------------------------------
# Render: run() 절대 실행 X
# ------------------------------------------------
if __name__ == "__main__":
    pass
