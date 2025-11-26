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
model = joblib.load("stroke_model.pkl")
print("✅ 모델 로드 완료")

# ❗ 최종 검증 결과 기준
THRESHOLD = 0.66     # Recall(1)=0.81 기준 최적 threshold

# ------------------------------------------------
# 2) GROQ API 설정
# ------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_advice(prob, age, bmi, sbp, dbp, glucose, smoking, drinking):
    """
    사용자 입력 기반으로 전문적이고 현실적인 조언 생성.
    한국어 ONLY + 외국어/기호 금지.
    """
    if not GROQ_API_KEY:
        return "AI 조언 생성이 활성화되지 않았습니다."

    prompt = f"""
    아래는 한국 성인의 건강검진 데이터를 입력한 사용자입니다.
    이 사용자의 특성을 반영해 뇌졸중 예방을 위한 전문적 생활조언을 6줄 이내 한국어로만 작성하세요.
    외국어, 이모지, 특수문자(*, !, ?, 영어문장)는 절대 금지합니다.

    사용자 특성:
    - 연령(만나이): {age}세
    - BMI: {bmi}
    - 수축기혈압(SBP): {sbp}
    - 이완기혈압(DBP): {dbp}
    - 공복혈당: {glucose}
    - 흡연 여부: {smoking}
    - 음주: {drinking}  (기준: 주 1회 이상을 음주자로 간주)
    - 뇌졸중 예측 확률: {prob}%

    포함해야 할 내용:
    - 위험 요인(혈압·혈당·비만·흡연·음주) 중 어떤 항목이 높은지 구체적으로 언급
    - 생활에서 즉시 개선할 점
    - 주의해야 할 뇌졸중 전조증상
    - 병원 검진 필요성이 있는지 여부
    - 한국 성인 기준 의학적 권고 수준으로 간결하게 작성
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
    return render_template("index.html")   # index.html에 “아래로 스크롤하세요 ↓” 문구 추가해야 함!


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

        X = np.array([[gender, age, bmi, sbp, dbp, glucose, smoking, drinking]])
        proba = model.predict_proba(X)[0][1]
        prob_percent = round(proba * 100, 1)

        # ---------------------------------------------------
        # 🔥 정상 / 위험 / 고위험 기준
        # ---------------------------------------------------
        # 0.66 이상 → 고위험
        # 0.40 ~ 0.65 → 위험군 (중위험)
        # < 0.40 → 정상군
        # ---------------------------------------------------

        if proba >= 0.66:
            risk_class = "result-high"
            risk_text = "고위험"
        elif proba >= 0.40:
            risk_class = "result-mid"
            risk_text = "위험"
        else:
            risk_class = "result-low"
            risk_text = "정상"

        # ---------------------------------------------------
        #  AI 조언 생성
        # ---------------------------------------------------
        advice = generate_advice(
            prob_percent, age, bmi, sbp, dbp, glucose, smoking, drinking
        )

        return jsonify({
            "prob": prob_percent,
            "risk_text": risk_text,
            "risk_class": risk_class,
            "advice": advice,
            "threshold": THRESHOLD
        })

    except Exception as e:
        return jsonify({"error": f"서버 오류: {str(e)}"})


# ------------------------------------------------
# Render: run() 사용 금지
# ------------------------------------------------
if __name__ == "__main__":
    pass
