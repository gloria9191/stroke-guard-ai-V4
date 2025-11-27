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

# 🔥 너 모델의 실제 최적 threshold = 0.66
THRESHOLD = 0.66


# ------------------------------------------------
# 2) GROQ API 설정
# ------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_advice(prob, user_info):
    if not GROQ_API_KEY:
        return "AI 조언 생성이 활성화되지 않았습니다."

    # 사용자 특성 반영 조언
    prompt = f"""
    아래 사용자의 건강 정보를 바탕으로 한국인 기준 뇌졸중 예방 조언을 6줄 이내 한국어 문장으로 작성하세요.
    절대 외국어와 이모지 금지.

    [사용자 정보]
    - 성별: {user_info['gender']}
    - 만나이: {user_info['age']}세
    - BMI: {user_info['bmi']}
    - 수축기혈압: {user_info['sbp']}
    - 이완기혈압: {user_info['dbp']}
    - 공복혈당: {user_info['glucose']}
    - 흡연 여부: {user_info['smoking']}
    - 음주(주 1회 이상): {user_info['drinking']}
    - 예측된 뇌졸중 위험도: {prob}%

    [조언 조건]
    - 혈압 관리, 혈당 조절, 금연/절주, 운동, 위험 신호 체크 중심
    - 사용자 수치에 따라 맞춤형 조언 포함
    - 의료적 맥락 유지
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

        # 🔥 모델 기준 위험군 정의 (Threshold = 0.66)
        if proba >= THRESHOLD:
            risk_text  = "고위험"
            risk_class = "result-high"
        else:
            risk_text  = "저위험"
            risk_class = "result-low"

        # 사용자 정보 텍스트로 전달하여 맞춤형 조언 강화
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
        return jsonify({"error": f"서버 오류: {str(e)}"})


# ------------------------------------------------
# Render: run() 없음
# ------------------------------------------------
if __name__ == "__main__":
    pass
