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
print("🔑 Loaded GROQ_API_KEY:", GROQ_API_KEY)

def generate_advice(prob):
    if not GROQ_API_KEY:
        return "AI 조언 생성이 활성화되지 않았습니다."

    prompt = f"""
    사용자의 뇌졸중 발병 확률은 {prob}% 입니다.
    한국 성인 기준 맞춤 건강 조언을 5줄 이내 한국어로 작성하세요.
    외국어, 이모지, 특수문자 금지.
    """

    try:
        r = requests.post(
            "https://api.groq.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6
            },
            timeout=12
        )

        ans = r.json()

        # 1) 정상 구조
        if "choices" in ans:
            msg = ans["choices"][0].get("message", {})
            content = msg.get("content")
            if content: 
                return content.strip()

        # 2) Stream 형태 fallback
        if "content" in ans:
            return ans["content"].strip()

        # 3) 실패
        return "AI 조언 생성 중 오류가 발생했습니다."

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
