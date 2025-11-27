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

def generate_advice(prob, user_info):
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY 없음")
        return "AI 조언 생성이 활성화되지 않았습니다."
    user_text = f"""
    성별: { '남성' if user_info['gender']==1 else '여성' }
    나이: {user_info['age']}세
    BMI: {user_info['bmi']}
    수축기 혈압: {user_info['sbp']}
    이완기 혈압: {user_info['dbp']}
    공복 혈당: {user_info['glucose']}
    흡연 여부: {'예' if user_info['smoking']==1 else '아니오'}
    음주 여부: {'예' if user_info['drinking']==1 else '아니오'}
    """
    prompt = f"""
    사용자의 뇌졸중 발병 확률은 {prob}% 입니다.

    아래는 사용자의 건강 정보입니다:
    - 성별: {"남성" if user_info['gender']==1 else "여성"}
    - 나이: {user_info['age']}세
    - BMI: {user_info['bmi']}
    - 수축기 혈압(sbp): {user_info['sbp']}
    - 이완기 혈압(dbp): {user_info['dbp']}
    - 공복 혈당(glucose): {user_info['glucose']} mg/dL
    - 흡연 여부: {"흡연" if user_info['smoking']==1 else "비흡연"}
    - 음주 여부: {"음주" if user_info['drinking']==1 else "비음주"}

    위 정보를 종합해,
    한국 성인 기준 건강관리 조언을 5줄 이내 한국어 문장만으로 작성하세요.

    반드시 다음 원칙을 지킬 것:
    - 외국어, 이모지, 특수문자 금지
    - 너무 원론적인 말 금지
    - 입력된 수치(BMI, 혈압, 혈당)에 근거한 개인 맞춤형 조언 포함
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
            timeout=30
        )

        ans = r.json()
        print("🔥 RAW LLM 응답:", ans)

        if "choices" not in ans:
            return "AI 조언 생성 중 오류가 발생했습니다."

        return ans["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print("❌ LLM 요청 실패:", e)
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
        print("❌ LLM 요청 실패:", e)
        return jsonify({"error": f"서버 오류: {str(e)}"})


# ------------------------------------------------
# Render: run() 없음
# ------------------------------------------------
if __name__ == "__main__":
    pass
