import streamlit as st # ใช้สร้างเว็บแอป Python
import pandas as pd
import pickle # ใช้โหลดโมเดล Machine Learning

# โหลดโมเดลที่ฝึกแล้ว + Scaler
# ตรวจสอบให้แน่ใจว่ามีไฟล์ model.pkl อยู่ในโฟลเดอร์เดียวกับโค้ดนี้
try:
    model, scaler = pickle.load(open("model.pkl", "rb"))
except FileNotFoundError:
    st.error("ไม่พบไฟล์ model.pkl กรุณาตรวจสอบชื่อไฟล์")

# ส่วนตกแต่ง CSS (สีพื้นหลัง + ปุ่ม + เปลี่ยนสี Slider)
page_bg = """
<style>
/* เปลี่ยนสีพื้นหลังหน้าเว็บ */
.stApp {
    background: #CCFF99;
}

/* เปลี่ยนสีข้อความทั่วไป */
h1, h2, h3, p, span, label {
    color: #3a3a3a !important;
}

/* --- ส่วนสำคัญ: เปลี่ยนสี Slider --- */
/* เปลี่ยนสีแถบที่ลากแล้ว (Track) และจุดกลมๆ (Thumb) */
div[data-baseweb="slider"] > div > div {
    background-color: #FFFFFF !important; /* สีแถบที่เลือก */
}
div[role="slider"] {
    background-color: #990000 !important; /* สีปุ่มวงกลม */
    border: 2px solid #990000!important;
}
/* เปลี่ยนสีตัวเลขบน Slider */
div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"], [data-testid="stThumbValue"] {
    color: #3a3a3a !important;
}

/* ตกแต่งปุ่ม Predict */
div.stButton > button {
    background-color: #FF99CC;
    color: #CC0000;
    border-radius: 10px;
    border: 2px solid #000000;
    padding: 0.6rem 1rem;
    font-size: 1.1rem;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #ff8e80;
    color: white;
    border: 2px solid #ff8e80;
}

.box {
    background: rgba(255, 255, 255, 0.4);
    padding: 25px;
    border-radius: 15px;
    margin-top: 20px;
    border: 1px solid #003300;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# หัวข้อเว็บ
st.title("🎌 Anime Rating Prediction 🎌")
st.write("🩷 ทำนายความนิยมของอนิเมะ 🩷")

# Input Form
with st.container():
    st.markdown("<div class='box'>", unsafe_allow_html=True)
    st.subheader("🌼 Anime data 🌼")

    # ช่องกรอกข้อมูลผู้ใช้
    genre_count = st.slider("จำนวนประเภทของอนิเมะ (Genre Count) เช่น Action+Fantasy = 2", 1, 10, 3)
    episodes = st.slider("Number of episodes (จำนวนตอนของเรื่อง)", 1, 200, 12)
    studio_score = st.slider("Studio credibility score (คะแนนความน่าเชื่อถือของสตูดิโอ) (0-10)", 0.0, 10.0, 7.5)
    release_year = st.slider("Year of release (ปีที่ออกฉาย)", 1980, 2025, 2015)
    
    is_sequel = st.selectbox("Is there a sequel? (มีภาคต่อหรือไม่)", ["❌ ไม่มี (No)", "🌺 มี (Yes)"])
    
    # แปลงค่า Selectbox เป็นตัวเลข 0 หรือ 1
    # แก้ไข Logic: ค้นหาคำว่า "มี" ในข้อความที่เลือก
    is_sequel_num = 1 if "มี (Yes)" in is_sequel else 0

    st.markdown("</div>", unsafe_allow_html=True)

st.write("") # เว้นวรรค

# เมื่อผู้ใช้กดปุ่ม Predict
if st.button("🍎 Predict Rating (ทำนายคะแนน)"):
    # เตรียมข้อมูลสำหรับ Prediction
    input_data = pd.DataFrame([{
        "genre_count": genre_count,
        "episodes": episodes,
        "studio_score": studio_score,
        "release_year": release_year,
        "is_sequel": is_sequel_num
    }])

    # Scale ข้อมูลและทำนาย
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        st.markdown("<hr style='border: 1px solid #003300; margin: 20px 0;'>", unsafe_allow_html=True)
        st.subheader("🍀 Prediction results (ผลการทำนาย)")
        st.success(f"Predicted score (คะแนนที่คาดการณ์): **{prediction:.2f} / 10**")
        
        # แสดงคำแนะนำเพิ่มเติมตามคะแนน
        if prediction >= 8.0:
            st.balloons()
            st.info("⭐ อนิเมะเรื่องนี้มีแนวโน้มที่จะเป็นระดับ Masterpiece!")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการคำนวณ: {e}")
 