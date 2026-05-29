import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import platform  # OS 확인을 위한 라이브러리 추가

# 폰트 매핑용 사전 (그래프 축 이름을 한글로 바꾸기 위함)
NAME_MAPPING = {
    'Age': '나이',
    'Smokes': '흡연량',
    'Alkhol': '음주량',
    'AreaQ': '지역',
    'Result': '결과'
}

# 🛠️ Matplotlib 한글 폰트 설정
def set_matplot_font():
    os_name = platform.system()
    if os_name == "Windows":
        plt.rc('font', family='Malgun Gothic')
    elif os_name == "Darwin":  # Mac
        plt.rc('font', family='AppleGothic')
    else:  # 리눅스 (스트림릿 클라우드 배포 등)
        plt.rc('font', family='NanumGothic')
    
    # 마이너스 기호 깨짐 방지
    plt.rc('axes', unicode_minus=False)

set_matplot_font()

# 1. 웹 페이지 기본 설정
st.set_page_config(page_title="환자 군집 예측", layout="centered")

st.title("🩺 환자 군집 예측 시스템")
st.write("데이터와 모델을 연결하고 있습니다. 잠시만 기다려주세요...")

# 2. 필수 파일 존재 여부 확인
required_files = ["lung_model.pkl", "scaler.pkl", "lung.csv"]
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error(f"⚠️ 폴더 내에 다음 필수 파일이 없습니다: {missing_files}")
else:
    try:
        # 3. 모델 및 데이터 불러오기
        model = joblib.load("lung_model.pkl")
        scaler = joblib.load("scaler.pkl")
        df = pd.read_csv("lung.csv")
        
        # 4. 입력 창으로 만들 숫자형 컬럼 필터링 (군집 결과 제외)
        feature_cols = []
        for col in df.columns:
            if 'cluster' not in col.lower():
                if pd.api.types.is_numeric_dtype(df[col]):
                    feature_cols.append(col)
        
        # 5. 사이드바에 환자 정보 입력 슬라이더 생성
        st.sidebar.header("👤 환자 정보 입력")
        user_inputs = {}
        
        for col in feature_cols:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            
            # 슬라이더 이름도 한글로 표시되도록 변경
            display_name = NAME_MAPPING.get(col, col)
            
            if df[col].dtype == 'int64':
                user_inputs[col] = st.sidebar.slider(f"{display_name}", int(min_val), int(max_val), int(mean_val))
            else:
                user_inputs[col] = st.sidebar.slider(f"{display_name}", min_val, max_val, mean_val)
        
        # 6. 예측 버튼 클릭 시 작동
        if st.sidebar.button("🎯 군집 예측하기"):
            input_df = pd.DataFrame([user_inputs])
            
            # [영어 -> 한글 컬럼명 자동 번역 변경]
            input_df = input_df.rename(columns=NAME_MAPPING)
            
            # [모델이 학습했던 정확한 컬럼 순서대로 정렬]
            if hasattr(scaler, 'feature_names_in_'):
                expected_cols = scaler.feature_names_in_
                input_df = input_df[expected_cols]
            
            # 데이터 변환 및 예측 수행
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]
            
            # 결과 화면 출력
            st.subheader("📊 예측 결과")
            st.success(f"입력하신 환자의 예측 군집은 **{prediction}번 군집**입니다.")
            
            # 7. 데이터 시각화 그래프 그리기
            if len(feature_cols) >= 2:
                st.subheader("📈 데이터 분포 내 환자 위치")
                fig, ax = plt.subplots(figsize=(6, 4))
                
                x_col = feature_cols[0]
                y_col = feature_cols[1]
                
                cluster_col = [c for c in df.columns if 'cluster' in c.lower()]
                if cluster_col:
                    scatter = ax.scatter(df[x_col], df[y_col], c=df[cluster_col[0]], cmap='viridis', alpha=0.4)
                    ax.legend(*scatter.legend_elements(), title="군집")
                else:
                    ax.scatter(df[x_col], df[y_col], alpha=0.4)
                
                # 새 환자의 위치를 빨간색 별표(*)로 표시
                ax.scatter(user_inputs[x_col], user_inputs[y_col], c='red', marker='*', s=250, label='신규 환자')
                
                # 💡 축 이름을 한글 매핑 사전을 이용하여 한글로 변경
                ax.set_xlabel(NAME_MAPPING.get(x_col, x_col))
                ax.set_ylabel(NAME_MAPPING.get(y_col, y_col))
                ax.legend()
                st.pyplot(fig)
                
    except Exception as e:
        st.error(f"🚨 시스템 구동 중 오류가 발생했습니다:\n`{e}`")