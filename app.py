import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    pass

st.set_page_config(
    page_title="서울랜드 키즈 멤버십",
    page_icon="🎡",
    layout="centered"
)

# 모바일 최적화 CSS
st.markdown("""
<style>
    .block-container { padding: 1.5rem 1rem 3rem 1rem; max-width: 480px; }
    .stButton > button {
        width: 100%; height: 3rem; font-size: 1rem;
        background-color: #C8101E; color: white;
        border: none; border-radius: 8px;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 0.5rem !important;
    }
    div[data-testid="column"] {
        min-width: 0 !important;
        flex: 1 !important;
    }
    div[data-testid="column"] > div {
        width: 100% !important;
    }
    div[data-testid="column"] button {
        width: 100% !important;
        font-size: 0.85rem !important;
    }
    .stButton > button:hover { background-color: #A00D18; }
    h1 { font-size: 1.5rem !important; color: #C8101E !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1rem !important; }
    .stSelectbox, .stMultiselect { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'page' not in st.session_state:
    st.session_state.page = 'membership'
if 'membership' not in st.session_state:
    st.session_state.membership = None
if 'child_age' not in st.session_state:
    st.session_state.child_age = 5
if 'child_type' not in st.session_state:
    st.session_state.child_type = []
if 'visit_purpose' not in st.session_state:
    st.session_state.visit_purpose = None
if 'visit_month' not in st.session_state:
    st.session_state.visit_month = 5
if 'visit_weekday' not in st.session_state:
    st.session_state.visit_weekday = '토'
if 'visit_weather' not in st.session_state:
    st.session_state.visit_weather = '맑음'
if 'min_temp' not in st.session_state:
    st.session_state.min_temp = 10
if 'max_temp' not in st.session_state:
    st.session_state.max_temp = 25
if 'predicted_visitors' not in st.session_state:
    st.session_state.predicted_visitors = None
if 'recommended_time' not in st.session_state:
    st.session_state.recommended_time = None
if 'recommended_course' not in st.session_state:
    st.session_state.recommended_course = None

# 데이터 로드 및 모델 학습
@st.cache_data
def load_and_train():
    months = ['1월','2월','3월','4월','5월','6월',
              '7월','8월','9월','10월','11월','12월']
    dfs = []
    for year, filepath in [(2023,'일별입장객및기상조건(23년)_260116.xlsx'),
                           (2024,'일별입장객및기상조건(24년)_260116.xlsx'),
                           (2025,'일별입장객및기상조건(25년)_260116.xlsx')]:
        for month in months:
            try:
                df = pd.read_excel(filepath, sheet_name=month, header=None)
                df = df.iloc[4:].reset_index(drop=True)
                df.columns = ['일','요일','날씨','최저기온','최고기온','개장','폐장','매표입장객','실입장객']
                df = df[pd.to_numeric(df['일'], errors='coerce').notna()].copy()
                df['연도'] = year
                df['월'] = int(month.replace('월',''))
                dfs.append(df)
            except:
                pass
    df = pd.concat(dfs, ignore_index=True)
    for col in ['일','최저기온','최고기온','매표입장객','실입장객']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    def simplify_weather(w):
        w = str(w)
        if '눈' in w: return '눈'
        elif '비' in w or '우' in w: return '비/우천'
        elif '흐림' in w or '구름많음' in w: return '흐림'
        elif '구름조금' in w: return '구름조금'
        elif '맑음' in w: return '맑음'
        else: return '기타'
    df['날씨분류'] = df['날씨'].apply(simplify_weather)
    weekday_map = {'월':0,'화':1,'수':2,'목':3,'금':4,'토':5,'일':6}
    df['요일코드'] = df['요일'].map(weekday_map)
    df['주말여부'] = df['요일코드'].apply(lambda x: '주말' if x in [5,6] else '평일')
    weather_score = {'맑음':5,'구름조금':4,'흐림':3,'비/우천':2,'눈':1,'기타':3}
    df['날씨점수'] = df['날씨분류'].map(weather_score)
    df['주말'] = (df['주말여부']=='주말').astype(int)
    df['기온범위'] = df['최고기온'] - df['최저기온']
    df['계절코드'] = df['월'].apply(lambda x: 1 if x in [3,4,5] else 2 if x in [6,7,8] else 3 if x in [9,10,11] else 4)
    df['금요일'] = (df['요일']=='금').astype(int)
    df['성수기'] = df['월'].apply(lambda x: 1 if x in [4,5,9,10] else 0)
    features = ['월','요일코드','주말','최저기온','최고기온','날씨점수','기온범위','계절코드','금요일','성수기']
    df_model = df.dropna(subset=features+['실입장객'])
    X = df_model[features]
    y = df_model['실입장객']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model, df

@st.cache_data
def load_hourly():
    all_dfs = []
    for year_str, year in [('23년',2023),('24년',2024),('25년',2025)]:
        try:
            df = pd.read_excel('시간대별입장객(23-25년)_260116.xlsx', sheet_name=year_str, header=None)
            mask = df.iloc[:,1].apply(lambda x: hasattr(x,'hour'))
            df_time = df[mask].copy()
            df_time['시간'] = df_time.iloc[:,1].apply(lambda x: x.hour)
            month_cols = list(range(2,14))
            df_sel = df_time[month_cols+['시간']].copy()
            df_sel.columns = ['1월','2월','3월','4월','5월','6월',
                              '7월','8월','9월','10월','11월','12월','시간']
            for m in ['1월','2월','3월','4월','5월','6월',
                      '7월','8월','9월','10월','11월','12월']:
                df_sel[m] = pd.to_numeric(df_sel[m], errors='coerce')
            df_sel = df_sel.groupby('시간').mean().reset_index()
            df_sel['연도'] = year
            all_dfs.append(df_sel)
        except:
            pass
    return pd.concat(all_dfs, ignore_index=True)

# 연령별 추천 코스
def get_course(age, child_type=None):
    if child_type is None:
        child_type = []
    is_timid = "겁이 많은" in child_type
    is_tired = "쉽게 지치는" in child_type
    is_curious = "탐구적인" in child_type

    if age <= 3:
        return {
            'name': '영아 동반 코스',
            'zones': ['세계의 광장', '삼천리 동산', '캐릭터타운'],
            'rides': ['회전목마', '꼬마기차', '동물농장'],
            'tip': '이동 거리를 최소화하고 수유실·휴게 공간을 중심으로 동선을 구성했습니다.',
            'rest': '캐릭터타운 내 수유실, 삼천리 동산 벤치 구역'
        }
    elif age <= 6:
        rides = ['회전목마', '꼬마비행기', '범퍼카', '미니바이킹']
        if is_curious:
            rides.append('동물농장')
        if is_tired:
            rides = rides[:3]
        return {
            'name': '유아 중심 코스',
            'zones': ['세계의 광장', '캐릭터타운'] if is_tired else ['세계의 광장', '캐릭터타운', '삼천리 동산'],
            'rides': rides,
            'tip': '동화풍 놀이기구 중심 코스입니다.' + (' 체력을 고려해 코스를 단축했습니다.' if is_tired else ' 3~4시간 이내 방문을 권장합니다.'),
            'rest': '캐릭터타운 중앙 광장, 삼천리 동산 휴게소'
        }
    elif age <= 10:
        if is_timid:
            rides = ['회전목마', '범퍼카', '깜짝모험관', '타임머신 5D']
            zones = ['세계의 광장', '캐릭터타운', '미래의 나라']
            tip = '스릴이 강한 시설을 제외하고 체험 중심으로 구성했습니다.'
        else:
            rides = ['범퍼카', '타임머신 5D', '깜짝모험관', '후룸라이드']
            zones = ['세계의 광장', '캐릭터타운', '미래의 나라', '모험의 나라 일부']
            tip = '4D/5D 체험을 포함한 코스입니다. 키 제한 시설 사전 확인을 권장합니다.'
        if is_curious:
            rides.append('쥬라기랜드')
        if is_tired:
            zones = zones[:2]
            rides = rides[:3]
            tip += ' 체력을 고려해 코스를 단축했습니다.'
        return {
            'name': '초등 저학년 코스',
            'zones': zones,
            'rides': rides,
            'tip': tip,
            'rest': '미래의 나라 푸드코트, 모험의 나라 입구 벤치'
        }
    else:
        if is_timid:
            rides = ['범퍼카', '타임머신 5D', '깜짝모험관', '회전목마']
            zones = ['세계의 광장', '미래의 나라', '캐릭터타운', '삼천리 동산']
            tip = '스릴이 강한 시설을 제외하고 체험 중심으로 구성했습니다.'
        else:
            rides = ['후룸라이드', '자이로드롭', '타임머신 5D', '범퍼카', '회전목마']
            zones = ['세계의 광장', '모험의 나라', '미래의 나라', '캐릭터타운', '삼천리 동산']
            tip = '하루 종일 방문 가능한 풀코스입니다. 오전 스릴 → 오후 체험 순서를 추천합니다.'
        if is_curious:
            rides.append('쥬라기랜드')
        if is_tired:
            zones = zones[:3]
            rides = rides[:3]
            tip += ' 체력을 고려해 코스를 단축했습니다.'
        return {
            'name': '초등 고학년 풀코스',
            'zones': zones,
            'rides': rides,
            'tip': tip,
            'rest': '세계의 광장 푸드코트, 모험의 나라 휴게 공간'
        }

# 방문 시간 추천 함수
def recommend_time(month, weekday, min_temp, max_temp, weather, model):
    weather_score = {'맑음':5,'구름조금':4,'흐림':3,'비/우천':2,'눈':1,'기타':3}
    weekday_map = {'월':0,'화':1,'수':2,'목':3,'금':4,'토':5,'일':6}
    weekday_code = weekday_map[weekday]
    is_weekend = 1 if weekday_code in [5,6] else 0
    is_friday = 1 if weekday_code == 4 else 0
    is_peak = 1 if month in [4,5,9,10] else 0
    season = 1 if month in [3,4,5] else 2 if month in [6,7,8] else 3 if month in [9,10,11] else 4
    temp_range = max_temp - min_temp
    features = [[month, weekday_code, is_weekend, min_temp, max_temp,
                 weather_score.get(weather,3), temp_range, season, is_friday, is_peak]]
    predicted = model.predict(features)[0]
    if predicted >= 7000:
        time_pool = ['14시','15시','16시','17시']
    elif predicted >= 3000:
        time_pool = ['10시','15시','16시','17시']
    else:
        time_pool = ['9시','10시','11시','15시']
    return int(predicted), time_pool

with st.spinner(''):
    model, df_daily = load_and_train()
    df_hourly = load_hourly()

# =====================
# 화면 1: 멤버십 선택
# =====================
if st.session_state.page == 'membership':
    st.title("서울랜드 키즈 멤버십")
    st.markdown("아이와 함께하는 최적의 방문 계획을 세워드립니다")
    st.divider()

    st.subheader("멤버십을 선택해주세요")
    st.caption("방문 목적에 맞는 플랜을 골라보세요")

    st.markdown("""
    <div style='background:#fff; border:0.5px solid #e0e0e0; border-radius:12px; padding:16px; margin-bottom:12px;'>
        <div style='font-size:12px; color:#C8101E; background:#FDECEA; padding:3px 10px; border-radius:20px; display:inline-block; margin-bottom:8px;'>진입 단계</div>
        <div style='font-size:16px; font-weight:600;'>1회권</div>
        <div style='font-size:22px; font-weight:700; color:#C8101E; margin:4px 0;'>₩4,000</div>
        <div style='font-size:12px; color:#888; margin-bottom:10px;'>1회 방문에 한해 이용 가능</div>
        <div style='font-size:12px; color:#555;'>✓ 추천 방문 시간 안내</div>
        <div style='font-size:12px; color:#555;'>✓ 가족 친화 이용 가이드</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("1회권으로 시작하기"):
        st.session_state.membership = '1회권'
        st.session_state.page = 'profile'
        st.rerun()

    st.markdown("""
    <div style='background:#fff; border:2px solid #C8101E; border-radius:12px; padding:16px; margin-bottom:12px; margin-top:12px;'>
        <div style='font-size:12px; color:#C8101E; background:#FDECEA; padding:3px 10px; border-radius:20px; display:inline-block; margin-bottom:8px;'>추천</div>
        <div style='font-size:16px; font-weight:600;'>정기 멤버십</div>
        <div style='font-size:22px; font-weight:700; color:#C8101E; margin:4px 0;'>₩79,000 <span style='font-size:13px; color:#888; font-weight:400;'>/ 년</span></div>
        <div style='font-size:11px; color:#C8101E; background:#FDECEA; padding:2px 8px; border-radius:20px; display:inline-block; margin-bottom:4px;'>월 ₩6,583 — 34% 절약</div>
        <div style='font-size:12px; color:#888; text-decoration:line-through; margin-bottom:10px;'>월 ₩9,900</div>
        <div style='font-size:12px; color:#555;'>✓ 추천 방문 시간 안내</div>
        <div style='font-size:12px; color:#555;'>✓ 가족 친화 이용 가이드</div>
        <div style='font-size:12px; color:#555;'>✓ 맞춤형 방문 코스 추천</div>
        <div style='font-size:12px; color:#555;'>✓ 멤버십 전용 할인 혜택</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("정기 멤버십으로 시작하기"):
        st.session_state.membership = '정기 멤버십'
        st.session_state.page = 'profile'
        st.rerun()

# =====================
# 화면 2: 아이 프로필 입력
# =====================
elif st.session_state.page == 'profile':

    st.title("아이 정보 입력")
    st.caption("입력하신 정보는 맞춤 추천에 활용됩니다")
    st.divider()

    child_age = st.selectbox(
        "아이 연령",
        list(range(1, 14)),
        index=st.session_state.child_age - 1,
        format_func=lambda x: f"만 {x}세"
    )

    child_type = st.multiselect(
        "아이 성향 (복수 선택 가능)",
        ["활동적인", "조용한", "탐구적인", "신중한", "겁이 많은", "사교적인", "쉽게 지치는"],
        default=st.session_state.child_type if st.session_state.child_type else ["활동적인"]
    )

    visit_purpose = st.selectbox(
        "방문 목적",
        ["놀이 중심", "체험/학습", "가족 나들이", "특별한 날"],
        index=["놀이 중심", "체험/학습", "가족 나들이", "특별한 날"].index(
            st.session_state.visit_purpose) if st.session_state.visit_purpose else 0
    )

    st.divider()

    btn_col1, btn_col2 = st.columns([1,1])
    with btn_col1:
        if st.button("이전", key="profile_prev"):
            st.session_state.page = 'membership'
            st.rerun()
    with btn_col2:
        if st.button("다음", key="profile_next"):
            st.session_state.child_age = child_age
            st.session_state.child_type = child_type
            st.session_state.visit_purpose = visit_purpose
            st.session_state.page = 'date'
            st.rerun()

# =====================
# 화면 3: 방문 날짜/날씨 입력
# =====================
elif st.session_state.page == 'date':
    st.title("방문 날짜 선택")
    st.caption("방문 예정일의 날씨 정보를 입력해주세요")
    st.divider()

    visit_month = st.slider("방문 월", 1, 12, st.session_state.visit_month)

    visit_weekday = st.selectbox(
        "요일",
        ["월", "화", "수", "목", "금", "토", "일"],
        index=["월", "화", "수", "목", "금", "토", "일"].index(st.session_state.visit_weekday)
    )

    visit_weather = st.selectbox(
        "예상 날씨",
        ["맑음", "구름조금", "흐림", "비/우천", "눈"],
        index=["맑음", "구름조금", "흐림", "비/우천", "눈"].index(st.session_state.visit_weather)
    )

    min_temp = st.slider("최저 기온 (℃)", -15, 30, st.session_state.min_temp)
    max_temp = st.slider("최고 기온 (℃)", -10, 40, st.session_state.max_temp)

    if visit_weather in ['비/우천', '눈']:
        st.warning("우천 예보가 있습니다. 실내 시설 위주로 동선을 구성해드립니다.")

    st.divider()

    btn_col1, btn_col2 = st.columns([1,1])
    with btn_col1:
        if st.button("이전", key="date_prev"):
            st.session_state.page = 'profile'
            st.rerun()
    with btn_col2:
        if st.button("추천 받기", key="date_next"):
            st.session_state.visit_month = visit_month
            st.session_state.visit_weekday = visit_weekday
            st.session_state.visit_weather = visit_weather
            st.session_state.min_temp = min_temp
            st.session_state.max_temp = max_temp
            predicted, time_pool = recommend_time(
                visit_month, visit_weekday, min_temp, max_temp, visit_weather, model
            )
            st.session_state.predicted_visitors = predicted
            st.session_state.recommended_time = time_pool
            st.session_state.recommended_course = get_course(st.session_state.child_age, st.session_state.child_type)
            st.session_state.page = 'recommendation'
            st.rerun()

# =====================
# 화면 4: 사전 추천 결과
# =====================
elif st.session_state.page == 'recommendation':
    predicted = st.session_state.predicted_visitors
    time_pool = st.session_state.recommended_time
    course = st.session_state.recommended_course if st.session_state.membership == '정기 멤버십' else None

    if predicted >= 7000:
        congestion_label = "혼잡 예상"
        congestion_color = "#C8101E"
    elif predicted >= 3000:
        congestion_label = "보통 예상"
        congestion_color = "#FFA500"
    else:
        congestion_label = "여유 예상"
        congestion_color = "#2ECC71"

    st.title("방문 추천 결과")
    st.caption(f"만 {st.session_state.child_age}세 아이를 위한 맞춤 추천입니다")
    st.divider()

    # 추천 방문 시간
    st.subheader("추천 방문 시간")
    st.markdown(
        f"<div style='background:#FDECEA; border-radius:10px; padding:14px; text-align:center;'>"
        f"<div style='font-size:28px; font-weight:700; color:#C8101E;'>{time_pool[0]}</div>"
        f"<div style='font-size:13px; color:#888; margin-top:4px;'>최적 입장 시간</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown(f"""
    <div style='display:flex; gap:8px; margin-top:8px;'>
        {"".join([f"<div style='flex:1; background:#f9f9f9; border:0.5px solid #e0e0e0; border-radius:8px; padding:10px; text-align:center; font-size:13px;'>{t}</div>" for t in time_pool[1:]])}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"<div style='margin-top:8px; font-size:13px; color:{congestion_color}; text-align:center;'>"
        f"당일 예상 입장객 {int(predicted):,}명 — {congestion_label}</div>",
        unsafe_allow_html=True
    )

    st.divider()

    # 추천 코스 (정기 멤버십만)
    if st.session_state.membership == '정기 멤버십':
        st.subheader("추천 방문 코스")
        st.markdown(f"**{course['name']}**")
        st.markdown("**구역 순서**")
        for i, zone in enumerate(course['zones']):
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>"
                f"<div style='width:24px; height:24px; background:#C8101E; color:#fff; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:600;'>{i+1}</div>"
                f"<div style='font-size:14px;'>{zone}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        st.markdown("**추천 놀이기구**")
        for ride in course['rides']:
            st.markdown(f"• {ride}")
        st.markdown(
            f"<div style='background:#f9f9f9; border-left:3px solid #C8101E; padding:10px 14px; border-radius:4px; margin-top:10px; font-size:13px; color:#555;'>"
            f"{course['tip']}</div>",
            unsafe_allow_html=True
        )
        if st.session_state.visit_weather in ['비/우천', '눈']:
            st.warning("우천 예보가 있어 실내 시설 위주로 동선을 조정했습니다.")
        st.divider()

        # 아이 연령 맞춤 안내
        age = st.session_state.child_age
        if age <= 3:
            st.info("영아의 경우 수유실과 휴게 공간 위치를 미리 확인하세요.")
        elif age <= 6:
            st.info("유아의 경우 3~4시간 이내 방문을 권장합니다.")
        elif age <= 10:
            st.info("키 제한이 있는 시설은 사전에 확인해주세요.")
        else:
            st.info("하루 종일 방문 가능합니다. 오전 스릴 → 오후 체험 순서를 추천합니다.")
        st.divider()

    st.divider()

    if st.button("입장하기 (메인 홈으로)"):
        st.session_state.page = 'home'
        st.rerun()

# =====================
# 화면 5: 메인 홈
# =====================
elif st.session_state.page == 'home':
    course = st.session_state.recommended_course if st.session_state.membership == '정기 멤버십' else None
    time_pool = st.session_state.recommended_time
    predicted = st.session_state.predicted_visitors

    if predicted >= 7000:
        congestion_label = "혼잡"
        congestion_color = "#C8101E"
    elif predicted >= 3000:
        congestion_label = "보통"
        congestion_color = "#FFA500"
    else:
        congestion_label = "여유"
        congestion_color = "#2ECC71"

    st.title("오늘의 방문 플랜")
    st.caption(f"만 {st.session_state.child_age}세 · {st.session_state.visit_purpose}")
    st.divider()

    # 추천 시간 요약
    st.markdown(
        f"<div style='background:#FDECEA; border-radius:10px; padding:14px;'>"
        f"<div style='font-size:12px; color:#888; margin-bottom:4px;'>추천 입장 시간</div>"
        f"<div style='font-size:26px; font-weight:700; color:#C8101E;'>{time_pool[0]}</div>"
        f"<div style='font-size:12px; color:{congestion_color}; margin-top:4px;'>"
        f"현재 혼잡도 예상: {congestion_label} ({int(predicted):,}명)</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.divider()

# 오늘의 코스 요약 (정기 멤버십만)
    if st.session_state.membership == '정기 멤버십' and course:
        st.subheader("오늘의 추천 코스")
        st.markdown(f"**{course['name']}**")
        zones_html = ""
        for i, zone in enumerate(course['zones']):
            arrow = " → " if i < len(course['zones']) - 1 else ""
            zones_html += f"<span style='background:#C8101E; color:#fff; padding:3px 10px; border-radius:20px; font-size:12px; margin-right:4px;'>{zone}</span>{arrow}"
        st.markdown(zones_html, unsafe_allow_html=True)

    st.divider()

    # 시간대별 혼잡도 현황
    st.subheader("시간대별 혼잡도")
    month_col = f"{st.session_state.visit_month}월"
    df_hourly_month = df_hourly.groupby('시간')[month_col].mean()
    df_hourly_op = df_hourly_month[df_hourly_month.index.isin(range(9, 22))]
    max_val = df_hourly_op.max()

    for hour, val in df_hourly_op.items():
        ratio = val / max_val
        if ratio >= 0.66:
            grade, color = "혼잡", "#C8101E"
        elif ratio >= 0.33:
            grade, color = "보통", "#FFA500"
        else:
            grade, color = "여유", "#2ECC71"
        bar_width = int(ratio * 100)
        st.markdown(
            f"""<div style='display:flex; align-items:center; margin-bottom:5px;'>
                <div style='width:36px; font-size:12px;'>{hour}시</div>
                <div style='flex:1; background:#f0f0f0; border-radius:4px; height:16px; margin:0 8px;'>
                    <div style='width:{bar_width}%; background:{color}; height:100%; border-radius:4px;'></div>
                </div>
                <div style='width:36px; font-size:11px; color:{color};'>{grade}</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.divider()

    # 편의시설 안내
    if st.session_state.membership == '정기 멤버십' and course:
        st.subheader("편의시설 안내")
        st.markdown(f"**휴게 공간:** {course['rest']}")

    if st.session_state.child_age <= 6:
        st.markdown("**수유실:** 캐릭터타운 내 위치")
        st.markdown("**유모차 대여:** 정문 입구 안내소")

    st.divider()

    if st.session_state.membership == '정기 멤버십':
        btn_col1, btn_col2 = st.columns([1,1])
        with btn_col1:
            if st.button("동선 지도 보기", key="home_map"):
                st.session_state.page = 'map'
                st.rerun()
        with btn_col2:
            if st.button("방문 종료 후\n리포트 작성", key="home_report"):
                st.session_state.page = 'report'
                st.rerun()
    else:
        if st.button("방문 종료 후 리포트 작성"):
            st.session_state.page = 'report'
            st.rerun()

# =====================
# 화면 6: 동선 지도
# =====================
elif st.session_state.page == 'map':
    from PIL import Image, ImageDraw, ImageFont
    import io
    import base64

    course = st.session_state.recommended_course
    st.title("동선 안내")
    st.caption("서울랜드 내 추천 방문 경로입니다")
    st.divider()

    # 구역별 이미지 좌표 (가이드맵 기준 픽셀 위치)
    zones_px = {
        '세계의 광장':      (900, 780),
        '모험의 나라':      (1150, 480),
        '캐릭터타운':       (1050, 350),
        '미래의 나라':      (420, 280),
        '삼천리 동산':      (380, 480),
        '모험의 나라 일부':  (1150, 480),
    }

    course_zones = course['zones']

    try:
        img = Image.open('seoulland_map.png').convert('RGBA')
        orig_w, orig_h = img.size  # 1748 x 1237

        # 좌표를 비율로 정의 (x비율, y비율)
        zones_ratio = {
            '세계의 광장':      (0.420, 0.530),
            '모험의 나라':      (0.620, 0.580),
            '캐릭터타운':       (0.650, 0.300),
            '미래의 나라':      (0.300, 0.220),
            '삼천리 동산':      (0.330, 0.430),
            '모험의 나라 일부':  (0.620, 0.580),
        }

        # 비율을 실제 픽셀로 변환
        zones_px = {
            k: (int(v[0] * orig_w), int(v[1] * orig_h))
            for k, v in zones_ratio.items()
        }

        draw = ImageDraw.Draw(img)

        # 경로 선 그리기
        path_points = []
        for zone in course_zones:
            if zone in zones_px:
                path_points.append(zones_px[zone])

        if len(path_points) > 1:
            for i in range(len(path_points) - 1):
                draw.line(
                    [path_points[i], path_points[i+1]],
                    fill=(200, 16, 30, 200),
                    width=8
                )

        # 마커 그리기
        for i, zone in enumerate(course_zones):
            if zone in zones_px:
                x, y = zones_px[zone]
                r = 25
                draw.ellipse(
                    [x-r, y-r, x+r, y+r],
                    fill=(200, 16, 30, 230),
                    outline=(255, 255, 255, 255),
                    width=3
                )
                try:
                    font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 40)
                except:
                    font = ImageFont.load_default()
                draw.text(
                    (x, y),
                    str(i+1),
                    fill=(255, 255, 255),
                    anchor='mm',
                    font=font
                )

        # base64 변환
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()

        # 확대/축소 가능한 HTML로 표시
        map_html = f"""
        <div style="border-radius:12px; border:1px solid #e0e0e0; overflow:hidden;">
            <div style="overflow:auto; -webkit-overflow-scrolling:touch;">
                <img
                    id="map-img"
                    src="data:image/png;base64,{img_b64}"
                    style="width:100%; display:block; transform-origin:top left;"
                />
            </div>
        </div>
        <div style="display:flex; gap:8px; margin-top:8px; justify-content:center; position:sticky; bottom:0; background:white; padding:8px 0;">
            <button onclick="zoomIn()" id="btn-in" style="
                background:#C8101E; color:white; border:none;
                padding:8px 20px; border-radius:8px; font-size:16px; cursor:pointer;
            ">+ 확대</button>
            <button onclick="zoomOut()" id="btn-out" style="
                background:#888; color:white; border:none;
                padding:8px 20px; border-radius:8px; font-size:16px; cursor:pointer;
                opacity:0.4; pointer-events:none;
            ">- 축소</button>
            <button onclick="resetZoom()" style="
                background:#444; color:white; border:none;
                padding:8px 20px; border-radius:8px; font-size:16px; cursor:pointer;
            ">초기화</button>
        </div>
        <script>
            var scale = 1;
            var img = document.getElementById('map-img');
            var btnIn = document.getElementById('btn-in');
            var btnOut = document.getElementById('btn-out');

            function updateButtons() {{
                btnIn.style.opacity = scale >= 2 ? '0.4' : '1';
                btnIn.style.pointerEvents = scale >= 2 ? 'none' : 'auto';
                btnOut.style.opacity = scale <= 1 ? '0.4' : '1';
                btnOut.style.pointerEvents = scale <= 1 ? 'none' : 'auto';
            }}

            function zoomIn() {{
                if (scale < 2) {{
                    scale = 2;
                    img.style.width = '200%';
                    updateButtons();
                }}
            }}
            function zoomOut() {{
                if (scale > 1) {{
                    scale = 1;
                    img.style.width = '100%';
                    updateButtons();
                }}
            }}
            function resetZoom() {{
                scale = 1;
                img.style.width = '100%';
                updateButtons();
            }}
        </script>
        """
        components.html(map_html, height=520)

    except Exception as e:
        st.error(f'지도 로드 실패: {e}')

    st.divider()

    # 추천 방문 순서
    st.subheader("추천 방문 순서")
    for i, zone in enumerate(course_zones):
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:10px; margin-bottom:8px;'>"
            f"<div style='width:26px; height:26px; background:#C8101E; color:#fff; "
            f"border-radius:50%; display:flex; align-items:center; justify-content:center; "
            f"font-size:12px; font-weight:600;'>{i+1}</div>"
            f"<div style='font-size:14px;'>{zone}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.divider()
    if st.button("메인 홈으로"):
        st.session_state.page = 'home'
        st.rerun()

# =====================
# 화면 7: 방문 리포트
# =====================
elif st.session_state.page == 'report':
    st.title("방문 리포트")
    st.caption("오늘 방문은 어떠셨나요?")
    st.divider()

    st.subheader("오늘의 방문 요약")
    st.markdown(f"- **방문 월:** {st.session_state.visit_month}월")
    st.markdown(f"- **아이 연령:** 만 {st.session_state.child_age}세")
    st.markdown(f"- **멤버십:** {st.session_state.membership}")

    st.divider()

    st.subheader("만족도 평가")
    overall = st.slider("전반적인 만족도", 1, 5, 4,
                        format="%d점")
    if st.session_state.membership == '정기 멤버십':
        recommend_score = st.slider("추천 코스 적절성", 1, 5, 4,
                                    format="%d점")
    time_score = st.slider("추천 방문 시간 적절성", 1, 5, 4,
                           format="%d점")

    st.divider()

    st.subheader("아이가 가장 좋아한 활동")
    favorite = st.multiselect(
        "선택해주세요 (복수 선택 가능)",
        ["놀이기구", "공연/퍼레이드", "체험 프로그램", "식사/휴식", "기타"]
    )

    feedback = st.text_area("자유 의견 (선택)", placeholder="서비스 개선을 위한 의견을 남겨주세요")

    st.divider()

    if st.button("제출하기"):
        st.success("소중한 의견 감사합니다! 다음 방문 시 더 정확한 추천을 드릴게요.")
        st.markdown(
            f"<div style='background:#FDECEA; border-radius:10px; padding:16px; margin-top:12px;'>"
            f"<div style='font-size:14px; font-weight:600; color:#C8101E; margin-bottom:8px;'>다음 방문 미리보기</div>"
            f"<div style='font-size:13px; color:#555;'>누적 방문 데이터를 바탕으로 더 정교한 추천을 준비할게요.</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        if st.button("처음으로 돌아가기"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()