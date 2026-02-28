import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium import features
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.cluster import KMeans
import os
from pathlib import Path
import requests
# -----------------------------------------------------
# 1. 網頁基本設定 (Streamlit)
# -----------------------------------------------------
st.set_page_config(page_title="太平區篩檢站選址系統", layout="wide")
st.title("太平區篩檢站配置模擬")

BASE_DIR = Path(__file__).resolve().parent
VILL_GEOJSON = BASE_DIR / "Tai-Ping-Ge-Li.json"
POP_CSV = BASE_DIR / "Xia-Zai.csv"

# -----------------------------------------------------
# 2. 側邊欄控制面板 (Sidebar)
# -----------------------------------------------------
st.sidebar.header(" 參數控制面板")

# 讓使用者選擇要模擬的總人口點數
total_points_input = st.sidebar.slider(" 模擬人口總點數 (樣本數)", min_value=100, max_value=3000, value=800, step=100)

# 讓使用者選擇 K 值 (篩檢站數量)
k_value_input = st.sidebar.slider(" 篩檢站數量 (K值)", min_value=2, max_value=8, value=4, step=1)

# 底圖切換
map_style = st.sidebar.selectbox(" 地圖底圖樣式", ["街道圖 (OpenStreetMap)", "地形圖 (OpenTopoMap)"])

st.sidebar.markdown("---")
st.sidebar.info("調整上方參數後，右側的地圖與圖表會即時重新運算！")

# -----------------------------------------------------
# 3. 讀取與處理資料 (使用 cache 加速，不用每次拉動拉桿都重讀檔案)
# -----------------------------------------------------
@st.cache_data
def load_and_merge_data():
    if not VILL_GEOJSON.exists() or not POP_CSV.exists():
        st.error("找不到地圖或人口檔案！請確認 json 與 csv 在同一資料夾。")
        st.stop()
        
    gdf_vill = gpd.read_file(VILL_GEOJSON, encoding="utf-8")
    df_pop = pd.read_csv(POP_CSV, encoding="cp950")
    df_pop["人口數"] = df_pop["人口數"].astype(str).str.replace(",", "").astype(int)
    
    # 合併圖資與人口
    gdf = gdf_vill.merge(df_pop, left_on="VILLNAME", right_on="里別", how="inner")
    
    # 確保座標系為經緯度 (EPSG:4326) 以供 Folium 使用
    if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
        
    return gdf, df_pop["人口數"].sum()

gdf, total_real_pop = load_and_merge_data()

# -----------------------------------------------------
# 4. 依據面板輸入動態生成人口點
# -----------------------------------------------------
# 計算動態 SCALE：真實總人口 / 預期點數
SCALE = total_real_pop / total_points_input
gdf["點數"] = (gdf["人口數"] / SCALE).round().astype(int)

# 確保生成點數的函數
def random_points_in_polygon(polygon, n_points):
    points = []
    if polygon is None or polygon.is_empty:
        return points
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < n_points:
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        p = Point(x, y)
        if polygon.contains(p):
            points.append([p.x, p.y])  # [lon, lat]
    return points

all_points = []
for idx, row in gdf.iterrows():
    n = int(row["點數"])
    if n > 0:
        pts = random_points_in_polygon(row.geometry, n)
        all_points.extend(pts)

coords = np.array(all_points)

# -----------------------------------------------------
# 5. K-Means 分群演算法
# -----------------------------------------------------
if len(coords) < k_value_input:
    st.warning("模擬點數太少，無法進行分群！")
    st.stop()

# 進行 K-Means 分群
km = KMeans(n_clusters=k_value_input, n_init=10, random_state=42)
km.fit(coords)
raw_centers = km.cluster_centers_  # 這是純數學算出來的原始中心
labels = km.labels_

# 定義一個呼叫 OSRM API 來尋找最近道路的函數
def get_nearest_road(lon, lat):
    # OSRM Nearest API 格式: lon, lat
    url = f"http://router.project-osrm.org/nearest/v1/driving/{lon},{lat}?number=1"
    try:
        response = requests.get(url, timeout=3)
        data = response.json()
        if data.get("code") == "Ok":
            # 抓取最近道路的座標與名稱
            nearest_lon, nearest_lat = data["waypoints"][0]["location"]
            street_name = data["waypoints"][0].get("name", "")
            if not street_name:
                street_name = "無名道路 / 巷弄"
            return nearest_lon, nearest_lat, street_name
    except Exception as e:
        pass
    # 如果 API 失敗，就回傳原始座標
    return lon, lat, "未知道路"

# 開始將每個數學中心點吸附到道路上
snapped_centers = []
street_names = []

with st.spinner("🌍 正在將篩檢站對齊至實際道路..."):
    for center in raw_centers:
        lon, lat = center[0], center[1]
        n_lon, n_lat, s_name = get_nearest_road(lon, lat)
        snapped_centers.append([n_lon, n_lat])
        street_names.append(s_name)

snapped_centers = np.array(snapped_centers)
# -----------------------------------------------------
# 6. 繪製互動式地圖 (Folium)
# -----------------------------------------------------
st.subheader(f" 互動式選址地圖 (K={k_value_input})")

# 計算太平區中心點
center_lat = coords[:, 1].mean()
center_lon = coords[:, 0].mean()

# 設定底圖
tiles_url = "OpenStreetMap"
attr = None
if map_style == "地形圖 (OpenTopoMap)":
    tiles_url = "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
    attr = "Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)"

m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=tiles_url, attr=attr)

# 【亮點 1】加入各里邊界與滑鼠懸停資訊 (Tooltip)
folium.GeoJson(
    gdf,
    style_function=lambda feature: {
        'fillColor': 'transparent',
        'color': 'black',
        'weight': 1.5,
        'fillOpacity': 0.1
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['里別', '人口數'],
        aliases=[' 里名:', ' 人口數:'],
        localize=True
    )
).add_to(m)

# 準備群組顏色
colors = ["#E63946", "#457B9D", "#2A9D8F", "#F4A261", "#E9C46A", "#8D99AE", "#9B5DE5", "#F15BB5"]

# 【亮點 2】畫出模擬人口點
for i, coord in enumerate(coords):
    cluster_idx = labels[i]
    color = colors[cluster_idx % len(colors)]
    folium.CircleMarker(
        location=[coord[1], coord[0]], # folium 吃 [lat, lon]
        radius=3,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        weight=0
    ).add_to(m)

# 【亮點 3】畫出篩檢站 (大星星，已對齊道路)
for i, center in enumerate(snapped_centers): # 改用 snapped_centers
    street = street_names[i]
    
    # 設計超酷的彈出資訊框 (支援 HTML)
    tooltip_html = f"""
    <div style='font-family: Microsoft JhengHei;'>
        <b>🏥 建議篩檢站 {i+1}</b><br>
        🛣️ 位於: <span style='color:blue;'>{street}</span>
    </div>
    """
    
    folium.Marker(
        location=[center[1], center[0]],
        icon=folium.Icon(color="red", icon="info-sign"), # 換成帶有資訊符號的紅色標記
        tooltip=tooltip_html
    ).add_to(m)

# -----------------------------------------------------
# 7. 顯示肘部法圖表 (Matplotlib)
# -----------------------------------------------------
st.subheader("📊 肘部法分析 (Elbow Method)")

with st.spinner("計算肘部法數據中..."):
    sse = []
    K_range = range(1, min(11, len(coords)//2))
    for k in K_range:
        km_temp = KMeans(n_clusters=k, n_init=5, random_state=42)
        km_temp.fit(coords)
        sse.append(km_temp.inertia_)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(K_range, sse, 'bo-', linewidth=2, markersize=8)
    ax.set_title('Elbow Method')
    ax.set_xlabel('Number of clusters (K)')
    ax.set_ylabel('SSE')
    ax.grid(True)
    st.pyplot(fig)
