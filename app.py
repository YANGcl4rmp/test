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
from scipy.spatial.distance import cdist
import os
from pathlib import Path
import requests

# -----------------------------------------------------
# 1. 網頁基本設定 (Streamlit)
# -----------------------------------------------------
st.set_page_config(page_title="太平區篩檢站選址系統", layout="wide")
st.title("📍 太平區篩檢站配置模擬系統")

BASE_DIR = Path(__file__).resolve().parent
VILL_GEOJSON = BASE_DIR / "Tai-Ping-Ge-Li.json"
POP_CSV = BASE_DIR / "Xia-Zai.csv"

# -----------------------------------------------------
# 2. 側邊欄控制面板 (Sidebar)
# -----------------------------------------------------
st.sidebar.header("⚙️ 參數控制面板")

# 加入偏遠山區開關
use_remote = st.sidebar.checkbox("⛰️ 獨立設置「偏遠山區」專用站", value=True)

# 讓使用者選擇要模擬的總人口點數
total_points_input = st.sidebar.slider("👥 模擬人口總點數 (樣本數)", min_value=100, max_value=3000, value=800, step=100)

# 修改 K 值說明
k_value_input = st.sidebar.slider(
    "🏥 市區篩檢站數量 (K值)", 
    min_value=1, max_value=8, value=3, step=1, 
    help="注意：此數量為市區站點。若有勾選上方『偏遠山區』，系統會額外再+1個山區站點！"
)

# 底圖切換
map_style = st.sidebar.selectbox("🗺️ 地圖底圖樣式", ["街道圖 (OpenStreetMap)", "地形圖 (OpenTopoMap)"])

st.sidebar.markdown("---")
st.sidebar.info("💡 調整上方參數後，右側的地圖與圖表會即時重新運算！")

# -----------------------------------------------------
# 3. 讀取與處理資料 (使用 cache 加速)
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
SCALE = total_real_pop / total_points_input
gdf["點數"] = (gdf["人口數"] / SCALE).round().astype(int)

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
# 5. K-Means 分群演算法 & 道路吸附 (避開無名巷弄)
# -----------------------------------------------------
if len(coords) < k_value_input + 1:
    st.warning("模擬點數太少，無法進行分群！")
    st.stop()

# 處理山區獨立站邏輯
if use_remote:
    # 太平區東邊是山區，以經度前 85% 作為市區與山區的分界
    lon_threshold = np.percentile(coords[:, 0], 85)
    remote_mask = coords[:, 0] > lon_threshold
    
    urban_coords = coords[~remote_mask]
    remote_coords = coords[remote_mask]
    
    # 若有成功分出山區點位
    if len(remote_coords) > 0 and len(urban_coords) >= k_value_input:
        remote_center = remote_coords.mean(axis=0)
        
        km = KMeans(n_clusters=k_value_input, n_init=10, random_state=42)
        km.fit(urban_coords)
        urban_centers = km.cluster_centers_
        
        raw_centers = np.vstack([urban_centers, remote_center])
        is_remote_flag = [False] * k_value_input + [True]
        labels = np.argmin(cdist(coords, raw_centers), axis=1)
    else:
        # 防呆：如果點數過少無法拆分，退回一般模式
        km = KMeans(n_clusters=k_value_input, n_init=10, random_state=42)
        km.fit(coords)
        raw_centers = km.cluster_centers_
        labels = km.labels_
        is_remote_flag = [False] * k_value_input
else:
    # 沒勾選山區，全區一起算
    km = KMeans(n_clusters=k_value_input, n_init=10, random_state=42)
    km.fit(coords)
    raw_centers = km.cluster_centers_
    labels = km.labels_
    is_remote_flag = [False] * k_value_input

# 尋找最近的「有名字」大路 API
def get_nearest_named_road(lon, lat):
    url = f"http://router.project-osrm.org/nearest/v1/driving/{lon},{lat}?number=10&radius=2000"
    try:
        response = requests.get(url, timeout=3)
        data = response.json()
        if data.get("code") == "Ok":
            for wp in data["waypoints"]:
                street_name = wp.get("name", "")
                if street_name: 
                    return wp["location"][0], wp["location"][1], street_name
            if data["waypoints"]:
                wp = data["waypoints"][0]
                return wp["location"][0], wp["location"][1], "無名道路 / 巷弄"
    except Exception as e:
        pass
    return lon, lat, "未知道路"

snapped_centers = []
street_names = []

with st.spinner("🌍 正在尋找最近的主要道路..."):
    for center in raw_centers:
        lon, lat = center[0], center[1]
        n_lon, n_lat, s_name = get_nearest_named_road(lon, lat)
        snapped_centers.append([n_lon, n_lat])
        street_names.append(s_name)

snapped_centers = np.array(snapped_centers)

# -----------------------------------------------------
# 6. 繪製互動式地圖 (Folium)
# -----------------------------------------------------
st.subheader(f"🗺️ 互動式選址地圖")

center_lat = coords[:, 1].mean()
center_lon = coords[:, 0].mean()

tiles_url = "OpenStreetMap"
attr = None
if map_style == "地形圖 (OpenTopoMap)":
    tiles_url = "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
    attr = "Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)"

m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=tiles_url, attr=attr)

# 加入各里邊界與滑鼠懸停資訊
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
        aliases=['📍 里名:', '👥 人口數:'],
        localize=True
    )
).add_to(m)

# 畫出模擬人口點
colors = ["#E63946", "#457B9D", "#2A9D8F", "#F4A261", "#E9C46A", "#8D99AE", "#9B5DE5", "#F15BB5"]
for i, coord in enumerate(coords):
    cluster_idx = labels[i]
    color = colors[cluster_idx % len(colors)]
    folium.CircleMarker(
        location=[coord[1], coord[0]],
        radius=3,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        weight=0
    ).add_to(m)

# 畫出篩檢站 (區分市區與山區)
for i, center in enumerate(snapped_centers): 
    street = street_names[i]
    is_remote = is_remote_flag[i]
    
    if is_remote:
        icon_color = "green"
        icon_shape = "leaf"
        title_text = "⛰️ 偏遠山區專用篩檢站"
    else:
        icon_color = "red"
        icon_shape = "plus"
        title_text = f"🏥 市區建議篩檢站 {i+1}"
    
    tooltip_html = f"""
    <div style='font-family: Microsoft JhengHei;'>
        <b>{title_text}</b><br>
        🛣️ 位於: <span style='color:blue;'>{street}</span>
    </div>
    """
    
    folium.Marker(
        location=[center[1], center[0]],
        icon=folium.Icon(color=icon_color, icon=icon_shape), 
        tooltip=tooltip_html
    ).add_to(m)

# 🚨 關鍵：顯示地圖並關閉自動重整機制
st_folium(m, width=1000, height=600, returned_objects=[])

# -----------------------------------------------------
# 7. 顯示肘部法圖表 (Matplotlib)
# -----------------------------------------------------
st.subheader("📊 肘部法分析 (Elbow Method)")

with st.spinner("計算肘部法數據中..."):
    sse = []
    # 肘部法統一使用全區數據計算以保持趨勢基準一致
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
