import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import os
from pathlib import Path
import requests
import time

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
st.sidebar.info("💡 調整上方參數後，右側的地圖與圖表會重新運算 (尋路 API 約需數秒)。")

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
            points.append([p.x, p.y])
    return points

all_points = []
for idx, row in gdf.iterrows():
    n = int(row["點數"])
    if n > 0:
        pts = random_points_in_polygon(row.geometry, n)
        all_points.extend(pts)

coords = np.array(all_points)

# -----------------------------------------------------
# 5. K-Means 分群演算法 & 道路吸附 (回歸極端距離演算法)
# -----------------------------------------------------
if len(coords) < k_value_input + 1:
    st.warning("模擬點數太少，無法進行分群！")
    st.stop()

# 處理山區獨立站邏輯 (回歸 Yang 的極端點距離演算法)
if use_remote:
    # 1. 先算出所有人口的整體中心
    center_data = coords.mean(axis=0)
    # 2. 計算每個點到整區中心的直線距離
    d_center = np.linalg.norm(coords - center_data, axis=1)
    
    # 3. 抓出距離最遠的 5% (閾值 threshold)
    threshold = np.percentile(d_center, 95)
    remote_mask = d_center >= threshold
    core_mask = ~remote_mask
    
    urban_coords = coords[core_mask]
    remote_coords = coords[remote_mask]
    
    if len(remote_coords) > 0 and len(urban_coords) >= k_value_input:
        # 4. 為了不被拉回市區，我們只拿「最遠的那幾個人」來算山區篩檢站的位置
        top_k = max(5, int(len(remote_coords) * 0.3)) # 取最遠的 30% 或至少 5 個人
        top_k = min(top_k, len(remote_coords))
        
        remote_d = d_center[remote_mask]
        order = np.argsort(remote_d)
        farthest_coords = remote_coords[order[-top_k:]]
        
        # 山區站的精確位置：極端遙遠點的重心
        remote_center = farthest_coords.mean(axis=0)
        
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

# 增強版尋路 API：改用 Nominatim 官方地圖解析，避免雲端 IP 被封鎖
def get_nearest_named_road(lon, lat, retries=3):
    url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=17"
    headers = {"User-Agent": "TaipingScreeningApp/1.0 (Student Project)"}
    
    for attempt in range(retries):
        try:
            time.sleep(1.1)  # 遵守官方規定，每次請求間隔 1.1 秒
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                address = data.get("address", {})
                street_name = address.get("road", "")
                
                if street_name: 
                    n_lat = float(data.get("lat", lat))
                    n_lon = float(data.get("lon", lon))
                    return n_lon, n_lat, street_name
                else:
                    return lon, lat, "無名道路 / 巷弄"
            else:
                time.sleep(1)
                
        except requests.exceptions.RequestException:
            time.sleep(1)
            continue
            
    return lon, lat, "未知道路 (API連線失敗)"

snapped_centers = []
street_names = []

with st.spinner("🌍 正在尋找最近的主要道路 (遵守 API 流量限制，請稍候幾秒鐘)..."):
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

st_folium(m, width=1000, height=600, returned_objects=[])

# -----------------------------------------------------
# 7. 顯示肘部法圖表與詳細數據
# -----------------------------------------------------
st.subheader("📊 肘部法分析 (Elbow Method)")

with st.spinner("計算肘部法數據中..."):
    sse = []
    table_data = []
    K_range = list(range(1, min(11, len(coords)//2)))
    
    for k in K_range:
        km_temp = KMeans(n_clusters=k, n_init=5, random_state=42)
        km_temp.fit(coords)
        current_sse = km_temp.inertia_
        sse.append(current_sse)
        
        if k == 1:
            drop = 0
            improvement = 0.0
        else:
            prev_sse = sse[-2]
            drop = prev_sse - current_sse
            improvement = (drop / prev_sse) * 100
            
        table_data.append({
            "K 值": k,
            "SSE (誤差平方和)": round(current_sse, 2),
            "縮小幅度": round(drop, 2) if k > 1 else "-",
            "改善率 (%)": f"{improvement:.2f}%" if k > 1 else "-"
        })

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(K_range, sse, 'bo-', linewidth=2, markersize=8)
    ax.set_title('Elbow Method')
    ax.set_xlabel('Number of clusters (K)')
    ax.set_ylabel('SSE')
    ax.grid(True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.pyplot(fig)
        
    with col2:
        st.markdown("#### 📈 詳細數據表")
        df_elbow = pd.DataFrame(table_data)
        st.dataframe(df_elbow, use_container_width=True, hide_index=True)
