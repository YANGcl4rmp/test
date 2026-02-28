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
use_remote = st.sidebar.checkbox("⛰️ 獨立設置「偏遠山區」專用站", value=True
                                ,help="【選址邏輯】：先找出距離全區人口重心「最遠的 5% 極端偏遠人口」，再獨立計算這群人的中心點來設置此站。確保醫療站真正深入山區，避免重心被市區人口拉回。"
)

# 讓使用者選擇要模擬的總人口點數
total_points_input = st.sidebar.slider("👥 模擬人口總點數 (樣本數)", min_value=100, max_value=3000, value=800, step=100)

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
    
    gdf = gdf_vill.merge(df_pop, left_on="VILLNAME", right_on="里別", how="inner")
    
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
# 5. K-Means 分群演算法 & 大馬路吸附 (Overpass API)
# -----------------------------------------------------
if len(coords) < k_value_input + 1:
    st.warning("模擬點數太少，無法進行分群！")
    st.stop()

if use_remote:
    center_data = coords.mean(axis=0)
    d_center = np.linalg.norm(coords - center_data, axis=1)
    
    threshold = np.percentile(d_center, 95)
    remote_mask = d_center >= threshold
    core_mask = ~remote_mask
    
    urban_coords = coords[core_mask]
    remote_coords = coords[remote_mask]
    
    if len(remote_coords) > 0 and len(urban_coords) >= k_value_input:
        top_k = max(5, int(len(remote_coords) * 0.3)) 
        top_k = min(top_k, len(remote_coords))
        
        remote_d = d_center[remote_mask]
        order = np.argsort(remote_d)
        farthest_coords = remote_coords[order[-top_k:]]
        
        remote_center = farthest_coords.mean(axis=0)
        
        km = KMeans(n_clusters=k_value_input, n_init=10, random_state=42)
        km.fit(urban_coords)
        urban_centers = km.cluster_centers_
        
        raw_centers = np.vstack([urban_centers, remote_center])
        is_remote_flag = [False] * k_value_input + [True]
        labels = np.argmin(cdist(coords, raw_centers), axis=1)
    else:
        km = KMeans(n_clusters=k_value_input, n_init=10, random_state=42)
        km.fit(coords)
        raw_centers = km.cluster_centers_
        labels = km.labels_
        is_remote_flag = [False] * k_value_input
else:
    km = KMeans(n_clusters=k_value_input, n_init=10, random_state=42)
    km.fit(coords)
    raw_centers = km.cluster_centers_
    labels = km.labels_
    is_remote_flag = [False] * k_value_input

# 🚀 全新升級：Overpass API 巷弄過濾器
def get_nearest_major_road(lon, lat, retries=3):
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # 查詢指令：找出 3000 公尺內，分類為「主幹道、次幹道、一般道路」且有名字的線段
    overpass_query = f"""
    [out:json];
    way(around:3000,{lat},{lon})["highway"~"^(trunk|primary|secondary|tertiary|residential)$"]["name"];
    out center;
    """
    
    for attempt in range(retries):
        try:
            time.sleep(1.5)  
            response = requests.post(overpass_url, data={'data': overpass_query}, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                elements = data.get("elements", [])
                
                best_road = None
                min_dist = float('inf')
                
                # 遍歷所有找到的道路，尋找合法且距離最近的
                for el in elements:
                    name = el.get("tags", {}).get("name", "")
                    
                    # 🚨 核心過濾器：看到「巷」跟「弄」直接淘汰！
                    if "巷" in name or "弄" in name:
                        continue
                        
                    c_lat = el.get("center", {}).get("lat")
                    c_lon = el.get("center", {}).get("lon")
                    
                    if c_lat and c_lon:
                        # 計算距離平方來比對遠近
                        dist = (c_lat - lat)**2 + (c_lon - lon)**2
                        if dist < min_dist:
                            min_dist = dist
                            best_road = (c_lon, c_lat, name)
                            
                if best_road:
                    return best_road
                else:
                    return lon, lat, "附近無合適大馬路"
            else:
                time.sleep(1)
        except requests.exceptions.RequestException:
            time.sleep(1)
            continue
            
    return lon, lat, "未知道路 (API連線失敗)"

snapped_centers = []
street_names = []

with st.spinner("🌍 正在尋找最近的大馬路 (自動過濾巷弄，請稍候)..."):
    for center in raw_centers:
        lon, lat = center[0], center[1]
        n_lon, n_lat, s_name = get_nearest_major_road(lon, lat)
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

