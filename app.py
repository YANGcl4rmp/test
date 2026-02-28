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
import time  # 新增：用於 API 請求的延遲與重試機制

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
# 5. K-Means 分群演算法 & 道路吸附 (增強版 API)
# -----------------------------------------------------
if len(coords) < k_value_input + 1:
    st.warning("模擬點數太少，無法進行分群！")
    st.stop()

# 處理山區獨立站邏輯
if use_remote:
    lon_threshold = np.percentile(coords[:, 0], 85)
    remote_mask =
