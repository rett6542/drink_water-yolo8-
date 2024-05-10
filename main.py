# Import required libraries
import PIL # pillow影像處理套件

import streamlit as st

from ultralytics import YOLO
# ultralytics是一間公司


# 載入Yolov8模型參數
model_path = './best.pt'
try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"未找到指定模型，請重新檢查參數路徑: {model_path}")
    st.error(ex)


# 網頁設定
st.set_page_config(
    page_title="drink water",  # Setting page title
    page_icon="cup_with_straw",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",    # Expanding sidebar by default直接讓測邊欄顯示   
)


# 主頁面設定
#&nbsp代表一個空格符號
page_title = """
<div>
    <span style="font-size:2rem;font-weight:bold">喝水偵測</span>
    <span style="font-size:1rem;">&nbsp;上傳要偵測的照片，找出哪個有喝水的動作</span>
</div>
"""
st.markdown(page_title, unsafe_allow_html=True)
st.markdown('')  #加入一段分行

# 將主頁分成兩欄
col1, col2 = st.columns(2)


# 設定側邊欄
with st.sidebar:
    st.header("上傳圖片")     # 側邊欄的標題

    # 在側邊欄加入 file uploader 來上傳圖片
    source_img = st.file_uploader('',
         type=("jpg", "jpeg", "png", "bmp", "webp"))
    st.caption('(檔案大小限制:200MB)')
    st.caption('(檔案類型限制:jpg, jpeg, png, bmp, webp)')

    # Model Options(調整模型信心水準)
    confidence = float(st.slider(
        "選擇信心水準", 25, 100, 50)) / 100
    
    st.caption('按下後開始')

# 圖片上傳後加到第一欄
with col1:
    st.write('##### 上傳的圖片:')
    if source_img:
        # 顯示上傳的照片
        uploaded_image = PIL.Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(source_img,
                 caption="Uploaded Image",
                 use_column_width=True
                 )


# 模型執行區塊
if st.sidebar.button('確定', type="primary"): # 按鈕點選 按下後啟用模型
    try: 
        res = model.predict(uploaded_image,
                        conf=confidence )     # 載入圖片跟信心水準 
        boxes = res[0].boxes # 框出位置
        res_plotted = res[0].plot()[:, :, ::-1]    
    except:
        st.write("圖片尚未上傳!")
    
    # 將測試結果顯示到第二欄
    with col2:
        st.write('##### 偵測的結果:')
        st.image(res_plotted,
                 caption='Detected Image',
                 use_column_width=True
                 )
        try:
            with st.expander("偵測的座標"):
                for box in boxes:
                    st.write(box.xywh)
        except Exception as ex:
            st.write("沒有偵測的資料!")
