import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from transformers import AutoTokenizer, EsmModel

# --- 页面设置 ---
st.set_page_config(page_title="AI真菌/细菌耐药性突变演化预测工作台", layout="wide")
st.title("🍄微生物耐药性全流程分析工作台")

# --- 加载资源 ---
@st.cache_resource
def load_assets():
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    esm_model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
    clf = joblib.load('amr_model.pkl')
    pca_proc = joblib.load('pca_processor.pkl')
    return tokenizer, esm_model, clf, pca_proc

tokenizer, esm_model, clf, pca_proc = load_assets()

# --- 核心提取函数 ---
def get_embedding(sequence):
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = esm_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# --- 侧边栏导航 ---
mode = st.sidebar.selectbox("选择分析模式", ["CSV 批量分析", "单点扫描预览"])

if mode == "CSV 批量分析":
    st.header("📂 CSV 批量处理与数据可视化")
    uploaded_file = st.file_uploader("上传 CSV 文件 (需包含 'sequence' 列)", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'sequence' not in df.columns:
            st.error("CSV 文件必须包含 'sequence' 列！")
        else:
            if st.button("开始全流程分析"):
                with st.spinner('正在提取 ESM-2 特征并进行 PCA 降维...'):
                    # 1. 提取 Embeddings
                    embeddings = []
                    for seq in df['sequence']:
                        embeddings.append(get_embedding(seq).flatten())
                    X = np.array(embeddings)
                    
                    # 2. 预测标签
                    df['predicted_prob'] = clf.predict_proba(X)[:, 1]
                    df['label'] = (df['predicted_prob'] > 0.5).astype(int)
                    
                    # 3. PCA 可视化
                    X_pca = pca_proc.transform(X)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("PCA 语义空间投影")
                        fig, ax = plt.subplots()
                        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df['label'], cmap='coolwarm', edgecolors='k')
                        plt.colorbar(scatter, label='Resistance')
                        st.pyplot(fig)
                    
                    with col2:
                        st.subheader("分析结果预览")
                        st.write(df[['sequence', 'predicted_prob', 'label']].head())
                        
                st.success("批量分析完成！")
                st.download_button("下载分析结果", df.to_csv(index=False), "analysis_results.csv", "text/csv")

elif mode == "单点扫描预览":
    st.header("🧬 位点突变的耐药演化风险扫描 (Deep Mutational Scanning)")
    # 此处放置你之前的位点扫描逻辑代码...
    # (逻辑同前，用于展示那张高低错落的柱状图)