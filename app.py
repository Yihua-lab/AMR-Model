import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import gc
from transformers import AutoTokenizer, EsmModel

# --- 1. 页面设置 ---
st.set_page_config(page_title="AMR演化分析平台", layout="wide", initial_sidebar_state="expanded")

st.title("🧬真菌Erg11基因耐药性AI分析平台")
st.markdown("""
本工作台集成了ESM-2蛋白质语言模型与机器学习分类器，支持批量序列评估与单基因位点演化预测。
""")

# --- 2. 轻量化资源加载 ---
@st.cache_resource
def load_static_assets():
    """预加载 Tokenizer 和训练好的小模型，这些占用内存极小"""
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    # 载入你在 Colab 导出的文件
    clf = joblib.load('amr_model.pkl')
    pca_proc = joblib.load('pca_processor.pkl')
    return tokenizer, clf, pca_proc

# 动态加载 ESM-2 基础模型 (仅在计算时调用以节省内存)
def load_esm_model():
    return EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

# --- 3. 核心计算函数 ---
def extract_embedding(text_sequence, _tokenizer, _model):
    """提取序列的 ESM-2 平均表征"""
    inputs = _tokenizer(text_sequence, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = _model(**inputs)
    # 取最后一层隐藏状态的平均值
    return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()

# 初始化基础组件
tokenizer, clf, pca_proc = load_static_assets()

# --- 4. 界面功能区 ---
tab1, tab2 = st.tabs(["📂批量CSV分析(PCA)", "🧬单位点演化分析"])

# --- Tab 1: 批量分析 ---
with tab1:
    st.header("CSV 批量分析模式")
    uploaded_file = st.file_uploader("上传 CSV 文件 (需包含 'sequence' 列)", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if st.button("开始批量处理"):
            if 'sequence' not in df.columns:
                st.error("错误：CSV 必须包含 sequence 列")
            else:
                with st.spinner('正在激活 ESM-2 引擎并提取特征...'):
                    # 动态加载大模型
                    esm_model = load_esm_model()
                    
                    # 批量提取
                    embeddings = []
                    for s in df['sequence']:
                        emb = extract_embedding(s, tokenizer, esm_model)
                        embeddings.append(emb.flatten())
                    
                    X = np.array(embeddings)
                    df['Resistance_Prob'] = clf.predict_proba(X)[:, 1]
                    df['Label'] = ["Resistant" if p > 0.5 else "Susceptible" for p in df['Resistance_Prob']]
                    
                    # PCA 绘图
                    X_pca = pca_proc.transform(X)
                    st.subheader("PCA 语义空间聚类可视化")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    for label, color in zip(["Susceptible", "Resistant"], ["#4A90E2", "#E35454"]):
                        mask = df['Label'] == label
                        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=label, edgecolors='k', alpha=0.7)
                    ax.set_xlabel("PC1 (Variance Explained)")
                    ax.set_ylabel("PC2")
                    ax.legend()
                    st.pyplot(fig)
                    
                    st.dataframe(df)
                    
                    # 释放大模型内存
                    del esm_model
                    gc.collect()

# --- Tab 2: 位点扫描 ---
with tab2:
    st.header("Deep Mutational Scanning (DMS) 模拟")
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        wild_seq = st.text_area("输入原始序列", value="MSIVETVVDGINYKGKDLKVWIP...", height=200)
    with col_b:
        site_index = st.number_input("扫描位点索引 (例如 132)", value=132)
        scan_btn = st.button("生成扫描报告")

   # --- 修正后的 Tab 2 核心逻辑 ---
if scan_btn:
    with st.spinner('计算演化风险路径...'):
        esm_model = load_esm_model()
        AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
        
        # 1. 先计算原始序列（Wild Type）的基准得分
        base_emb = extract_embedding(wild_seq, tokenizer, esm_model)
        base_prob = clf.predict_proba(base_emb.reshape(1, -1))[0][1]
        
        scan_results = []
        for aa in AMINO_ACIDS:
            mut_list = list(wild_seq)
            if site_index <= len(mut_list):
                mut_list[site_index - 1] = aa
                mut_seq = "".join(mut_list)
                
                emb = extract_embedding(mut_seq, tokenizer, esm_model)
                prob = clf.predict_proba(emb.reshape(1, -1))[0][1]
                
                # 计算相对风险增量 (Delta)
                # 这样即使所有概率都高，也能看出谁比原始序列更危险
                delta = prob - base_prob 
                scan_results.append({'AA': aa, 'Prob': prob, 'Delta': delta})
        
        res_df = pd.DataFrame(scan_results)
        
        # --- 绘图逻辑：展示“风险偏移”而不是“绝对值” ---
        st.subheader(f"第 {site_index} 位点的演化风险偏移 (Delta Analysis)")
        st.info(f"原始序列在该位点的基准耐药概率为: {base_prob:.2%}")

        fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
        
        # 使用 Delta 绘图：高于基准线的变红，低于基准线的变蓝
        # 这种方式能极其敏锐地捕捉到微小的理化性质变化
        bar_colors = ['#E35454' if d > 0 else '#74ADD1' for d in res_df['Delta']]
        ax_bar.bar(res_df['AA'], res_df['Delta'], color=bar_colors, edgecolor='black')
        
        ax_bar.axhline(0, color='black', linewidth=1) # 零基准线
        ax_bar.set_ylabel("Risk Change (Delta from Wild Type)")
        ax_bar.set_xlabel("Amino Acid Mutation")
        
        # 添加标注
        ax_bar.text(0, max(res_df['Delta'])*1.1 if len(res_df)>0 else 0.1, 
                    "更高风险 ↑", color='red', fontsize=10)
        ax_bar.text(0, min(res_df['Delta'])*1.1 if len(res_df)>0 else -0.1, 
                    "风险降低 ↓", color='blue', fontsize=10)
        
        st.pyplot(fig_bar)
        
        # 释放内存
        del esm_model
        gc.collect()
