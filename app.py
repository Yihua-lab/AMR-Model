import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import gc
import os
from transformers import AutoTokenizer, EsmForMaskedLM, EsmModel

# --- 1. 配置与临床数据库 (基于 Table S7) ---
st.set_page_config(page_title="AMR Advanced Analyzer", layout="wide")

# 模拟文献中的 36 个耐药变异（部分核心展示）
CLINICAL_VARIANTS = {
    132: "Y132H/F: 三唑类广谱耐药位点，但泊沙康唑对其亲和力略高于氟康唑",
    464: "G464S: 位于血红素结合区，是泊沙康唑耐药的关键突变点",
    121: "F121L: 泊沙康唑特异性相关的通道突变",
    467: "G467S: 影响血红素附近的结构，可导致对泊沙康唑敏感度下降",
    450: "I450V: 临床常见，常与其他突变协同影响泊沙康唑结合",
    252: "S252P: 可能影响长链唑类药物进入活性口袋"
}

# 标准参考序列 (C. albicans CBS 562 Erg11 - 部分展示)
REF_SEQ = "MSIVETVVDGINYKGKDLKVWIP..." 

# --- 2. 内存优化加载 ---
@st.cache_resource
def load_assets():
    # 统一使用 8M 模型
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    clf = joblib.load('amr_model.pkl')
    pca = joblib.load('pca_processor.pkl')
    return tokenizer, clf, pca

tokenizer, clf, pca = load_assets()

# --- 3. 核心功能函数 ---
def get_stability_and_prob(sequence, site_idx, target_aa):
    """
    动态加载模型计算稳定性 (Likelihood) 和 耐药概率
    """
    # 动态加载以节省初始内存
    model_mlm = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model_base = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
    
    inputs = tokenizer(sequence, return_tensors="pt")
    with torch.no_grad():
        # A. 计算耐药概率 (MLP + Base Model)
        base_out = model_base(**inputs)
        emb = base_out.last_hidden_state.mean(dim=1).numpy()
        prob = clf.predict_proba(emb)[0][1]
        
        # B. 计算稳定性 (MLM Log-Likelihood)
        mlm_out = model_mlm(**inputs)
        logits = mlm_out.logits[0, site_idx]
        softmax_probs = torch.softmax(logits, dim=-1)
        token_id = tokenizer.convert_tokens_to_ids(target_aa)
        stability = softmax_probs[token_id].item()

    # 显式清理
    del model_mlm, model_base
    gc.collect()
    return prob, stability

# --- 4. 界面展示 ---
st.title("C. albicans🍄ERG11基因对泊沙康唑耐药相关的突变位点预测平台")

tab1, tab2 = st.tabs(["📂批量CSV分析", "🧬单位点演化分析&稳定性分析"])
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
                    
with tab2:
    st.subheader("单位点演化风险与蛋白稳定性评估")
    # 1. 基础输入区
    user_seq = st.text_area("待分析序列 (Protein Sequence)", 
                            placeholder="请粘贴 Erg11 蛋白质序列...", 
                            height=150, 
                            key="input_seq")
    
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        site = st.number_input("扫描位点索引 (1-based)", value=132, min_value=1)  
    scan_clicked = st.button("🚀开始分析")

if scan_clicked:
        # --- 安全检查 A: 序列是否存在 ---
        if not user_seq or len(user_seq.strip()) == 0:
            st.warning("⚠️ 请先输入蛋白质序列。")
        
        # --- 安全检查 B: 位点是否越界 (修复 IndexError) ---
        elif site > len(user_seq.strip()):
            st.error(f"❌ 索引越界：当前序列长度为 {len(user_seq.strip())}，无法访问第 {site} 位点。")
            st.info("请检查序列是否完整，或位点输入是否有误。")
            
        else:
            # --- 模块一：动态临床预警 (仅点击后显示) ---
            st.subheader("🔍 1. 临床背景评估")
            if site in CLINICAL_VARIANTS:
                st.error(f"⚠️临床耐药热点识别:\n\n {CLINICAL_VARIANTS[site]}")
            else:
                st.success(f"该位点 (Site {site}) 目前非泊沙康唑核心临床热点位点。")

            # --- 模块二：序列比对预览 ---
            st.subheader("🔗 2. 局部序列比对预览")
            # 自动截取位点前后各 10 个氨基酸
            start_v = max(0, site - 11)
            end_v = min(len(user_seq), site + 10)
            view_segment = user_seq[start_v:end_v]
            # 计算指针位置
            pointer_pos = site - start_v - 1
            st.code(f"区域: {view_segment}\n标记: {' ' * pointer_pos}^ (Site {site})")

            # --- 模块三：执行深度模拟 ---
            with st.spinner(f'正在模拟第 {site} 位点的 20 种氨基酸突变...'):
                try:
                    # 动态载入模型以节省启动内存
                    esm_mlm = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
                    esm_base = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
                    
                    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
                    scan_data = []
                    prog_bar = st.progress(0)
                    
                    for i, aa in enumerate(AMINO_ACIDS):
                        # 构造突变序列并执行修改 (这里已通过前面的 elif 确保安全)
                        mut_list = list(user_seq.strip())
                        mut_list[site-1] = aa
                        mut_seq = "".join(mut_list)
                        
                        inputs = tokenizer(mut_seq, return_tensors="pt")
                        with torch.no_grad():
                            # 耐药概率预测
                            b_out = esm_base(**inputs)
                            emb = b_out.last_hidden_state.mean(dim=1).numpy()
                            prob = clf.predict_proba(emb)[0][1]
                            
                            # 稳定性预测
                            m_out = esm_mlm(**inputs)
                            logits = m_out.logits[0, site-1]
                            stab = torch.softmax(logits, dim=-1)[tokenizer.convert_tokens_to_ids(aa)].item()
                            
                            scan_data.append({'AA': aa, 'Prob': prob, 'Stability': stab})
                        prog_bar.progress((i + 1) / 20)
                    
                    # --- 模块四：多维结果可视化 ---
                    st.subheader("📊3.风险与稳定性多维扫描图")
                    res_df = pd.DataFrame(scan_data)
                    
                    fig, ax1 = plt.subplots(figsize=(10, 5))
                    # 柱状图：耐药概率
                    ax1.bar(res_df['AA'], res_df['Prob'], color='#4a90e2', alpha=0.4, label='Resistance Prob')
                    ax1.set_ylabel("Posaconazole Resistance Probability", color='#4a90e2', fontsize=12)
                    ax1.axhline(0.5, color='red', linestyle='--', alpha=0.3)
                    ax1.set_ylim(0, 1.05)
                    
                    # 折线图：蛋白质稳定性
                    ax2 = ax1.twinx()
                    ax2.plot(res_df['AA'], res_df['Stability'], color='#d0021b', marker='o', linewidth=1.5, label='Stability')
                    ax2.set_ylabel("Protein Stability (Likelihood)", color='#d0021b', fontsize=12)
                    
                    plt.title(f"Posaconazole Mutational Landscape at Site {site}", fontsize=14)
                    st.pyplot(fig)
                    
                    # 清理内存
                    del esm_mlm, esm_base
                    gc.collect()
                    st.success("✅分析完成。")
  st.info("💡 **分析结论提示**：如果某一氨基酸突变导致柱状图极高且红点极低，说明该突变虽然极度耐药但蛋白极不稳定，可能在真实环境下难以存活。")
                except Exception as e:
                    st.error(f"分析失败，原因: {e}")
