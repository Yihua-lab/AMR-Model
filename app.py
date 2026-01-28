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
    
    c1, c2 = st.columns([2, 1])
    with c1:
        # 默认序列建议使用与泊沙康唑研究相关的序列
        user_seq = st.text_area("输入序列 (Protein Sequence)", placeholder="Paste Erg11 sequence here...", height=150)
        site = st.number_input("扫描位点索引 (1-based)", value=132, min_value=1)
    
    with c2:
        st.markdown("💊泊沙康唑临床关联")
        if site in CLINICAL_VARIANTS:
            st.error(f"已知泊沙康唑耐药相关位点: \n {CLINICAL_VARIANTS}")
        else:
            st.success("该位点在目前泊沙康唑常见耐药研究中不属于核心热点。")
if st.button("运行泊沙康唑风险模拟"):
        if not user_seq:
            st.warning("请先输入序列")
        else:
            # 动态加载大模型（仅在计算时，防止 OOM）
            with st.spinner('正在通过 ESM-2 模拟泊沙康唑结合环境下的蛋白稳定性...'):
                esm_mlm = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
                esm_base = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
                
                AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
                scan_results = []
                
                # 进度条
                bar = st.progress(0)
                for i, aa in enumerate(AMINO_ACIDS):
                    mut_list = list(user_seq)
                    mut_list[site-1] = aa
                    mut_seq = "".join(mut_list)
                    
                    inputs = tokenizer(mut_seq, return_tensors="pt")
                    with torch.no_grad():
                        # 耐药概率 (MLP)
                        base_out = esm_base(**inputs)
                        emb = base_out.last_hidden_state.mean(dim=1).numpy()
                        prob = clf.predict_proba(emb)[0][1]
                        
                        # 稳定性得分 (MLM)
                        mlm_out = esm_mlm(**inputs)
                        logits = mlm_out.logits[0, site-1]
                        s_prob = torch.softmax(logits, dim=-1)
                        stability = s_prob[tokenizer.convert_tokens_to_ids(aa)].item()
                        
                        scan_results.append({'AA': aa, 'Prob': prob, 'Stability': stability})
                    bar.progress((i+1)/len(AMINO_ACIDS))

                # 绘图可视化
                res_df = pd.DataFrame(scan_results)
                
                

                fig, ax1 = plt.subplots(figsize=(10, 5))
                # 绘制耐药风险 (柱状图)
                ax1.bar(res_df['AA'], res_df['Prob'], color='#A9C9E2', alpha=0.6, label='Posa-Resistance Prob')
                ax1.set_ylabel("Posaconazole Resistance Probability", color='#2E5A88')
                ax1.axhline(0.5, color='red', linestyle='--', alpha=0.3, label='Threshold')
                
                # 绘制稳定性 (折线图)
                ax2 = ax1.twinx()
                ax2.plot(res_df['AA'], res_df['Stability'], color='#D65A5A', marker='D', linewidth=1.5, label='Structural Fitness')
                ax2.set_ylabel("Structural Stability Score", color='#D65A5A')
                
                plt.title(f"In-silico Scan for Posaconazole Resistance at Site {site}")
                st.pyplot(fig)
                
                # 内存清理
                del esm_mlm, esm_base
                gc.collect()

                st.info("💡 **分析结论提示**：如果某一氨基酸突变导致柱状图极高且红点极低，说明该突变虽然极度耐药但蛋白极不稳定，可能在真实环境下难以存活。")


