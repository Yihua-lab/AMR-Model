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
    132: "Y132H/F: 临床最常见突变，显著降低氟康唑亲和力",
    467: "G467S: 位于血红素结合区，临床已知高频突变",
    105: "K105E: 改变蛋白入口柔性，已被多项研究证实",
    143: "P143R: 常见于临床分离株，伴随高水平耐药",
    450: "I450V: 临床突变，常与其他位点协同作用",
    372: "M372V: Harrison et al. 2025重点提到的自然/临床演化位点"
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

with tab2:
    st.subheader("单位点演化风险与蛋白稳定性评估")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        user_seq = st.text_area("输入序列 (Protein Sequence)", value=REF_SEQ, height=150)
        site = st.number_input("扫描位点索引 (1-based)", value=132, min_value=1)
    
    with c2:
        st.markdown("### 🔍 临床背景")
        if site in CLINICAL_VARIANTS:
            st.error(f"**临床热点位点预警:** \n {CLINICAL_VARIANTS[site]}")
        else:
            st.success("该位点目前未在 36 个已知临床耐药变异中列出。")

    if st.button("开始深度评估"):
        # 1. 序列比对预览 (Alignment Viewer)
        st.markdown("---")
        st.subheader("🔗 局部序列比对 (对比野生型 CBS 562)")
        start, end = max(0, site-11), min(len(user_seq), site+10)
        st.code(f"野生型: {REF_SEQ[start:end]}\n待测株: {user_seq[start:end]}\n        {' '*(site-start-1)}^ 分析点")

        # 2. 深度扫描计算
        AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
        scan_results = []
        
        progress = st.progress(0)
        for i, aa in enumerate(AMINO_ACIDS):
            mut_list = list(user_seq)
            mut_list[site-1] = aa
            mut_seq = "".join(mut_list)
            
            p, s = get_stability_and_prob(mut_seq, site, aa)
            scan_results.append({'AA': aa, 'Prob': p, 'Stability': s})
            progress.progress((i+1)/len(AMINO_ACIDS))

        # 3. 双维度可视化
        res_df = pd.DataFrame(scan_results)
        st.subheader("综合风险评价图")
        
        
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        # 绘制耐药风险
        ax1.bar(res_df['AA'], res_df['Prob'], color='skyblue', alpha=0.5, label='Resistance Probability')
        ax1.set_ylabel("Predicted Resistance Prob", color='skyblue')
        ax1.axhline(0.5, color='gray', linestyle='--')
        
        # 绘制稳定性 (折线)
        ax2 = ax1.twinx()
        ax2.plot(res_df['AA'], res_df['Stability'], color='crimson', marker='o', label='Structure Stability')
        ax2.set_ylabel("Stability (Likelihood Score)", color='crimson')
        
        plt.title(f"Multi-dimensional Scan at Site {site}")
        st.pyplot(fig)
        
        st.caption("注：柱状图越高表示耐药风险越大；红点越高表示突变对蛋白结构的破坏越小。")

        # 释放内存
        del esm_model
        gc.collect()


