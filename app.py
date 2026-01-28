import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import gc
from transformers import AutoTokenizer, EsmForMaskedLM # 注意：这里改用 MLM 模型来预测稳定性

# --- 1. 静态数据：根据文献 Table S6/S7 整理的临床位点 ---
CLINICAL_SITES = {
    132: "Y132H/F - 临床高频耐药位点",
    467: "G467S - 影响血红素结合",
    105: "K105E - 改变蛋白柔性",
    143: "P143R - 常见于白念珠菌",
    372: "M372V - 重要进化位点"
}

# 标准野生型 Erg11 序列 (片段示例，建议替换为完整 CBS 562 序列)
WILD_TYPE_REF = "MSIVETVVDGINYKGKDLKVWIP..." 

@st.cache_resource
def load_advanced_assets():
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    # 使用 MaskedLM 来计算序列的 Log-likelihood (稳定性指标)
    esm_mlm = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    clf = joblib.load('amr_model.pkl')
    return tokenizer, esm_mlm, clf

tokenizer, esm_mlm, clf = load_advanced_assets()

# --- 2. 界面设计 ---
st.title("🍄 真菌 Erg11 耐药演化分析终端 (Advanced)")

tab1, tab2 = st.tabs(["📂批量CSV分析", "🧬单位点演化分析&稳定性分析"])

with tab2:
    st.header("单位点演化分析与生物学评估")
    
    col_input, col_info = st.columns([2, 1])
    with col_input:
        user_seq = st.text_area("输入待测 Erg11 序列", height=150, value=WILD_TYPE_REF)
        site = st.number_input("分析位点 (Index)", value=132)
    
    with col_info:
        # 模块 1：临床位点标注
        if site in CLINICAL_SITES:
            st.warning(f"⚠️ 预警：该位点是已知临床耐药热点\n\n备注: {CLINICAL_SITES[site]}")
        else:
            st.success("✅ 该位点尚未在主流临床文献中报道为热点")

    if st.button("开始深度多维度评估"):
        # 模块 2：简单的序列比对工具 (Alignment Viewer)
        st.subheader("🔍 局部序列比对 (User vs. CBS 562 Wildtype)")
        start_view = max(0, site-10)
        end_view = min(len(user_seq), site+10)
        ref_segment = WILD_TYPE_REF[start_view:end_view]
        user_segment = user_seq[start_view:end_view]
        
        st.code(f"Ref: {ref_segment}\nUser: {user_segment}\n      {' '*(site-start_view-1)}^ 指针位置")

        # 模块 3：稳定性与风险预测
        AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
        results = []
        
        with st.spinner('AI 正在计算稳定性得分与耐药概率...'):
            for aa in AMINO_ACIDS:
                mut_list = list(user_seq)
                mut_list[site-1] = aa
                mut_seq = "".join(mut_list)
                
                # 计算耐药概率 (MLP)
                # (此处略去 Embedding 提取代码，同前)
                prob = 0.85 # 示例值
                
                # 计算稳定性 (ESM Log-Likelihood)
                # 分值越高表示该氨基酸在该位置越“自然”，越低表示越不稳定
                inputs = tokenizer(mut_seq, return_tensors="pt")
                with torch.no_grad():
                    logits = esm_mlm(**inputs).logits
                    # 简化算法：取该位点的 softmax 概率作为稳定性指标
                    token_id = tokenizer.convert_tokens_to_ids(aa)
                    stability_score = torch.softmax(logits[0, site], dim=-1)[token_id].item()
                
                results.append({'AA': aa, 'Prob': prob, 'Stability': stability_score})

        # 绘图：双指标展示
        res_df = pd.DataFrame(results)
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        # 绘制耐药风险 (柱状图)
        ax1.bar(res_df['AA'], res_df['Prob'], alpha=0.3, color='gray', label='Resistance Prob')
        ax1.set_ylabel("Resistance Probability", color='gray')
        
        # 绘制稳定性 (折线图)
        ax2 = ax1.twinx()
        ax2.plot(res_df['AA'], res_df['Stability'], color='red', marker='o', label='Protein Stability (ESM)')
        ax2.set_ylabel("Stability Score (higher is better)", color='red')
        
        plt.title(f"Site {site}: Resistance Risk vs. Protein Stability")
        st.pyplot(fig)
        
        st.info("💡 提示：高风险且高稳定性的突变（红点在上方且柱条较长）在临床中最具威胁。")
        # 释放内存
        del esm_model
        gc.collect()

