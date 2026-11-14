import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import io

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

# ==================== 1. 页面配置与自定义 CSS 美化 ====================
st.set_page_config(
    page_title="威远页岩气藏地层压力系数智能预测平台",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入现代能源仪表盘 CSS 样式
st.markdown("""
<style>
    /* 全局字体与背景微调 */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    /* 顶部渐变主横幅 */
    .hero-banner {
        background: linear-gradient(135deg, #0B2545 0%, #134074 60%, #1D63B8 100%);
        padding: 1.6rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(11, 37, 69, 0.15);
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .hero-banner p {
        margin: 0.4rem 0 0 0;
        font-size: 0.95rem;
        color: #CBD5E1;
    }
    /* KPI 指标卡片美化 */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #0F52BA;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        transition: transform 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(15, 23, 42, 0.08);
    }
    /* 标签页样式优化 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #E2E8F0;
        padding: 4px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #0F52BA !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# 顶部主横幅
st.markdown("""
<div class="hero-banner">
    <h1>🛢️ 威远地区连续地层压力系数智能预测平台</h1>
    <p>Weiyuan Shale Gas Formation Pressure Prediction · 随机森林集成学习 · 超参数自动寻优 · 测井多道联动解释</p>
</div>
""", unsafe_allow_html=True)

# 初始化 Session State
for key, default_val in [
    ("best_model", None), ("feature_cols", []), ("log_rt_cols", []),
    ("target_col", ""), ("X_test", None), ("y_test", None),
    ("depth_test", None), ("df_test_raw", None), ("best_params", {}), ("cv_results", None)
]:
    if key not in st.session_state:
        st.session_state[key] = default_val

# 模拟威远地区五峰-龙马溪组测井数据
@st.cache_data
def generate_mock_weiyuan_data(n_samples=600):
    np.random.seed(42)
    depth = np.linspace(2600, 3100, n_samples)
    # 模拟底部龙马溪组优质页岩段（高GR、高AC、低DEN、超压特征明显）
    shale_zone = np.exp(-((depth - 2950) / 90) ** 2)
    gr = 95 + 85 * shale_zone + 15 * np.sin((depth - 2600) / 30) + np.random.normal(0, 8, n_samples)
    ac = 68 + 24 * shale_zone + 5 * np.cos((depth - 2600) / 40) + np.random.normal(0, 2.5, n_samples)
    den = 2.68 - 0.16 * shale_zone - 0.002 * (ac - 70) + np.random.normal(0, 0.02, n_samples)
    cnl = 12 + 10 * shale_zone + np.random.normal(0, 1.8, n_samples)
    rt = 10 ** (1.4 + 0.9 * shale_zone + np.random.normal(0, 0.25, n_samples))

    pressure_coeff = (
        1.12
        + 0.015 * (ac - 65)
        - 0.65 * (den - 2.65)
        + 0.0018 * (gr - 90)
        + 0.08 * np.log10(rt)
        + np.random.normal(0, 0.025, n_samples)
    )
    pressure_coeff = np.clip(pressure_coeff, 1.05, 2.15)

    return pd.DataFrame({
        "DEPTH": np.round(depth, 2),
        "GR": np.round(gr, 2),
        "AC": np.round(ac, 2),
        "DEN": np.round(den, 3),
        "CNL": np.round(cnl, 2),
        "RT": np.round(rt, 2),
        "Pressure_Coeff": np.round(pressure_coeff, 3)
    })

# ==================== 2. 侧边栏配置 ====================
with st.sidebar:
    st.markdown("### 🗂️ 数据接入与测井配置")
    data_source = st.radio(
        "数据源选择",
        ["使用威远模拟测井数据体验", "上传本地测井数据 (CSV/Excel)"],
        label_visibility="collapsed"
    )

    df = None
    if data_source == "上传本地测井数据 (CSV/Excel)":
        uploaded_file = st.file_uploader("拖拽或选择测井数据文件", type=["csv", "xlsx"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    else:
        df = generate_mock_weiyuan_data()
        st.caption("✨ 已加载威远模拟剖面（2600m-3100m，含超压页岩段响应）")

    if df is not None:
        st.divider()
        all_cols = df.columns.tolist()
        default_depth = [c for c in all_cols if "DEPTH" in c.upper() or "深度" in c]
        depth_col = st.selectbox(
            "📏 深度索引列", ["无"] + all_cols,
            index=(all_cols.index(default_depth[0]) + 1) if default_depth else 0
        )
        target_col = st.selectbox("🎯 预测标签列 (压力系数)", all_cols, index=len(all_cols) - 1)

        exclude_cols = [target_col] + ([depth_col] if depth_col != "无" else [])
        candidate_features = [c for c in all_cols if c not in exclude_cols]
        feature_cols = st.multiselect("📈 输入测井曲线特征", candidate_features, default=candidate_features)

        log_rt_cols = st.multiselect(
            "🔢 电阻率对数变换 (log10)", feature_cols,
            default=[c for c in feature_cols if c.upper() in ["RT", "RD", "RLLD", "ILD"]]
        )

        st.divider()
        st.markdown("### 🧪 数据集划分策略")
        test_size = st.slider("测试集比例", 0.1, 0.4, 0.25, 0.05)
        split_mode = st.selectbox(
            "划分方式",
            ["按深度连续划分 (模拟盲井预测)", "随机采样划分 (Shuffle)"]
        )

if df is None:
    st.info("👈 请在左侧边栏上传测井数据或选择模拟数据进行体验。")
    st.stop()

def preprocess_features(data_df, features, log_cols):
    X = data_df[features].copy()
    for col in log_cols:
        if col in X.columns:
            X[col] = np.log10(np.clip(X[col], 1e-3, None))
    return X

# ==================== 3. 主界面四大模块 ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 测井数据探索 (EDA)",
    "⚙️ 超参数优选与模型评估",
    "🔍 特征敏感性与主控因素",
    "🚀 未知井段批量预测与导出"
])

# -------------------- TAB 1: 数据探索 --------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总采样点数", f"{len(df):,}")
    c2.metric("输入测井曲线数", f"{len(feature_cols)} 条")
    c3.metric("压力系数均值", f"{df[target_col].mean():.3f}")
    c4.metric("最大超压系数", f"{df[target_col].max():.3f}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.15, 1])

    with col_left:
        st.markdown("#### 相关性热力图 (测井响应 vs 压力系数)")
        corr_df = preprocess_features(df, feature_cols, log_rt_cols)
        corr_df[target_col] = df[target_col]
        corr_matrix = corr_df.corr()

        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1
        )
        fig_corr.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=420,
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_right:
        st.markdown("#### 压力系数分布与超压区间统计")
        fig_hist = px.histogram(
            df, x=target_col, nbins=30, marginal="box",
            color_discrete_sequence=["#0F52BA"], opacity=0.85
        )
        # 添加常压/超压分界参考线
        fig_hist.add_vline(x=1.2, line_dash="dash", line_color="#10B981", annotation_text="弱超压 (1.2)")
        fig_hist.add_vline(x=1.5, line_dash="dash", line_color="#EF4444", annotation_text="强超压 (1.5)")
        fig_hist.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=420,
            xaxis_title="地层压力系数",
            yaxis_title="采样点频数",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#F8FAFC"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with st.expander("📋 查看原始测井数据表与统计摘要", expanded=False):
        t_col1, t_col2 = st.columns([1.2, 1])
        with t_col1:
            st.dataframe(df.head(20), use_container_width=True)
        with t_col2:
            st.dataframe(df[feature_cols + [target_col]].describe().T[["count", "mean", "std", "min", "50%", "max"]], use_container_width=True)

# -------------------- TAB 2: 超参数优选与多道测井评估 --------------------
with tab2:
    with st.container(border=True):
        st.markdown("#### 🛠️ 随机森林超参数搜索空间配置")
        cfg1, cfg2, cfg3, cfg4 = st.columns(4)
        with cfg1:
            search_method = st.selectbox("寻优算法", ["随机搜索 (RandomizedSearchCV)", "网格搜索 (GridSearchCV)", "手动指定"])
            cv_folds = st.number_input("K-Fold 交叉验证折数", 3, 10, 5)
        with cfg2:
            n_estimators_range = st.multiselect("决策树数量 (n_estimators)", [50, 100, 200, 300, 500], default=[100, 200, 300])
            max_depth_range = st.multiselect("最大树深 (max_depth)", [None, 6, 10, 15, 20], default=[None, 10, 15])
        with cfg3:
            min_samples_split_range = st.multiselect("最小划分样本数", [2, 5, 10], default=[2, 5])
            min_samples_leaf_range = st.multiselect("叶子最小样本数", [1, 2, 4, 8], default=[1, 2, 4])
        with cfg4:
            if search_method.startswith("随机搜索"):
                n_iter = st.number_input("随机采样组合数 (n_iter)", 5, 60, 15)
            else:
                st.write("")
                st.caption("网格搜索将遍历所有参数笛卡尔积组合。")
            st.write("")
            run_btn = st.button("🚀 启动超参数优选与模型训练", type="primary", use_container_width=True)

    if run_btn:
        if not feature_cols:
            st.error("请至少选择一条输入测井曲线！")
        else:
            clean_cols = list(dict.fromkeys(feature_cols + [target_col] + ([depth_col] if depth_col != "无" else [])))
            df_clean = df[clean_cols].dropna().reset_index(drop=True)

            X_all = preprocess_features(df_clean, feature_cols, log_rt_cols)
            y_all = df_clean[target_col]
            depth_all = df_clean[depth_col] if depth_col != "无" else pd.Series(np.arange(len(df_clean)))

            if split_mode == "随机采样划分 (Shuffle)":
                X_train, X_test, y_train, y_test, _, depth_test, _, df_test_raw = train_test_split(
                    X_all, y_all, depth_all, df_clean, test_size=test_size, random_state=42
                )
            else:
                split_idx = int(len(df_clean) * (1 - test_size))
                X_train, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
                y_train, y_test = y_all.iloc[:split_idx], y_all.iloc[split_idx:]
                depth_test = depth_all.iloc[split_idx:]
                df_test_raw = df_clean.iloc[split_idx:]

            param_dist = {
                "n_estimators": n_estimators_range or [100],
                "max_depth": max_depth_range or [None],
                "min_samples_split": min_samples_split_range or [2],
                "min_samples_leaf": min_samples_leaf_range or [1],
                "max_features": ["sqrt", "log2", 1.0]
            }

            with st.spinner("正在执行交叉验证与超参数空间寻优..."):
                rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

                if search_method.startswith("随机搜索"):
                    search = RandomizedSearchCV(rf_base, param_dist, n_iter=n_iter, cv=kf, scoring="r2", n_jobs=-1, random_state=42)
                    search.fit(X_train, y_train)
                    best_model, best_params, cv_results = search.best_estimator_, search.best_params_, pd.DataFrame(search.cv_results_)
                elif search_method.startswith("网格搜索"):
                    search = GridSearchCV(rf_base, param_dist, cv=kf, scoring="r2", n_jobs=-1)
                    search.fit(X_train, y_train)
                    best_model, best_params, cv_results = search.best_estimator_, search.best_params_, pd.DataFrame(search.cv_results_)
                else:
                    best_params = {k: v[0] for k, v in param_dist.items()}
                    best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1).fit(X_train, y_train)
                    cv_results = None

                st.session_state.update({
                    "best_model": best_model, "feature_cols": feature_cols, "log_rt_cols": log_rt_cols,
                    "target_col": target_col, "X_test": X_test, "y_test": y_test,
                    "depth_test": depth_test, "df_test_raw": df_test_raw,
                    "best_params": best_params, "cv_results": cv_results
                })
            st.toast("模型训练与超参数寻优完成！", icon="🎉")

    if st.session_state.best_model is not None:
        model = st.session_state.best_model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        depth_test = st.session_state.depth_test
        df_test_raw = st.session_state.df_test_raw

        # 计算所有决策树的预测均值与标准差（衡量预测不确定性）
        all_tree_preds = np.array([tree.predict(X_test.values) for tree in model.estimators_])
        y_pred = np.mean(all_tree_preds, axis=0)
        y_std = np.std(all_tree_preds, axis=0)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # KPI 卡片栏
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("测试集决定系数 (R²)", f"{r2:.4f}", delta="拟合优度")
        m2.metric("均方根误差 (RMSE)", f"{rmse:.4f}")
        m3.metric("平均绝对误差 (MAE)", f"{mae:.4f}")
        m4.metric("平均相对误差 (MAPE)", f"{mape:.2f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        # ---- 专业测井多道联动剖面图（核心视觉升级） ----
        st.markdown("#### 🧭 测试井段多道测井响应与压力系数置信带综合剖面")
        sort_idx = np.argsort(depth_test.values)
        d_s = depth_test.values[sort_idx]
        yt_s = y_test.values[sort_idx]
        yp_s = y_pred[sort_idx]
        ystd_s = y_std[sort_idx]
        df_raw_s = df_test_raw.iloc[sort_idx]

        # 自动匹配可用常规测井道
        has_gr = any(c.upper() == "GR" for c in df_raw_s.columns)
        has_ac = any(c.upper() in ["AC", "DT"] for c in df_raw_s.columns)
        has_den = any(c.upper() in ["DEN", "RHOB"] for c in df_raw_s.columns)
        has_rt = any(c.upper() in ["RT", "RD"] for c in df_raw_s.columns)

        tracks = []
        if has_gr:
            tracks.append("自然伽马 (GR)")
        if has_ac or has_den:
            tracks.append("孔隙度曲线 (AC/DEN)")
        if has_rt:
            tracks.append("深电阻率 (RT)")
        tracks.append("压力系数预测 & ±1σ置信带")

        n_tracks = len(tracks)
        widths = [0.18] * (n_tracks - 1) + [1.0 - 0.18 * (n_tracks - 1)]

        fig_log = make_subplots(
            rows=1, cols=n_tracks,
            shared_yaxes=True,
            horizontal_spacing=0.025,
            subplot_titles=tracks,
            column_widths=widths
        )

        col_idx = 1
        # 道 1：GR 填充道
        if has_gr:
            gr_col = [c for c in df_raw_s.columns if c.upper() == "GR"][0]
            fig_log.add_trace(go.Scatter(
                x=df_raw_s[gr_col], y=d_s, name="GR (API)",
                line=dict(color="#15803D", width=1.5),
                fill="tozerox", fillcolor="rgba(34, 197, 94, 0.18)"
            ), row=1, col=col_idx)
            fig_log.update_xaxes(title_text="API", row=1, col=col_idx, showgrid=True, gridcolor="#E2E8F0")
            col_idx += 1

        # 道 2：AC / DEN 道
        if has_ac or has_den:
            if has_ac:
                ac_col = [c for c in df_raw_s.columns if c.upper() in ["AC", "DT"]][0]
                fig_log.add_trace(go.Scatter(
                    x=df_raw_s[ac_col], y=d_s, name="AC (μs/ft)",
                    line=dict(color="#2563EB", width=1.5)
                ), row=1, col=col_idx)
            elif has_den:
                den_col = [c for c in df_raw_s.columns if c.upper() in ["DEN", "RHOB"]][0]
                fig_log.add_trace(go.Scatter(
                    x=df_raw_s[den_col], y=d_s, name="DEN (g/cm³)",
                    line=dict(color="#9333EA", width=1.5)
                ), row=1, col=col_idx)
            fig_log.update_xaxes(title_text="AC / DEN", row=1, col=col_idx, showgrid=True, gridcolor="#E2E8F0")
            col_idx += 1

        # 道 3：RT 对数道
        if has_rt:
            rt_col = [c for c in df_raw_s.columns if c.upper() in ["RT", "RD"]][0]
            fig_log.add_trace(go.Scatter(
                x=df_raw_s[rt_col], y=d_s, name="RT (Ω·m)",
                line=dict(color="#D97706", width=1.5)
            ), row=1, col=col_idx)
            fig_log.update_xaxes(title_text="Ω·m (Log)", type="log", row=1, col=col_idx, showgrid=True, gridcolor="#E2E8F0")
            col_idx += 1

        # 主道：压力系数对比 + 置信区间带
        fig_log.add_trace(go.Scatter(
            x=yp_s + ystd_s, y=d_s, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip"
        ), row=1, col=col_idx)
        fig_log.add_trace(go.Scatter(
            x=yp_s - ystd_s, y=d_s, mode="lines", line=dict(width=0),
            fill="tonextx", fillcolor="rgba(15, 82, 186, 0.18)",
            name="预测不确定性 (±1σ)"
        ), row=1, col=col_idx)
        fig_log.add_trace(go.Scatter(
            x=yt_s, y=d_s, mode="lines", name="实测/参考压力系数",
            line=dict(color="#0F172A", width=2.2)
        ), row=1, col=col_idx)
        fig_log.add_trace(go.Scatter(
            x=yp_s, y=d_s, mode="lines", name="RF 预测压力系数",
            line=dict(color="#EF4444", width=2.2, dash="dash")
        ), row=1, col=col_idx)
        fig_log.update_xaxes(title_text="地层压力系数", row=1, col=col_idx, showgrid=True, gridcolor="#E2E8F0")

        fig_log.update_yaxes(title_text="深度 (m)", autorange="reversed", row=1, col=1, showgrid=True, gridcolor="#E2E8F0")
        fig_log.update_layout(
            height=560,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            hovermode="y unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1),
            margin=dict(l=50, r=20, t=65, b=40)
        )
        st.plotly_chart(fig_log, use_container_width=True)

        # ---- 下方双图：交会图 + 预测残差分布 ----
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            st.markdown("#### 🎯 实测 vs 预测交会图（颜色映射误差绝对值）")
            abs_err = np.abs(y_test - y_pred)
            min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            fig_scatter = px.scatter(
                x=y_test, y=y_pred, color=abs_err,
                color_continuous_scale="Viridis_r",
                labels={"x": "实测压力系数", "y": "RF 预测压力系数", "color": "绝对误差"}
            )
            fig_scatter.add_trace(go.Scatter(
                x=[min_v, max_v], y=[min_v, max_v], mode="lines",
                name="1:1 理想拟合线", line=dict(color="#EF4444", dash="dash", width=2)
            ))
            fig_scatter.update_layout(height=380, plot_bgcolor="#F8FAFC", paper_bgcolor="#FFFFFF")
            st.plotly_chart(fig_scatter, use_container_width=True)

        with c_sub2:
            st.markdown("#### 📉 预测残差分布诊断 (Residuals)")
            residuals = y_test - y_pred
            fig_res = px.histogram(
                x=residuals, nbins=25, marginal="rug",
                color_discrete_sequence=["#0D9488"],
                labels={"x": "残差 (实测 - 预测)", "y": "频数"}
            )
            fig_res.add_vline(x=0, line_dash="dash", line_color="#EF4444", line_width=2)
            fig_res.update_layout(height=380, plot_bgcolor="#F8FAFC", paper_bgcolor="#FFFFFF")
            st.plotly_chart(fig_res, use_container_width=True)

        with st.expander("🏆 查看最优超参数 JSON 与交叉验证排行榜"):
            st.json(st.session_state.best_params)
            if st.session_state.cv_results is not None:
                cv_df = st.session_state.cv_results.sort_values("rank_test_score").head(10)
                st.dataframe(cv_df[["rank_test_score", "mean_test_score", "std_test_score", "params"]], use_container_width=True)

# -------------------- TAB 3: 特征敏感性分析 --------------------
with tab3:
    if st.session_state.best_model is None:
        st.info("请先在【超参数优选与模型评估】标签页完成模型训练。")
    else:
        model = st.session_state.best_model
        features = st.session_state.feature_cols
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        mdi_scores = model.feature_importances_
        perm_res = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        perm_scores = np.maximum(perm_res.importances_mean, 0)

        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.markdown("#### 1. 树分裂方差减少重要性 (MDI)")
            mdi_df = pd.DataFrame({"测井曲线": features, "权重": mdi_scores}).sort_values("权重", ascending=True)
            fig_mdi = px.bar(
                mdi_df, x="权重", y="测井曲线", orientation="h", text_auto=".3f",
                color="权重", color_continuous_scale="Blues"
            )
            fig_mdi.update_layout(height=400, showlegend=False, plot_bgcolor="#F8FAFC", paper_bgcolor="#FFFFFF")
            st.plotly_chart(fig_mdi, use_container_width=True)

        with col_i2:
            st.markdown("#### 2. 排列扰动重要性 (Permutation R² 衰减)")
            perm_df = pd.DataFrame({
                "测井曲线": features,
                "R²衰减": perm_scores,
                "误差": perm_res.importances_std
            }).sort_values("R²衰减", ascending=True)
            fig_perm = px.bar(
                perm_df, x="R²衰减", y="测井曲线", orientation="h", error_x="误差", text_auto=".3f",
                color="R²衰减", color_continuous_scale="Tealgrn"
            )
            fig_perm.update_layout(height=400, showlegend=False, plot_bgcolor="#F8FAFC", paper_bgcolor="#FFFFFF")
            st.plotly_chart(fig_perm, use_container_width=True)

        # 归一化双指标对比雷达图
        if len(features) >= 3:
            st.markdown("#### 🕸️ 双维度归一化特征敏感性雷达对比")
            mdi_norm = mdi_scores / (mdi_scores.sum() + 1e-9)
            perm_norm = perm_scores / (perm_scores.sum() + 1e-9)

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=mdi_norm, theta=features, fill="toself", name="MDI 归一化权重",
                line_color="#0F52BA", fillcolor="rgba(15, 82, 186, 0.2)"
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=perm_norm, theta=features, fill="toself", name="Permutation 归一化权重",
                line_color="#0D9488", fillcolor="rgba(13, 148, 136, 0.2)"
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, max(mdi_norm.max(), perm_norm.max()) * 1.15])),
                height=420, paper_bgcolor="#FFFFFF"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

# -------------------- TAB 4: 未知井段批量预测 --------------------
with tab4:
    if st.session_state.best_model is None:
        st.info("请先完成模型训练，再进行新数据预测或模型导出。")
    else:
        with st.container(border=True):
            st.markdown("#### 💾 模型资产导出")
            model_buffer = io.BytesIO()
            joblib.dump({
                "model": st.session_state.best_model,
                "features": st.session_state.feature_cols,
                "log_cols": st.session_state.log_rt_cols
            }, model_buffer)
            st.download_button(
                label="下载训练完成的随机森林模型 (.pkl)",
                data=model_buffer.getvalue(),
                file_name="weiyuan_rf_pressure_model.pkl",
                mime="application/octet-stream",
                type="primary"
            )

        with st.container(border=True):
            st.markdown("#### 📂 上传待预测井测井数据 (CSV / Excel)")
            pred_file = st.file_uploader("需包含与训练集一致的测井曲线列", type=["csv", "xlsx"], key="pred_uploader")
            if pred_file is not None:
                pred_df = pd.read_csv(pred_file) if pred_file.name.endswith(".csv") else pd.read_excel(pred_file)
                req_cols = st.session_state.feature_cols
                missing_cols = [c for c in req_cols if c not in pred_df.columns]

                if missing_cols:
                    st.error(f"缺少必要测井曲线列：{missing_cols}")
                else:
                    X_new = preprocess_features(pred_df, req_cols, st.session_state.log_rt_cols)
                    valid_mask = ~X_new.isna().any(axis=1)
                    pred_df["Predicted_Pressure_Coeff"] = np.nan
                    pred_df.loc[valid_mask, "Predicted_Pressure_Coeff"] = np.round(
                        st.session_state.best_model.predict(X_new[valid_mask]), 4
                    )

                    st.success(f"预测完成！共生成 {valid_mask.sum()} 个有效采样点预测结果。")
                    st.dataframe(pred_df.head(20), use_container_width=True)

                    csv_out = pred_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="📥 导出完整预测结果 (CSV)",
                        data=csv_out,
                        file_name="weiyuan_predicted_pressure.csv",
                        mime="text/csv"
                    )
