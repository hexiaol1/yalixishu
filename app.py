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

# 页面基础配置
st.set_page_config(
    page_title="威远地区地层压力系数随机森林预测平台",
    page_icon="🛢️",
    layout="wide"
)

st.title("🛢️ 威远地区连续地层压力系数预测平台 (Random Forest)")
st.markdown("基于常规测井曲线构建随机森林回归模型，支持**超参数自动优选**、**特征敏感性分析**及**单井剖面预测**。")

# 初始化 Session State
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = []
if "target_col" not in st.session_state:
    st.session_state.target_col = ""
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "depth_test" not in st.session_state:
    st.session_state.depth_test = None

# 生成威远地区模拟测井数据（方便无数据时直接测试平台功能）
@st.cache_data
def generate_mock_weiyuan_data(n_samples=600):
    np.random.seed(42)
    depth = np.linspace(2500, 3100, n_samples)
    gr = np.random.normal(120, 35, n_samples) + 30 * np.sin((depth - 2500) / 50)
    ac = np.random.normal(75, 8, n_samples) + 12 * np.sin((depth - 2500) / 80)
    den = 2.65 - 0.004 * (ac - 70) + np.random.normal(0, 0.03, n_samples)
    cnl = np.random.normal(15, 4, n_samples) + 0.1 * (gr - 120)
    rt = 10 ** (np.random.normal(1.8, 0.4, n_samples))
    
    # 模拟威远龙马溪组超压特征：高GR、高AC、低DEN对应高压力系数(1.3~2.05)
    pressure_coeff = (
        1.15
        + 0.012 * (ac - 65)
        - 0.45 * (den - 2.60)
        + 0.0015 * (gr - 90)
        + 0.05 * np.log10(rt)
        + np.random.normal(0, 0.03, n_samples)
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

# ==================== 侧边栏：数据加载与字段映射 ====================
with st.sidebar:
    st.header("1. 数据接入与特征配置")
    data_source = st.radio("选择数据源", ["上传本地测井数据 (CSV/Excel)", "使用威远模拟测井数据测试"])
    
    df = None
    if data_source == "上传本地测井数据 (CSV/Excel)":
        uploaded_file = st.file_uploader("上传训练集文件", type=["csv", "xlsx"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
    else:
        df = generate_mock_weiyuan_data()
        st.info("已加载模拟数据（深度 2500m - 3100m，含 GR/AC/DEN/CNL/RT 及压力系数）")

    if df is not None:
        all_cols = df.columns.tolist()
        # 自动识别深度列
        default_depth = [c for c in all_cols if "DEPTH" in c.upper() or "深度" in c]
        depth_col = st.selectbox("选择深度列 (可选，用于剖面绘图)", ["无"] + all_cols, 
                                 index=(all_cols.index(default_depth[0]) + 1) if default_depth else 0)
        
        target_col = st.selectbox("选择标签列 (压力系数)", all_cols, index=len(all_cols)-1)
        
        exclude_cols = [target_col] + ([depth_col] if depth_col != "无" else [])
        candidate_features = [c for c in all_cols if c not in exclude_cols]
        feature_cols = st.multiselect("选择输入特征 (测井曲线)", candidate_features, default=candidate_features)
        
        log_rt_cols = st.multiselect("选择需取对数 (log10) 的电阻率曲线", feature_cols, 
                                     default=[c for c in feature_cols if c.upper() in ["RT", "RD", "RLLD", "ILD"]])
        
        test_size = st.slider("测试集比例", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        split_mode = st.selectbox("数据集划分方式", ["随机划分 (Shuffle)", "按深度顺序划分 (模拟盲井预测)"])

if df is None:
    st.warning("👈 请在左侧边栏上传威远地区测井数据或选择模拟数据进行体验。")
    st.stop()

# 数据预处理函数
def preprocess_features(data_df, features, log_cols):
    X = data_df[features].copy()
    for col in log_cols:
        if col in X.columns:
            X[col] = np.log10(np.clip(X[col], 1e-3, None))
    return X

# ==================== 主界面标签页 ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 数据概览与相关性", 
    "⚙️ 超参数优选与模型训练", 
    "🔍 特征重要性分析", 
    "🚀 未知井段批量预测"
])

# -------------------- TAB 1: 数据概览 --------------------
with tab1:
    st.subheader("样本数据预览与统计描述")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.dataframe(df.head(15), use_container_width=True)
    with col2:
        st.dataframe(df[feature_cols + [target_col]].describe().T[["count", "mean", "std", "min", "50%", "max"]], use_container_width=True)
    
    st.subheader("测井曲线与压力系数相关性矩阵 (Pearson)")
    corr_df = preprocess_features(df, feature_cols, log_rt_cols)
    corr_df[target_col] = df[target_col]
    corr_matrix = corr_df.corr()
    fig_corr = px.imshow(
        corr_matrix, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# -------------------- TAB 2: 超参数优选与训练 --------------------
with tab2:
    st.subheader("随机森林超参数寻优配置")
    
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        search_method = st.selectbox("寻优策略", ["随机搜索 (RandomizedSearchCV - 推荐)", "网格搜索 (GridSearchCV)", "手动指定参数"])
        cv_folds = st.number_input("交叉验证折数 (K-Fold)", min_value=3, max_value=10, value=5)
    with col_cfg2:
        n_estimators_range = st.multiselect("决策树数量 (n_estimators)", [50, 100, 200, 300, 500], default=[100, 200, 300])
        max_depth_range = st.multiselect("最大树深 (max_depth)", [None, 5, 10, 15, 20, 30], default=[None, 10, 20])
    with col_cfg3:
        min_samples_split_range = st.multiselect("内部节点最小划分样本数 (min_samples_split)", [2, 5, 10], default=[2, 5])
        min_samples_leaf_range = st.multiselect("叶子节点最小样本数 (min_samples_leaf)", [1, 2, 4, 8], default=[1, 2, 4])
        if search_method.startswith("随机搜索"):
            n_iter = st.number_input("随机采样组合次数 (n_iter)", min_value=5, max_value=100, value=15)

    if st.button("🚀 开始超参数寻优与模型训练", type="primary"):
        if len(feature_cols) == 0:
            st.error("请至少选择一个输入测井特征！")
        else:
            # 清理缺失值
            clean_cols = feature_cols + [target_col] + ([depth_col] if depth_col != "无" else [])
            df_clean = df[clean_cols].dropna().reset_index(drop=True)
            
            X_all = preprocess_features(df_clean, feature_cols, log_rt_cols)
            y_all = df_clean[target_col]
            depth_all = df_clean[depth_col] if depth_col != "无" else pd.Series(np.arange(len(df_clean)))
            
            if split_mode == "随机划分 (Shuffle)":
                X_train, X_test, y_train, y_test, _, depth_test = train_test_split(
                    X_all, y_all, depth_all, test_size=test_size, random_state=42
                )
            else:
                split_idx = int(len(df_clean) * (1 - test_size))
                X_train, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
                y_train, y_test = y_all.iloc[:split_idx], y_all.iloc[split_idx:]
                depth_test = depth_all.iloc[split_idx:]

            param_dist = {
                "n_estimators": n_estimators_range if n_estimators_range else [100],
                "max_depth": max_depth_range if max_depth_range else [None],
                "min_samples_split": min_samples_split_range if min_samples_split_range else [2],
                "min_samples_leaf": min_samples_leaf_range if min_samples_leaf_range else [1],
                "max_features": ["sqrt", "log2", 1.0]
            }

            with st.spinner("正在执行交叉验证与超参数寻优，请稍候..."):
                rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                
                if search_method.startswith("随机搜索"):
                    search = RandomizedSearchCV(
                        rf_base, param_distributions=param_dist, n_iter=n_iter,
                        cv=kf, scoring="r2", n_jobs=-1, random_state=42
                    )
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    best_params = search.best_params_
                    cv_results = pd.DataFrame(search.cv_results_)
                elif search_method.startswith("网格搜索"):
                    search = GridSearchCV(
                        rf_base, param_grid=param_dist,
                        cv=kf, scoring="r2", n_jobs=-1
                    )
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    best_params = search.best_params_
                    cv_results = pd.DataFrame(search.cv_results_)
                else:
                    best_params = {k: v[0] for k, v in param_dist.items()}
                    best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
                    best_model.fit(X_train, y_train)
                    cv_results = None

                # 保存状态
                st.session_state.best_model = best_model
                st.session_state.feature_cols = feature_cols
                st.session_state.log_rt_cols = log_rt_cols
                st.session_state.target_col = target_col
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.depth_test = depth_test
                st.session_state.best_params = best_params
                st.session_state.cv_results = cv_results

            st.success("✅ 模型训练与超参数优选完成！")

    # 展示训练结果
    if st.session_state.best_model is not None:
        model = st.session_state.best_model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        depth_test = st.session_state.depth_test
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        st.markdown("### 1. 最优超参数组合")
        st.json(st.session_state.best_params)

        st.markdown("### 2. 测试集精度评估指标")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("决定系数 (R²)", f"{r2:.4f}")
        m2.metric("均方根误差 (RMSE)", f"{rmse:.4f}")
        m3.metric("平均绝对误差 (MAE)", f"{mae:.4f}")
        m4.metric("平均相对误差 (MAPE)", f"{mape:.2f}%")

        st.markdown("### 3. 预测效果可视化（交会图 & 测井深度剖面对比）")
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("真实值 vs 预测值交会图", "测井深度剖面连续压力系数对比"),
            column_widths=[0.45, 0.55]
        )

        # 左图：交会图
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(x=y_test, y=y_pred, mode="markers", name="测试样本",
                       marker=dict(color="#1f77b4", opacity=0.7)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode="lines",
                       name="1:1 理想线", line=dict(color="red", dash="dash")),
            row=1, col=1
        )

        # 右图：测井深度道（按深度排序）
        sort_idx = np.argsort(depth_test.values)
        d_sorted = depth_test.values[sort_idx]
        yt_sorted = y_test.values[sort_idx]
        yp_sorted = y_pred[sort_idx]

        fig.add_trace(
            go.Scatter(x=yt_sorted, y=d_sorted, mode="lines", name="实测/参考压力系数",
                       line=dict(color="black", width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=yp_sorted, y=d_sorted, mode="lines", name="RF预测压力系数",
                       line=dict(color="#ff7f0e", width=2, dash="dot")),
            row=1, col=2
        )

        fig.update_xaxes(title_text="实测压力系数", row=1, col=1)
        fig.update_yaxes(title_text="预测压力系数", row=1, col=1)
        fig.update_xaxes(title_text="压力系数", row=1, col=2)
        fig.update_yaxes(title_text="深度 (m)", autorange="reversed", row=1, col=2)  # 测井深度轴向下增大
        fig.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        if st.session_state.cv_results is not None:
            with st.expander("查看超参数寻优交叉验证明细表 (Top 10)"):
                cv_df = st.session_state.cv_results.sort_values("rank_test_score").head(10)
                display_cols = ["rank_test_score", "mean_test_score", "std_test_score", "params"]
                st.dataframe(cv_df[display_cols], use_container_width=True)

# -------------------- TAB 3: 特征重要性分析 --------------------
with tab3:
    st.subheader("测井曲线特征重要性评估")
    if st.session_state.best_model is None:
        st.info("请先在【超参数优选与模型训练】标签页完成模型训练。")
    else:
        model = st.session_state.best_model
        features = st.session_state.feature_cols
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        col_imp1, col_imp2 = st.columns(2)

        # 1. 基尼不纯度/方差减少重要性 (MDI)
        with col_imp1:
            st.markdown("#### 1. 基于树分裂方差减少的重要性 (MDI)")
            mdi_df = pd.DataFrame({
                "测井曲线": features,
                "重要性权重": model.feature_importances_
            }).sort_values("重要性权重", ascending=True)

            fig_mdi = px.bar(
                mdi_df, x="重要性权重", y="测井曲线", orientation="h",
                text_auto=".3f", color="重要性权重", color_continuous_scale="Blues"
            )
            fig_mdi.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_mdi, use_container_width=True)

        # 2. 排列重要性 (Permutation Importance)
        with col_imp2:
            st.markdown("#### 2. 基于测试集扰动的排列重要性 (Permutation)")
            perm_res = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
            perm_df = pd.DataFrame({
                "测井曲线": features,
                "R²下降均值": perm_res.importances_mean,
                "标准差": perm_res.importances_std
            }).sort_values("R²下降均值", ascending=True)

            fig_perm = px.bar(
                perm_df, x="R²下降均值", y="测井曲线", orientation="h",
                error_x="标准差", text_auto=".3f", color="R²下降均值", color_continuous_scale="Oranges"
            )
            fig_perm.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_perm, use_container_width=True)

        st.markdown("""
        **地球物理测井解释参考：**
        * **MDI 重要性**反映了各测井曲线在构建回归树分裂节点时的累计贡献度；
        * **排列重要性**通过随机打乱单条测井曲线观察测试集 $R^2$ 的衰减幅度，更能有效排除共线性干扰（如 `AC` 与 `DEN` 的强负相关），识别真正控制威远地层超压的主控测井参数。
        """)

# -------------------- TAB 4: 未知井段预测与模型导出 --------------------
with tab4:
    st.subheader("新井/未知井段压力系数预测与模型下载")
    if st.session_state.best_model is None:
        st.info("请先完成模型训练，再进行新数据预测或模型导出。")
    else:
        # 导出训练好的模型
        model_buffer = io.BytesIO()
        joblib.dump({
            "model": st.session_state.best_model,
            "features": st.session_state.feature_cols,
            "log_cols": st.session_state.log_rt_cols
        }, model_buffer)
        st.download_button(
            label="💾 下载训练好的随机森林模型 (.pkl)",
            data=model_buffer.getvalue(),
            file_name="weiyuan_rf_pressure_model.pkl",
            mime="application/octet-stream"
        )

        st.divider()
        st.markdown("#### 上传待预测井的测井曲线文件 (CSV/Excel)")
        pred_file = st.file_uploader("需包含与训练集一致的测井曲线列名", type=["csv", "xlsx"], key="pred_uploader")
        if pred_file is not None:
            pred_df = pd.read_csv(pred_file) if pred_file.name.endswith(".csv") else pd.read_excel(pred_file)
            req_cols = st.session_state.feature_cols
            missing_cols = [c for c in req_cols if c not in pred_df.columns]
            
            if missing_cols:
                st.error(f"上传文件缺少以下必要测井曲线列：{missing_cols}")
            else:
                X_new = preprocess_features(pred_df, req_cols, st.session_state.log_rt_cols)
                # 处理可能存在的空值
                valid_mask = ~X_new.isna().any(axis=1)
                pred_df["Predicted_Pressure_Coeff"] = np.nan
                pred_df.loc[valid_mask, "Predicted_Pressure_Coeff"] = np.round(
                    st.session_state.best_model.predict(X_new[valid_mask]), 4
                )
                
                st.success(f"预测完成！共成功预测 {valid_mask.sum()} 个采样点。")
                st.dataframe(pred_df.head(20), use_container_width=True)
                
                csv_out = pred_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 导出预测结果 (CSV)",
                    data=csv_out,
                    file_name="weiyuan_predicted_pressure.csv",
                    mime="text/csv"
                )
