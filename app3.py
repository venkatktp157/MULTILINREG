import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.set_page_config(layout="wide", page_title="JMP-Style Regression Control")

st.title("🎛️ JMP-Inspired Stepwise Regression Platform")
st.caption("An interactive teaching interface for exploring Ordinary Least Squares (OLS) and feature selection logic.")

# 1. FILE UPLOAD AND PROCESSING VIA SIDEBAR
st.sidebar.header("📁 Data Source Configuration")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
        
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("The uploaded dataset must contain at least 2 numeric columns.")
        st.stop()

    target_var = st.sidebar.selectbox("🎯 Select Target Variable (Y)", numeric_cols)
    possible_predictors = [col for col in numeric_cols if col != target_var]

    if not possible_predictors:
        st.error("No available numeric predictors left.")
        st.stop()

    # FIXED BASELINE CLEANING
    columns_to_sync = [target_var] + possible_predictors
    df_clean = df[columns_to_sync].dropna()

    # --- FIX: INITIALIZE STATE KEY FOR EVERY FEATURE ---
    for feature in possible_predictors:
        state_key = f"cb_{feature}"
        if state_key not in st.session_state:
            st.session_state[state_key] = False

    # UPPER APP LAYOUT
    col_left, col_right = st.columns([2, 3])

    with col_left:
        st.subheader("📊 Data Viewer")
        st.write(f"**Active Dataset:** `{uploaded_file.name}` ({len(df_clean)} rows)")
        st.dataframe(df_clean, height=320, use_container_width=True)

    with col_right:
        st.subheader("🔺 Stepwise Regression Control")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            stopping_rule = st.selectbox("Stopping Rule", ["P-value Threshold"])
        with c2:
            prob_enter = st.number_input("Prob to Enter (Alpha)", value=0.05, step=0.01)
        with c3:
            prob_leave = st.number_input("Prob to Leave (Alpha)", value=0.10, step=0.01)
            
        c4, c5 = st.columns(2)
        with c4:
            direction = st.selectbox("Direction", ["Forward", "Backward", "Stepwise"])
        with col_right:
            # Create explicit clear buttons to reset states if needed
            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                run_auto = st.button("🚀 Run Stepwise Search", type="primary", use_container_width=True)
            with c_btn2:
                reset_all = st.button("🔄 Clear All Checks", type="secondary", use_container_width=True)
            
        st.markdown("---")
        
        # Handle clear button
        if reset_all:
            for feature in possible_predictors:
                st.session_state[f"cb_{feature}"] = False
            st.rerun()
        
        # --- ROBUST STEPWISE ALGORITHM ---
        if run_auto:
            current_features = []
            y_auto = df_clean[target_var]
            
            if direction == "Forward" or direction == "Stepwise":
                while True:
                    changed = False
                    remaining = [f for f in possible_predictors if f not in current_features]
                    best_p = 1.0
                    best_feature = None
                    
                    for feature in remaining:
                        test_vars = current_features + [feature]
                        X_test = sm.add_constant(df_clean[test_vars])
                        try:
                            model_test = sm.OLS(y_auto, X_test).fit(method='pinv')
                            p_val = model_test.pvalues[feature]
                            if p_val < best_p:
                                best_p = p_val
                                best_feature = feature
                        except:
                            continue
                            
                    if best_feature and best_p < prob_enter:
                        current_features.append(best_feature)
                        changed = True
                        
                    if direction == "Stepwise" and len(current_features) > 0:
                        X_test = sm.add_constant(df_clean[current_features])
                        model_test = sm.OLS(y_auto, X_test).fit(method='pinv')
                        p_vals = model_test.pvalues.drop('const', errors='ignore')
                        worst_p = p_vals.max()
                        if worst_p > prob_leave:
                            worst_feature = p_vals.idxmax()
                            current_features.remove(worst_feature)
                            changed = True
                            
                    if not changed:
                        break
                        
            elif direction == "Backward":
                current_features = list(possible_predictors)
                while len(current_features) > 0:
                    X_test = sm.add_constant(df_clean[current_features])
                    model_test = sm.OLS(y_auto, X_test).fit(method='pinv')
                    p_vals = model_test.pvalues.drop('const', errors='ignore')
                    if len(p_vals) == 0:
                        break
                    worst_p = p_vals.max()
                    worst_feature = p_vals.idxmax()
                    
                    if worst_p > prob_leave:
                        current_features.remove(worst_feature)
                    else:
                        break
            
            # --- FIX: WRITE DIRECTLY TO RESIDENT CHECKBOX KEYS ---
            for feature in possible_predictors:
                st.session_state[f"cb_{feature}"] = (feature in current_features)
            
            st.success(f"Optimized subset configured via {direction} selection.")
            st.rerun()

        # Display Selection Matrix
        st.write("**Manual Parameter Selection Matrix**")
        include_intercept = st.checkbox("Include Intercept (Beta 0)", value=True)
        
        # --- FIX: ASSIGN THE CHECKBOX VALUE EXCLUSIVELY VIA ITS SESSION KEY ---
        selected_features = []
        num_features = len(possible_predictors)
        if num_features > 0:
            cb_cols = st.columns(min(num_features, 4))
            for idx, feature in enumerate(possible_predictors):
                col_idx = idx % 4
                state_key = f"cb_{feature}"
                with cb_cols[col_idx]:
                    if st.checkbox(f"➕ {feature}", key=state_key):
                        selected_features.append(feature)

    st.markdown("---")

    # 2. STATISTICAL ESTIMATION ENGINE
    y = df_clean[target_var]
    if len(selected_features) == 0:
        X = pd.DataFrame(np.ones(len(df_clean)), columns=["Intercept"], index=df_clean.index)
    else:
        X = df_clean[selected_features]
        if include_intercept:
            X = sm.add_constant(X)

    if len(y) < len(X.columns) + 1:
        st.warning("⚠️ Insufficient observations to estimate parameters.")
    else:
        # Compute VIF Matrix first
        vif_dict = {}
        if X.shape[1] > 1:
            for idx, col_name in enumerate(X.columns):
                try:
                    vif_dict[col_name] = variance_inflation_factor(X.values, idx)
                except Exception:
                    vif_dict[col_name] = np.nan
        else:
            for col_name in X.columns:
                vif_dict[col_name] = 1.0 if col_name != "const" else np.nan

        high_vif_features = []
        perfect_vif_features = []
        for col_name, val in vif_dict.items():
            if col_name != "const":
                if np.isnan(val) or val > 999999:
                    perfect_vif_features.append(col_name)
                elif val > 10.0:
                    high_vif_features.append(f"{col_name} (VIF: {val:.2f})")

        # 3. FULL-WIDTH LOWER REPORTING DECK
        st.subheader("📊 Regression Execution Results")
        
        if perfect_vif_features:
            st.error(f"⚠️ **Perfect Multicollinearity Warning:** {', '.join(perfect_vif_features)} are structurally identical. The regression will use pseudo-inverses to solve parameters safely.")
        elif high_vif_features:
            st.warning(f"💡 **High Multicollinearity Notice:** Predictors heavily overlap: **{', '.join(high_vif_features)}**.")

        # Fit selected model safely
        model = sm.OLS(y, X).fit(method='pinv')
        
        sse = model.ssr  
        dfe = int(model.df_resid)
        rmse = np.sqrt(model.mse_resid) if dfe > 0 else 0
        rsquare = model.rsquared
        rsquare_adj = model.rsquared_adj
        aic = model.aic
        bic = model.bic
        
        # Forecasted R2 Engine
        influence = model.get_influence()
        leverage = influence.hat_matrix_diag
        press_residuals = model.resid / (1.0 - np.clip(leverage, 0, 0.999999))
        press = np.sum(press_residuals ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        predicted_r2 = 1.0 - (press / sst) if sst > 0 else 0.0

        X_full = sm.add_constant(df_clean[possible_predictors])
        full_model = sm.OLS(y, X_full).fit(method='pinv')
        sigma_hat_sq = full_model.mse_resid
        p_count = len(selected_features)
        mallows_cp = (sse / sigma_hat_sq) - len(df_clean) + 2 * (p_count + 1) if len(selected_features) > 0 else len(df_clean)

        st.write("#### 📈 Model Diagnostics Summary")
        metrics_df = pd.DataFrame({
            "Observations (N)": [len(y)],
            "SSE": [f"{sse:,.2f}"],
            "DFE": [dfe],
            "RMSE": [f"{rmse:.4f}"],
            "RSquare": [f"{rsquare:.4f}"],
            "RSquare Adj": [f"{rsquare_adj:.4f}"],
            "Predicted R2": [f"{predicted_r2:.4f}"],
            "Cp": [f"{mallows_cp:.2f}" if not np.isnan(mallows_cp) else "N/A"],
            "AIC": [f"{aic:.2f}"],
            "BIC": [f"{bic:.2f}"]
        })
        st.table(metrics_df)

        st.write("#### 🔑 Current Estimates Table")
        param_names = model.params.index
        vif_list = [vif_dict.get(name, np.nan) for name in param_names]
        
        estimates_summary = pd.DataFrame({
            "Parameter": param_names,
            "Estimate": model.params.values,
            "Std Error": model.bse.values,
            "t Ratio": model.tvalues.values,
            "Prob > |t| (P-value)": model.pvalues.values,
            "VIF": vif_list
        })
        
        def format_vif(val):
            if np.isnan(val) or val is None: return "∞ (Perfect Collinearity)"
            if val > 999999: return "∞ (Perfect Collinearity)"
            return f"{val:.2f}"

        formatted_summary = estimates_summary.copy()
        formatted_summary["VIF"] = formatted_summary["VIF"].apply(format_vif)

        st.dataframe(
            formatted_summary.style.format({
                "Estimate": "{:.4f}",
                "Std Error": "{:.4f}",
                "t Ratio": "{:.2f}",
                "Prob > |t| (P-value)": "{:.4e}"
            }),
            use_container_width=True,
            hide_index=True
        )

        # --- NEW EDUCATION ELEMENT: ALL POSSIBLE MODELS (BEST SUBSETS) ---
        st.markdown("---")
        st.subheader("🏁 All Possible Subsets Analysis (JMP Criterion Benchmarking)")
        st.write("This exhaustive bench evaluates every single variable combination to find the true mathematical optimum. Use this to verify if the Stepwise path found the true best choice.")
        
        import itertools
        subsets_results = []
        
        # Only run if number of features is reasonable to avoid severe web latency (limit to max 8 variables)
        eval_features = possible_predictors[:8]
        
        for k in range(1, len(eval_features) + 1):
            for combo in itertools.combinations(eval_features, k):
                combo_list = list(combo)
                X_sub = sm.add_constant(df_clean[combo_list])
                sub_model = sm.OLS(y, X_sub).fit(method='pinv')
                
                # Calculate Mallows Cp for this subset combo
                sub_cp = (sub_model.ssr / sigma_hat_sq) - len(df_clean) + 2 * (len(combo_list) + 1)
                
                subsets_results.append({
                    "Size": len(combo_list),
                    "Predictors Combination": ", ".join(combo_list),
                    "RSquare": sub_model.rsquared,
                    "RSquare Adj": sub_model.rsquared_adj,
                    "Mallows Cp": sub_cp,
                    "AIC": sub_model.aic,
                    "BIC": sub_model.bic
                })
                
        subsets_df = pd.DataFrame(subsets_results)
        if not subsets_df.empty:
            # Sort by lowest BIC (or highest Adj R2) to bring the absolute best models to the very top
            #  To this:
            subsets_df = subsets_df.sort_values(by="BIC", ascending=True)
            
            st.dataframe(
                subsets_df.style.format({
                    "RSquare": "{:.4f}",
                    "RSquare Adj": "{:.4f}",
                    "Mallows Cp": "{:.2f}",
                    "AIC": "{:.2f}",
                    "BIC": "{:.2f}"
                }),
                use_container_width=True,
                hide_index=True
            )
else:
    st.info("💡 Please upload a CSV or Excel data file from the sidebar panel to begin configuration.")