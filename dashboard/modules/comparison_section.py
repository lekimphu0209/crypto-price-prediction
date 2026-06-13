"""
Model Comparison Section Module
Displays model performance comparison
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_comparison_section():
    """Render Model Comparison section"""
    st.title("🤖 Model Comparison")
    st.markdown("---")
    
    models_data = {
        'Model': ['Linear Regression', 'RNN', 'LSTM', 'BiLSTM', 'Transformer', 'Ensemble'],
        'RMSE': [1520, 980, 910, 870, 820, 780],
        'MAE': [1200, 750, 700, 680, 620, 590],
        'MAPE': ['4.2%', '2.8%', '2.4%', '2.3%', '2.0%', '1.9%'],
        'R²': [0.88, 0.91, 0.92, 0.93, 0.94, 0.95]
    }
    
    df_models = pd.DataFrame(models_data)
    st.dataframe(df_models, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_rmse = go.Figure()
        fig_rmse.add_trace(go.Bar(x=df_models['Model'], y=df_models['RMSE']))
        fig_rmse.update_layout(title="RMSE Comparison", height=300)
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    with col2:
        fig_mae = go.Figure()
        fig_mae.add_trace(go.Bar(x=df_models['Model'], y=df_models['MAE']))
        fig_mae.update_layout(title="MAE Comparison", height=300)
        st.plotly_chart(fig_mae, use_container_width=True)
    
    with col3:
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Bar(x=df_models['Model'], y=df_models['R²']))
        fig_r2.update_layout(title="R² Score Comparison", height=300)
        st.plotly_chart(fig_r2, use_container_width=True)
