import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

FEEDBACK_FILE = "feedback_humano.csv"
DADOS_FILE = "dados.csv"

st.set_page_config(page_title="Detecção de Fraudes", layout="wide")

def formatar_moeda(valor):
    try: return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except: return valor

def formatar_dataframe_valor(df):
    df = df.copy()
    if 'valor' in df.columns: df['valor'] = df['valor'].apply(formatar_moeda)
    if 'probabilidade_fraude' in df.columns: df['probabilidade_fraude'] = df['probabilidade_fraude'].apply(lambda x: f"{x:.2%}")
    return df

@st.cache_data
def carregar_dados():
    if not os.path.exists(DADOS_FILE): st.error(f"Arquivo {DADOS_FILE} não encontrado."); st.stop()
    return pd.read_csv(DADOS_FILE)

def enrich_features(df):
    df = df.copy()
    df['valor_log'] = np.log1p(df['valor'])
    def extrair_hora(horario):
        match = re.match(r"(\d{1,2}):", str(horario))
        if match:
            return int(match.group(1))
        return np.nan
    df['hora_num'] = df['horario'].apply(extrair_hora)
    df['periodo'] = pd.cut(df['hora_num'], bins=[-1,5,12,18,24], labels=['madrugada','manha','tarde','noite'])
    df['freq_cartao'] = df.groupby('tipo_cartao')['id'].transform('count')
    media_cartao = df.groupby('tipo_cartao')['valor'].transform('mean').replace(0,1e-3)
    df['valor_sobre_media_cartao'] = df['valor']/(media_cartao+1e-3)
    if 'data' in df.columns: df['data'] = pd.to_datetime(df['data']); df['final_de_semana'] = df['data'].dt.dayofweek.isin([5,6]).astype(int)
    else: df['final_de_semana'] = 0
    return df

@st.cache_data
def preprocessar_dados(df):
    df = enrich_features(df)
    features = [
        'valor','valor_log','hora_num','freq_cartao','valor_sobre_media_cartao',
        'pais','canal','tipo_cartao','periodo','final_de_semana'
    ]
    df_encoded = pd.get_dummies(df[features], drop_first=True)
    X = StandardScaler().fit_transform(df_encoded)
    X = np.nan_to_num(X)
    return X, df_encoded.columns

def treinar_modelo_supervisionado(df, random_state=42, usar_smote=True, test_size=0.3):
    X, _ = preprocessar_dados(df)
    if 'fraude' not in df.columns:
        st.error("Coluna 'fraude' não encontrada nos dados. O modelo supervisionado exige esse campo!")
        st.stop()
    y = df['fraude']
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=test_size, random_state=random_state, stratify=y
    )
    if usar_smote:
        sm = SMOTE(random_state=random_state)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        modelo = RandomForestClassifier(n_estimators=150, random_state=random_state)
        modelo.fit(X_res, y_res)
    else:
        modelo = RandomForestClassifier(n_estimators=150, random_state=random_state)
        modelo.fit(X_train, y_train)

    probas = modelo.predict_proba(X_test)[:,1]
    pred = modelo.predict(X_test)
    df_result = df_test.copy()
    df_result['fraude_predita'] = pred
    df_result['probabilidade_fraude'] = probas

    return df_result, modelo, X_train, y_train, df_train

def gerar_explicacoes(df):
    valor_med = df['valor'].mean()
    def explicacao(row):
        motivos = []
        if row['valor'] > valor_med*2: motivos.append("valor muito acima da média")
        try: hora = int(str(row['horario']).split(":")[0])
        except: motivos.append("horário inválido"); hora = 0
        if hora < 6 or hora > 22: motivos.append("horário incomum")
        if row['pais'] != 'BR': motivos.append(f"transação fora do país ({row['pais']})")
        if row['canal'] == 'web': motivos.append("canal online")
        if row.get('valor_sobre_media_cartao',1) > 3: motivos.append("valor acima do habitual deste cartão")
        if row.get('final_de_semana',0) == 1: motivos.append("transação no final de semana")
        if not motivos: motivos.append("padrão incomum detectado")
        return ", ".join(motivos)
    return df.apply(explicacao, axis=1)

def salvar_feedback(df_confirmadas, df_falsos_positivos):
    frames = []
    if not df_confirmadas.empty: df_confirmadas['fraude'] = 1; frames.append(df_confirmadas[['id','fraude']])
    if not df_falsos_positivos.empty: df_falsos_positivos['fraude'] = 0; frames.append(df_falsos_positivos[['id','fraude']])
    if not frames: return
    combinado = pd.concat(frames)
    if not os.path.exists(FEEDBACK_FILE): combinado.to_csv(FEEDBACK_FILE, index=False)
    else:
        antigo = pd.read_csv(FEEDBACK_FILE)
        atualizado = pd.concat([antigo, combinado]).drop_duplicates(subset=['id'], keep='last')
        atualizado.to_csv(FEEDBACK_FILE, index=False)

def reaprender_com_feedback(df, random_state=42, usar_smote=True):
    if not os.path.exists(FEEDBACK_FILE): return None, "Nenhum feedback registrado ainda."
    feedback = pd.read_csv(FEEDBACK_FILE)
    df_ajustado = df.copy()
    if 'id' not in df_ajustado.columns or 'id' not in feedback.columns: return None, "Coluna 'id' ausente."
    df_ajustado.set_index('id', inplace=True)
    feedback.set_index('id', inplace=True)
    df_ajustado.update(feedback[['fraude']])
    df_ajustado.reset_index(inplace=True)
    df_result, _, _, _, _ = treinar_modelo_supervisionado(df_ajustado, random_state=random_state, usar_smote=usar_smote)
    return df_result, "Modelo reentreinado com feedback."

st.title("Detecção de Fraudes - Aprendizado Supervisionado")

df = carregar_dados()

col1, col2, col3, col4 = st.columns(4)
valor_min, valor_max = float(df['valor'].min()), float(df['valor'].max())
with col1: filtro_valor = st.slider("Valor da Transação (R$)", min_value=valor_min, max_value=valor_max, value=(valor_min, valor_max), step=1.0)
with col2: filtro_pais = st.multiselect("País", options=sorted(df['pais'].unique()), default=list(df['pais'].unique()))
with col3: filtro_canal = st.multiselect("Canal", options=sorted(df['canal'].unique()), default=list(df['canal'].unique()))
with col4: filtro_tipo = st.multiselect("Tipo de Cartão", options=sorted(df['tipo_cartao'].unique()), default=list(df['tipo_cartao'].unique()))

col6, col7 = st.columns(2)
with col6: filtro_hora = st.slider("Horário (hora)", min_value=0, max_value=23, value=(0,23))
with col7: filtro_fraude = st.radio("Exibir", options=['Todas', 'Apenas Fraudes', 'Apenas Normais'], horizontal=True)

def aplicar_filtros(df):
    df_filtrado = df[
        (df['valor'] >= filtro_valor[0]) & (df['valor'] <= filtro_valor[1]) &
        (df['pais'].isin(filtro_pais)) & (df['canal'].isin(filtro_canal)) & 
        (df['tipo_cartao'].isin(filtro_tipo))
    ]
    
    def extrair_hora(horario):
        match = re.match(r"(\d{1,2}):", str(horario))
        if match:
            return int(match.group(1))
        return np.nan
    df_filtrado['hora_num'] = df_filtrado['horario'].apply(extrair_hora)
    df_filtrado = df_filtrado[(df_filtrado['hora_num'] >= filtro_hora[0]) & (df_filtrado['hora_num'] <= filtro_hora[1])]
    if filtro_fraude == 'Apenas Fraudes' and 'fraude' in df_filtrado.columns: df_filtrado = df_filtrado[df_filtrado['fraude']==1]
    if filtro_fraude == 'Apenas Normais' and 'fraude' in df_filtrado.columns: df_filtrado = df_filtrado[df_filtrado['fraude']==0]
    return df_filtrado

if "detecao_realizada" not in st.session_state:
    st.session_state.detecao_realizada = False

btn_detectar = st.button("Analisar")

if btn_detectar:
    df_resultado, modelo, X_train, y_train, df_train = treinar_modelo_supervisionado(df, usar_smote=True)
    st.session_state.df_resultado = df_resultado
    df_suspeitas = df_resultado[df_resultado['fraude_predita']==1].copy()
    df_suspeitas['explicacao'] = gerar_explicacoes(df_suspeitas)
    st.session_state.df_suspeitas = df_suspeitas
    st.session_state.df_train = df_train
    st.session_state.detecao_realizada = True

if st.session_state.detecao_realizada:
    df_resultado = aplicar_filtros(st.session_state.df_resultado)
    df_suspeitas = aplicar_filtros(st.session_state.df_suspeitas)

    falsos_positivos = df_resultado[(df_resultado.get('fraude_predita',0)==1) & (df_resultado.get('fraude',0)==0)] if 'fraude' in df_resultado.columns else pd.DataFrame()
    falsos_negativos = df_resultado[(df_resultado.get('fraude_predita',0)==0) & (df_resultado.get('fraude',0)==1)] if 'fraude' in df_resultado.columns else pd.DataFrame()

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.metric("Transações testadas", len(df_resultado))
    with col_m2:
        st.metric("Suspeitas detectadas", int(df_resultado['fraude_predita'].sum()))
    with col_m3:
        if 'fraude' in df_resultado.columns:
            st.metric("Fraudes reais", int(df_resultado['fraude'].sum()))
    with col_m4:
        if not falsos_positivos.empty:
            st.metric("❌ Falsos positivos", len(falsos_positivos))
    with col_m5:
        if not falsos_negativos.empty:
            st.metric("⚠️ Falsos negativos", len(falsos_negativos))

    with st.expander("🔍 Ver suspeitas detectadas (com explicação)", expanded=False):
        st.dataframe(formatar_dataframe_valor(df_suspeitas[['id','valor','pais','canal','tipo_cartao','horario','probabilidade_fraude','explicacao']]), use_container_width=False)
    def painel_feedback(df_fp, nome, keyprefix):
        with st.expander(f"🔍 Ver {nome} com detalhes", expanded=False):
            fmt = formatar_dataframe_valor(df_fp[['id','valor','pais','canal','tipo_cartao','horario','probabilidade_fraude']])
            id_selecionado = st.selectbox(f"Selecione um {nome[:-1]} para analisar", fmt['id'].tolist(), key=f"{keyprefix}_select")
            row = fmt[fmt['id']==id_selecionado]
            st.write(f"Detalhes do {nome[:-1]} selecionado:"); st.write(row.T)
            if st.button("✅ Confirmar como fraude", key=f"{keyprefix}_confirma_{id_selecionado}"):
                salvar_feedback(df_fp[df_fp['id']==id_selecionado], pd.DataFrame()); st.success("Feedback salvo.")
            if st.button("❌ Marcar como normal", key=f"{keyprefix}_normal_{id_selecionado}"):
                salvar_feedback(pd.DataFrame(), df_fp[df_fp['id']==id_selecionado]); st.success("Feedback salvo.")
    if not falsos_positivos.empty: painel_feedback(falsos_positivos, "falsos positivos", "fp")
    if not falsos_negativos.empty: painel_feedback(falsos_negativos, "falsos negativos", "fn")

    if os.path.exists(FEEDBACK_FILE):
        with st.expander("📜 Ver histórico de feedback humano", expanded=False):
            st.dataframe(pd.read_csv(FEEDBACK_FILE), use_container_width=False)
            if st.button("🧹 Limpar feedback salvo"):
                os.remove(FEEDBACK_FILE)
                st.success("Feedback removido!")
        if st.button("🔄 Reaprender com feedback"):
            df_reap, msg = reaprender_com_feedback(st.session_state.df_train, usar_smote=True)
            if df_reap is not None:
                st.success(msg)
                st.dataframe(formatar_dataframe_valor(df_reap.head(15)), use_container_width=False)
            else:
                st.warning(msg)
    else:
        if st.button("🔄 Reaprender com feedback"):
            df_reap, msg = reaprender_com_feedback(st.session_state.df_train, usar_smote=True)
            if df_reap is not None:
                st.success(msg)
                st.dataframe(formatar_dataframe_valor(df_reap.head(15)), use_container_width=False)
            else:
                st.warning(msg)

    st.markdown("### 📊 Visualizações")
    def plot_hist(df): 
        fig, ax = plt.subplots(figsize=(5,2.8))
        sns.histplot(data=df, x='valor', hue='fraude_predita', bins=30, kde=True, palette='pastel', alpha=0.7, ax=ax)
        ax.set_xlabel("Valor (R$)"); ax.set_ylabel("Qtd"); st.pyplot(fig, use_container_width=False)
    def plot_box(df):
        fig2, ax2 = plt.subplots(figsize=(5,2.8))
        sns.boxplot(data=df, x="tipo_cartao", y="valor", hue="fraude_predita", palette="Set2", ax=ax2)
        ax2.set_ylabel("Valor (R$)"); ax2.set_xlabel("Tipo de Cartão"); st.pyplot(fig2, use_container_width=False)
    colg1, colg2 = st.columns(2)
    with colg1: st.markdown("**Distribuição dos Valores** ℹ️"); plot_hist(df_resultado)
    with colg2: st.markdown("**Boxplot por Tipo de Cartão** ℹ️"); plot_box(df_resultado)
    if 'fraude' in df_resultado.columns:
        y_true, y_score = df_resultado['fraude'], df_resultado['probabilidade_fraude']
        def plot_roc(y_true, y_score):
            fig, ax = plt.subplots(figsize=(4,2.6))
            fpr, tpr, _ = roc_curve(y_true, y_score); roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='blue'); ax.plot([0,1],[0,1],'--',color='gray')
            ax.set_xlabel('Taxa de Falsos Positivos'); ax.set_ylabel('Taxa de Verdadeiros Positivos'); ax.legend(loc='lower right', fontsize=8)
            st.pyplot(fig, use_container_width=False)
        def plot_pr(y_true, y_score):
            fig, ax = plt.subplots(figsize=(4,2.6))
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ax.plot(recall, precision, color='purple'); ax.set_xlabel('Recall'); ax.set_ylabel('Precisão'); st.pyplot(fig, use_container_width=False)
        colg3, colg4 = st.columns(2)
        with colg3: st.markdown("**Curva ROC** ℹ️"); plot_roc(y_true, y_score)
        with colg4: st.markdown("**Curva Precision-Recall** ℹ️"); plot_pr(y_true, y_score)
    else:
        st.info("Adicione a coluna 'fraude' para cálculo de métricas de avaliação.")
else:
    st.info("Clique em **Analisar** para iniciar a análise.")