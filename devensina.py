import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
# petro e vale (exportadoras), itaú (juros e crédito), renner(pib doméstico/consumo) etc...
tickers= ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'LREN3.SA', 'WEGE3.SA', 'AXIA3.SA']
data_inicial = '2015-01-01'
data_final = '2025-12-30'
lambda_ewma = 0.94

print("1. Baixando dados do Yahoo Finance...")

# Baixamos tudo sem filtrar coluna ainda
# o auto_adjust=True faz com que o 'Close' já venha descontado de dividendos
dados_brutos = yf.download(tickers, start=data_inicial, end=data_final, auto_adjust=True)

# Passo B: Seleção Inteligente da Coluna
# Se 'Adj Close' não existir, usamos 'Close' (que já está ajustado pelo parâmetro)
if 'Adj Close' in dados_brutos.columns:
    dados = dados_brutos['Adj Close']
else:
    print("Aviso: Coluna 'Adj Close' não encontrada. Usando 'Close' (já ajustada).")
    dados = dados_brutos['Close']

# Passo C: Limpeza de Erros
# Remove colunas vazias (caso algum ticker tenha falhado)
dados = dados.dropna(axis=1, how='all').dropna()

# Trava de Segurança: Se sobrou zero dados, para tudo.
if dados.empty:
    print("ERRO CRÍTICO: A tabela de preços está vazia.")
    print("Verifique sua internet ou os tickers.")
    exit()

print(f"Sucesso! Ativos carregados: {list(dados.columns)}")


print("2. Calculando Log-Retornos...")
log_retornos = np.log(dados / dados.shift(1)).dropna()

print("3. Calculando Matriz de Covariância EWMA...")
vol_ewma_historica = log_retornos.ewm(alpha=(1 - lambda_ewma)).cov()
ultima_matriz_cov = vol_ewma_historica.iloc[-len(tickers):]


print("4. Calculando Vetor de Retornos Esperados...")
retornos_esperados = log_retornos.ewm(alpha=(1 - lambda_ewma)).mean().iloc[-1]
retornos_esperados *= 21
ultima_matriz_cov *= 21

print("5. Exportando arquivos .pkl...")
log_retornos.to_pickle("dados_log_retornos.pkl")
ultima_matriz_cov.to_pickle("matriz_cov_ewma.pkl")
retornos_esperados.to_pickle("vetor_retornos_esp.pkl")

print("\n--- RELATÓRIO FINAL ---")
print(f"Dimensão da Matriz de Risco: {ultima_matriz_cov.shape}")
print("Verifique se os 3 arquivos apareceram na sua pasta.")



# Carregar o arquivo
# AJUSTE: Mudei o nome da variável de 'dados' para 'matriz_cov' para evitar o erro de definição
matriz_cov = pd.read_pickle("matriz_cov_ewma.pkl")
# Mostrar na tela
print(matriz_cov)

from scipy.optimize import minimize

# 1. Preparação e Carregamento
# Garantindo que as variáveis de dimensão estejam alinhadas com os dados carregados
tickers = retornos_esperados.index.tolist()
n_ativos = len(tickers)
np.random.seed(42)

# 2. Simulação de Monte Carlo com t-Student
def calcular_metricas_starr(pesos, ret_esp, cov_mat, n_sim=10000, nu=5):
    """
    Calcula o STARR Ratio usando a distribuição t-Student.
    O parâmetro 'nu' define os graus de liberdade (quanto menor, maior o risco de cauda).
    """
    np.random.seed(42)
  #Garantir que o código seja replicável
    # Decomposição de Cholesky para manter as correlações entre ativos
    L = np.linalg.cholesky(cov_mat + np.eye(n_ativos) * 1e-8)

    # Geração de ruído t-Student
    # Ajustamos a escala para que o ruído tenha variância unitária antes da correlação
    Z = np.random.standard_t(df=nu, size=(n_sim, n_ativos))
    Z = Z * np.sqrt((nu - 2) / nu)

    # Projeção dos retornos simulados: R = Média + Correlacao * Ruído
    ret_sim = ret_esp.values + np.dot(Z, L.T)
    ret_port = np.dot(ret_sim, pesos)

    # Cálculo do CVaR 95% (Média das perdas nos 5% piores cenários)
    var_95 = np.percentile(ret_port, 5)
    cvar_95 = ret_port[ret_port <= var_95].mean()

    # Retorno esperado do portfólio
    ret_medio = np.dot(ret_esp, pesos)

    # Equação do STARR Ratio: Retorno / |Risco de Cauda|
    starr_ratio = ret_medio / abs(cvar_95)

    return starr_ratio, ret_medio, cvar_95

# 3. Função Objetivo para o Otimizador
def objetivo(pesos, ret_esp, cov_mat):
    # O minimize busca o valor mínimo, por isso retornamos o negativo do STARR
    s, _, _ = calcular_metricas_starr(pesos, ret_esp, cov_mat)
    return -s

# 4. Processo de Otimização
# Restrições: Peso total = 100% e pesos individuais entre 0 e 1
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(n_ativos))
chute_inicial = [1/n_ativos] * n_ativos

print("Iniciando otimização do STARR Ratio via Monte Carlo (t-Student)...")
res = minimize(objetivo, chute_inicial, args=(retornos_esperados, matriz_cov),
               method='SLSQP', bounds=bounds, constraints=cons)

# 5. Consolidação e Resultados
pesos_otimos = res.x
starr, ret_port, cvar_port = calcular_metricas_starr(pesos_otimos, retornos_esperados, matriz_cov)

print("\n" + "="*40)
print("     Resultados da Otimização     ")
print("="*40)
for t, p in zip(tickers, pesos_otimos):
    print(f"{t:10}: {p:8.2%}")

print("-" * 40)
print(f"Retorno Esperado Simulado: {ret_port:10.4%}")
print(f"CVaR (95%) Simulado:      {cvar_port:10.4%}")
print(f"STARR Ratio Final:         {starr:10.4f}")
print("="*40)

# 6. Gráfico da Distribuição do Portfólio Ótimo
# Re-simulando com os pesos finais para gerar o histograma
L_f = np.linalg.cholesky(matriz_cov + np.eye(n_ativos) * 1e-8)
Z_f = np.random.standard_t(df=5, size=(10000, n_ativos)) * np.sqrt(3/5)
dist_final = np.dot(retornos_esperados.values + np.dot(Z_f, L_f.T), pesos_otimos)



plt.figure(figsize=(12, 6))
plt.hist(dist_final, bins=100, color='navy', alpha=0.6, label='Cenários (t-Student)')
plt.axvline(cvar_port, color='red', linestyle='--', label=f'CVaR 95% ({cvar_port:.4%})')
plt.axvline(ret_port, color='green', label=f'Retorno Médio ({ret_port:.4%})')
plt.title("Distribuição de Retornos do Portfólio", fontsize=14)
plt.xlabel("Log-Retorno", fontsize=12)
plt.ylabel("Frequência", fontsize=12)
plt.legend()
plt.grid(alpha=0.2)
plt.show()



#  --------- BACKTEST ---------

print("\n" + "="*80)
print(" 7. EXECUCAO WALK-FORWARD (2018 em diante)")
print("="*80)

# --- 1. DOWNLOAD BENCHMARK ---
try:
    ibov = yf.download('^BVSP', start=dados.index[0], end=dados.index[-1], progress=False, auto_adjust=True)
    if isinstance(ibov.columns, pd.MultiIndex): ibov = ibov.xs('Close', axis=1, level=0)
    if 'Adj Close' in ibov.columns: ibov = ibov['Adj Close']
    elif 'Close' in ibov.columns: ibov = ibov['Close']
    ibov = ibov.squeeze().dropna().reindex(dados.index).ffill()
except:
    ibov = None

# --- 2. CONFIGURACOES DE DATA ---
data_inicio_backtest = pd.Timestamp('2018-01-01')

if data_inicio_backtest < dados.index[0]:
    print("ERRO: A data de inicio do backtest é anterior aos dados baixados!")
    janela_inicial = 252 * 2
else:
    janela_inicial = dados.index.searchsorted(data_inicio_backtest)

print(f" > Dados disponíveis desde: {dados.index[0].date()}")
print(f" > O Backtest vai começar em: {dados.index[janela_inicial].date()} (Linha {janela_inicial})")
print(f" > Periodo de Aprendizado (Warm-up): 2015 até 2017 usado para calibração inicial.")

datas = dados.index
n_dias = len(datas)
n_ativos = len(dados.columns)

patrimonio = np.zeros(n_dias)
caixa = 100
qtd_acoes = np.zeros(n_ativos)
mudanca_mes = pd.Series(datas.month) != pd.Series(datas.month).shift(1)
pesos_atuais = np.ones(n_ativos) / n_ativos

print(f"Iniciando simulacao dia a dia ({n_dias} steps)...")

# --- 3. LOOP DE SIMULACAO ---
for t in range(n_dias):
    if t < janela_inicial:
        ret_hoje = dados.iloc[t].values / dados.iloc[t-1].values - 1 if t > 0 else 0
        patrimonio[t] = 100
        continue

    if mudanca_mes.iloc[t]:
        dados_passado = dados.iloc[:t]
        log_ret_passado = np.log(dados_passado / dados_passado.shift(1)).dropna()

        cov_passado = log_ret_passado.ewm(alpha=0.06).cov().iloc[-n_ativos:]
        ret_esp_passado = log_ret_passado.mean() * 21

        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_ativos))

        def obj(w):
            s, _, _ = calcular_metricas_starr(w, ret_esp_passado, cov_passado)
            return -s

        try:
            res = minimize(obj, pesos_atuais, method='SLSQP', bounds=bounds, constraints=cons)
            if res.success:
                pesos_atuais = res.x
        except:
            pass

        precos_hoje = dados.iloc[t].values
        # Se é o primeiro dia do backtest (t == janela_inicial), usamos o caixa inicial
        patrimonio_total = (np.sum(qtd_acoes * precos_hoje) + caixa) if np.sum(qtd_acoes) > 0 else 100
        qtd_acoes = (patrimonio_total * pesos_atuais) / precos_hoje
        caixa = 0

    precos_hoje = dados.iloc[t].values
    patrimonio[t] = np.sum(qtd_acoes * precos_hoje) + caixa

# --- 4. RESULTADOS E METRICAS ---
# Split no array de 2018 para frente
equity = pd.Series(patrimonio, index=datas).iloc[janela_inicial:]
ret_acum = (equity / equity.iloc[0] - 1) * 100
retorno_total_pct = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

# Drawdown
pico = equity.cummax()
drawdown = (equity / pico - 1) * 100

print("-" * 30)
print(f"Periodo:          {equity.index[0].date()} ate {equity.index[-1].date()}")
print(f"Capital Inicial:  R$ {equity.iloc[0]:.2f}")
print(f"Capital Final:    R$ {equity.iloc[-1]:.2f}")
print(f"Retorno Acumulado:{retorno_total_pct:.2f}%")
print("-" * 30)

if ibov is not None:
    ibov_cut = ibov.loc[equity.index]

    # Normaliza Ibov para começar em 0% (apenas para o gráfico)
    ibov_norm = (ibov_cut / ibov_cut.iloc[0] - 1) * 100

    # CORREÇÃO AQUI: Calcula o retorno total usando os preços originais (ibov_cut)
    # Em vez de usar a série normalizada que começa em 0
    retorno_ibov_pct = (ibov_cut.iloc[-1] / ibov_cut.iloc[0] - 1) * 100

    alpha = retorno_total_pct - retorno_ibov_pct

    pico_ibov = ibov_cut.cummax()
    drawdown_ibov = (ibov_cut / pico_ibov - 1) * 100

    print(f"Ibovespa:         {retorno_ibov_pct:.2f}%")
    print(f"Alpha:            {alpha:+.2f}%")
    print("-" * 30)
# --- 5. VISUALIZACAO ---
plt.style.use('seaborn-v0_8-darkgrid')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Grafico 1: Retorno
ax1.plot(ret_acum.index, ret_acum, label='Starr Ratio Strat', color='#2E86AB', linewidth=2)
if ibov is not None:
    ax1.plot(ibov_norm.index, ibov_norm, label='Ibov', color='#A23B72', linestyle='--', alpha=0.7)

ax1.set_title('Retorno Acumulado (Desde 2018)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Retorno Acumulado (%)', fontsize=12)
ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color='black', linewidth=0.8)

# Grafico 2: Drawdown
ax2.fill_between(drawdown.index, drawdown, 0, label='Drawdown Starr Strat', color='#2E86AB', alpha=0.6)
if ibov is not None:
    ax2.plot(drawdown_ibov.index, drawdown_ibov, label='Drawdown Ibov', color='#A23B72', linewidth=1, alpha=0.8)

ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
ax2.set_ylabel('Queda (%)', fontsize=12)
ax2.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ANALISE SOLO DO DRAWDOWN PARA INSIGHTS E CONCLUSÃO

# 1. Garantia de Alinhamento e Cálculo Limpo (Decimal)
# Recalculamos aqui para garantir que não esteja multiplicado por 100
ibov_aligned = ibov.loc[equity.index]

# Drawdown Portfólio
pico_port = equity.cummax()
dd_port_series = (equity / pico_port) - 1  # Formato decimal (ex: -0.30)

# Drawdown Ibovespa
pico_ibov = ibov_aligned.cummax()
dd_ibov_series = (ibov_aligned / pico_ibov) - 1 # Formato decimal (ex: -0.45)

def get_dd_metrics(dd_series, price_series, peak_series):
    """Extrai valor, data topo, data fundo e duração."""
    min_val = dd_series.min()
    data_fundo = dd_series.idxmin()

    # Achar a data do topo exato antes da queda
    val_pico_ref = peak_series.loc[data_fundo]
    # Filtra preços até o fundo e pega a última vez que tocou no pico
    subset = price_series.loc[:data_fundo]
    data_topo = subset[subset == val_pico_ref].index[-1]

    dias = (data_fundo - data_topo).days
    return min_val, data_topo, data_fundo, dias

# Extraindo métricas
m_port = get_dd_metrics(dd_port_series, equity, pico_port)
m_ibov = get_dd_metrics(dd_ibov_series, ibov_aligned, pico_ibov)

# Montando a Tabela Visual
print("\n" + "="*80)
print(f"{'MÉTRICA DE RISCO (DRAWDOWN)':<30} | {'SUA ESTRATÉGIA':^22} | {'IBOVESPA':^22}")
print("="*80)

# Linha 1: Queda Máxima
print(f"{'Queda Máxima (MDD)':<30} | {m_port[0]:^22.2%} | {m_ibov[0]:^22.2%}")

# Linha 2: Data do Topo
print(f"{'Data do Topo (Início)':<30} | {str(m_port[1].date()):^22} | {str(m_ibov[1].date()):^22}")

# Linha 3: Data do Fundo
print(f"{'Data do Fundo (Pior Momento)':<30} | {str(m_port[2].date()):^22} | {str(m_ibov[2].date()):^22}")

# Linha 4: Duração
print(f"{'Duração da Queda (Dias)':<30} | {str(m_port[3]) + ' dias':^22} | {str(m_ibov[3]) + ' dias':^22}")

print("-" * 80)

# Linha 5: Delta
delta_risco = (m_port[0] - m_ibov[0]) * 100
status = "MELHOR" if delta_risco > 0 else "PIOR"
print(f"CONCLUSÃO: Sua carteira segurou a queda {abs(delta_risco):.2f} p.p. {status} que o Ibovespa.")
print("="*80 + "\n")
