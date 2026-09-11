

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm


def analizza_titolo(TICKER, DATA_IPO, seed):
    N_GIORNI = 180
    FINESTRA = 10
    GIORNI_ANNO = 252
    SEED = seed

    # Dati e rolling volatility
    start = pd.Timestamp(DATA_IPO)
    end = start + pd.Timedelta(days=int(N_GIORNI * 1.6))
    df = yf.download(TICKER, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.head(N_GIORNI)
    df["ret"] = np.log(df["Close"]).diff()
    df["vol_roll"] = df["ret"].rolling(FINESTRA).std() * np.sqrt(GIORNI_ANNO)
    df = df.dropna(subset=["vol_roll"]).copy()

    X = df["vol_roll"].values.reshape(-1, 1)

    # Stima dell' hmm a due strati
    best_model, best_logL = None, -np.inf
    for s in range(30):
        m = GaussianHMM(n_components=2, covariance_type="full",
                        n_iter=500, random_state=s)
        m.fit(X)
        ll = m.score(X)
        if ll > best_logL:
            best_logL, best_model = ll, m
    model = best_model

    # creazione di una convenzione: stato high-vol = quello con media più alta
    ordine = np.argsort(model.means_.ravel())[::-1]
    mappa = {vecchio: nuovo for nuovo, vecchio in enumerate(ordine)}

    # calcolo delle smoothed probabilities (P(stato | intero campione))
    post = model.predict_proba(X)
    prob_high = post[:, ordine[0]]
    df["prob_high"] = prob_high
    df["stato"] = (prob_high >= 0.5).astype(int)  # 1 = giorno in high-vol

    # parametri e statistiche di regime
    mu = model.means_.ravel()
    sig = np.sqrt(model.covars_.ravel())
    P_raw = model.transmat_

    # riordino la transition matrix nella convenzione high=0, low=1
    P = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            P[mappa[i], mappa[j]] = P_raw[i, j]

    # durate attese (distribuzione geometrica)
    dur_high = 1 / (1 - P[0, 0]) if P[0, 0] < 1 else np.inf
    dur_low = 1 / (1 - P[1, 1]) if P[1, 1] < 1 else np.inf

    # distribuzione stazionaria e quota empirica in high-vol
    pi_high = P[1, 0] / (P[0, 1] + P[1, 0])
    quota_emp = df["stato"].mean()

    # rapporto tra le medie dei due regimi (quanto sono separati)
    rapporto_medie = mu[ordine[0]] / mu[ordine[1]]

    # log-likelihood, AIC, BIC
    logL = model.score(X)
    k = 6  # 2 medie + 2 varianze + 2 prob. transizione indip.
    n = len(X)
    aic = -2 * logL + 2 * k
    bic = -2 * logL + np.log(n) * k

    # MODELLO A 1 SOLO REGIME (benchmark)
    # se c'è un unico regime, la volatilità è una singola gaussiana: gli stimatori MLE sono media e dev.std campionarie
    mu_1 = X.mean()
    sigma_1 = X.std(ddof=0)  # ddof=0 -> stimatore MLE

    # log-likelihood: somma dei log della densità normale su tutte le osservazioni
    logL_1 = norm.logpdf(X.ravel(), loc=mu_1, scale=sigma_1).sum()

    k_1 = 2  # una media + una varianza
    aic_1 = -2 * logL_1 + 2 * k_1
    bic_1 = -2 * logL_1 + np.log(n) * k_1

    # quale modello preferiscono i criteri? (valore PIÙ BASSO = migliore)
    vince_aic = "2-stati" if aic < aic_1 else "1-stato"
    vince_bic = "2-stati" if bic < bic_1 else "1-stato"

    #grafici

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(df.index, df["vol_roll"], color="black", lw=1)
    ax1.fill_between(df.index, 0, df["vol_roll"].max() * 1.05,
                     where=df["stato"] == 1, color="crimson", alpha=0.20,
                     label="HIGH-vol regime (smoothed)")
    ax1.axhline(mu[ordine[0]], ls=":", color="crimson", label="HIGH mean")
    ax1.axhline(mu[ordine[1]], ls=":", color="steelblue", label="LOW mean")
    ax1.set_title(f"{TICKER} — rolling realized volatility and estimated regimes")
    ax1.set_ylabel("Annualized volatility")
    ax1.legend(loc="upper left", fontsize=8)

    ax2.plot(df.index, df["prob_high"], color="crimson")
    ax2.axhline(0.5, ls="--", color="gray")
    ax2.set_title("Smoothed probability of the HIGH-vol state  P(high | full sample)")
    ax2.set_ylabel("P(high)")
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(f"{TICKER}_hmm.png", dpi=140)
    plt.close(fig)


    return {
        "ticker": TICKER,
        "p00": P[0, 0],
        "p11": P[1, 1],
        "durata_high": dur_high,
        "durata_low": dur_low,
        "mu_high": mu[ordine[0]],
        "mu_low": mu[ordine[1]],
        "rapporto_medie": rapporto_medie,
        "quota_high_staz": pi_high,
        "quota_high_emp": quota_emp,
        "persistenza_30gg": P[0, 0] ** 30,
        "aic": aic,
        "bic": bic,
        "logL": logL,
        "aic_1stato": aic_1,
        "bic_1stato": bic_1,
        "delta_aic": aic_1 - aic,  # positivo = il 2-stati è migliore
        "delta_bic": bic_1 - bic,  # positivo = il 2-stati è migliore
        "vince_aic": vince_aic,
        "vince_bic": vince_bic,
    }