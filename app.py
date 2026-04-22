import warnings
import numpy as np
import pandas as pd
import streamlit as st
import io
from itertools import permutations
# ─────────────────────────────────────────
# KURAL TABANLI YORUM MOTORU
# ─────────────────────────────────────────

def _karar_str(karar_val: str) -> str:
    """Karar sütunundan temiz metin çıkar."""
    return karar_val.replace("🟢","").replace("🔴","").replace("🟡","").strip()

def interpret_results(test_name: str, df_res: pd.DataFrame, alpha: float) -> str:
    """
    Test sonuç tablosunu satır satır okuyarak kural tabanlı Türkçe yorum üretir.
    """
    lines = []
    sig_count = 0
    total = 0

    # ── Durağanlık testleri ──────────────────────────────────────
    if test_name in ("ADF", "PP", "DF-GLS"):
        lines.append(f"**{test_name} Testi — İstatistiksel Yorum**\n")
        grouped = df_res.groupby("Değişken") if "Değişken" in df_res.columns else [(None, df_res)]
        for var, grp in (df_res.groupby("Değişken") if "Değişken" in df_res.columns else [("", df_res)]):
            var_lines = []
            for _, row in grp.iterrows():
                total += 1
                stat = row.get("İstatistik", "")
                p    = row.get("p-value", 1)
                lag  = row.get("Optimal Lag", row.get("Lag", "—"))
                c5   = row.get("Krit.5%", "")
                model= row.get("Model", "")
                karar= _karar_str(str(row.get("Karar", "")))
                durag = p <= alpha
                if durag: sig_count += 1
                yorum_c = (
                    f"  - **{model}:** t-istatistiği = {stat:.4f}, p = {p:.4f}"
                    f"{f', optimal gecikme = {lag}' if lag not in ('—', None, '') else ''}. "
                    f"Kritik değer (%5) = {c5}. "
                )
                if durag:
                    yorum_c += f"**H₀ reddedildi** — seri %{int((1-p)*100)} güven düzeyinde durağandır (I(0))."
                else:
                    yorum_c += f"**H₀ reddedilemedi** — seri durağan değildir, fark alınması gerekebilir."
                var_lines.append(yorum_c)
            lines.append(f"**{var}**")
            lines.extend(var_lines)
            lines.append("")

    elif test_name == "KPSS":
        lines.append("**KPSS Testi — İstatistiksel Yorum**\n")
        lines.append("> ⚠️ KPSS'de H₀ 'seri durağandır' şeklindedir — diğer testlerin tersidir.\n")
        for _, row in df_res.iterrows():
            total += 1
            var   = row.get("Değişken","")
            model = row.get("Model","")
            stat  = row.get("İstatistik","")
            p     = row.get("p-value", 0)
            c5    = row.get("Krit.5%","")
            durag = p >= alpha  # KPSS'de p > alpha → durağan
            if durag: sig_count += 1
            sonuc = "**H₀ reddedilemedi** — seri durağandır." if durag else "**H₀ reddedildi** — seri durağan değildir."
            lines.append(f"- **{var} / {model}:** istatistik = {stat:.4f}, p = {p:.4f}, Krit.5% = {c5}. {sonuc}")
        lines.append("")

    elif test_name == "Zivot-Andrews":
        lines.append("**Zivot-Andrews Testi — İstatistiksel Yorum**\n")
        lines.append("> Kırılma noktası veriyi tarayarak minimum t-istatistiğini veren gözlemde belirlenir.\n")
        for _, row in df_res.iterrows():
            total += 1
            var   = row.get("Değişken","")
            model = row.get("Model","")
            stat  = row.get("İstatistik","")
            p     = row.get("p-value",1)
            bp    = row.get("Kırılma Noktası","")
            c5    = row.get("Krit.5%","")
            durag = p <= alpha
            if durag: sig_count += 1
            sonuc = f"**H₀ reddedildi** — yapısal kırılma ({bp}) dahil seri durağandır." if durag else f"**H₀ reddedilemedi** — kırılma ({bp}) ile bile seri durağan değildir."
            lines.append(f"- **{var} / {model}:** t = {stat:.4f}, p = {p:.4f}, Krit.5% = {c5}. {sonuc}")
        lines.append("")

    # ── Nedensellik testleri ─────────────────────────────────────
    elif test_name in ("Granger", "Toda-Yamamoto", "Hsiao"):
        lines.append(f"**{test_name} Nedensellik Testi — İstatistiksel Yorum**\n")
        for _, row in df_res.iterrows():
            total += 1
            pair  = row.get("X → Y","")
            stat  = row.get("F-stat", row.get("Chi2",""))
            p     = row.get("p-value", 1)
            lag   = row.get("Lag", row.get("Optimal Lag", row.get("p_opt","")))
            anl   = p <= alpha
            if anl: sig_count += 1
            try: stat_f = f"{float(stat):.4f}"
            except: stat_f = str(stat)
            try: p_f = f"{float(p):.4f}"
            except: p_f = str(p)
            sonuc = (
                f"**Nedensellik tespit edildi** (%{int((1-float(p))*100)} güven düzeyi). "
                f"{pair.split('→')[0].strip()}, {pair.split('→')[1].strip()} serisini açıklamada istatistiksel olarak anlamlı katkı sağlamaktadır."
            ) if anl else (
                f"**Nedensellik tespit edilemedi.** "
                f"{pair.split('→')[0].strip()} değişkeni {pair.split('→')[1].strip()} için anlamlı bir öngörü gücü sunmamaktadır."
            )
            lines.append(f"- **{pair}** (lag={lag}, stat={stat_f}, p={p_f}): {sonuc}")
        lines.append("")

    elif test_name == "ECM Nedensellik":
        lines.append("**Engle-Granger ECM Nedensellik — İstatistiksel Yorum**\n")
        lines.append("> ECM katsayısı negatif ve anlamlıysa uzun dönem nedensellik (hata düzeltme mekanizması) mevcuttur.\n")
        for _, row in df_res.iterrows():
            total += 1
            pair  = row.get("X → Y","")
            t     = row.get("ECM t-stat","")
            p     = row.get("p-value (ECM)",1)
            p_co  = row.get("p-value (Coint)","")
            anl   = p <= alpha
            if anl: sig_count += 1
            try: t_f = f"{float(t):.4f}"
            except: t_f = str(t)
            try: p_f = f"{float(p):.4f}"
            except: p_f = str(p)
            sonuc = (
                f"**Uzun dönem nedensellik mevcut** (ECM t={t_f}, p={p_f}). "
                "Hata düzeltme mekanizması çalışmakta; kısa dönem sapmalar uzun dönem dengeye yakınsamaktadır."
            ) if anl else (
                f"**Uzun dönem nedensellik tespit edilemedi** (ECM t={t_f}, p={p_f})."
            )
            lines.append(f"- **{pair}** (Koent. p={p_co}): {sonuc}")
        lines.append("")

    elif test_name == "Breitung-Candelon":
        lines.append("**Breitung-Candelon Frekans Alanı Nedensellik — İstatistiksel Yorum**\n")
        lines.append("> π/6 uzun dönem, π/4 orta dönem, π/2 kısa dönem döngüsel hareketlere karşılık gelir.\n")
        for _, row in df_res.iterrows():
            total += 1
            pair  = row.get("X → Y","")
            freq  = row.get("Frekans","")
            chi2  = row.get("Chi2","")
            p     = row.get("p-value",1)
            anl   = p <= alpha
            if anl: sig_count += 1
            try: p_f = f"{float(p):.4f}"
            except: p_f = str(p)
            sonuc = f"**Anlamlı** (p={p_f}) — bu frekansta nedensellik mevcuttur." if anl else f"Anlamlı değil (p={p_f})."
            lines.append(f"- **{pair} / {freq}:** χ²={chi2}, {sonuc}")
        lines.append("")

    # ── Eşbütünleşme testleri ─────────────────────────────────────
    elif test_name in ("Engle-Granger Eşbütünleşme", "Phillips-Ouliaris"):
        lines.append(f"**{test_name} — İstatistiksel Yorum**\n")
        for _, row in df_res.iterrows():
            total += 1
            pair  = row.get("Y ~ X","")
            stat  = row.get("İstatistik","")
            p     = row.get("p-value",1)
            c5    = row.get("Krit.5%","")
            anl   = p <= alpha
            if anl: sig_count += 1
            try: stat_f = f"{float(stat):.4f}"
            except: stat_f = str(stat)
            try: p_f = f"{float(p):.4f}"
            except: p_f = str(p)
            sonuc = (
                f"**Eşbütünleşme tespit edildi** (stat={stat_f}, p={p_f}, Krit.5%={c5}). "
                "Seriler arasında uzun dönem denge ilişkisi mevcuttur — VECM kurulabilir."
            ) if anl else (
                f"**Eşbütünleşme tespit edilemedi** (stat={stat_f}, p={p_f}). "
                "Seviyede uzun dönem ilişki kanıtlanamadı."
            )
            lines.append(f"- **{pair}:** {sonuc}")
        lines.append("")

    elif test_name == "Johansen":
        lines.append("**Johansen Eşbütünleşme Testi — İstatistiksel Yorum**\n")
        lines.append("> Trace ve Max-Eig istatistikleri kritik değeri aşarsa H₀ reddedilir — koentegrasyon vektörü sayısı artar.\n")
        r_max = 0
        for _, row in df_res.iterrows():
            total += 1
            r     = row.get("H0 (r ≤)","")
            tr    = row.get("Trace Stat","")
            cv_tr = row.get("CV Trace 5%","")
            mx    = row.get("MaxEig Stat","")
            cv_mx = row.get("CV MaxEig 5%","")
            ret_tr = str(row.get("Trace Karar","")).replace("🟢","").replace("🔴","").strip()
            ret_mx = str(row.get("MaxEig Karar","")).replace("🟢","").replace("🔴","").strip()
            tr_anl = "Ret" in ret_tr
            mx_anl = "Ret" in ret_mx
            if tr_anl: sig_count += 1; r_max = int(r) + 1
            lines.append(
                f"- **r ≤ {r}:** Trace={tr:.4f} (CV={cv_tr}) → {'**Ret**' if tr_anl else 'Red Edilemedi'} | "
                f"Max-Eig={mx:.4f} (CV={cv_mx}) → {'**Ret**' if mx_anl else 'Red Edilemedi'}"
            )
        lines.append("")
        if r_max > 0:
            lines.append(f"**Sonuç:** Trace testine göre en az **{r_max} koentegrasyon vektörü** mevcuttur. VECM({r_max}) modeli kurulabilir.")
        else:
            lines.append("**Sonuç:** Koentegrasyon ilişkisi tespit edilemedi. VAR modeli düzeyde değil farklarla kurulmalıdır.")
        lines.append("")

    elif test_name == "Gregory-Hansen":
        lines.append("**Gregory-Hansen Eşbütünleşme Testi — İstatistiksel Yorum**\n")
        lines.append("> Standart eşbütünleşme testlerinden farkı: ilişkinin yapısal kırılmayla birlikte var olup olmadığını sınar.\n")
        for _, row in df_res.iterrows():
            total += 1
            pair  = row.get("Y ~ X","")
            model = row.get("Model","")
            stat  = row.get("ADF Min Stat","")
            bp    = row.get("Kırılma Noktası","")
            c5    = row.get("Krit.5%","")
            karar = str(row.get("Karar",""))
            anl   = "Var" in karar
            if anl: sig_count += 1
            try: stat_f = f"{float(stat):.4f}"
            except: stat_f = str(stat)
            sonuc = (
                f"**Kırılmalı eşbütünleşme tespit edildi** (ADF min={stat_f}, Krit.5%={c5}). "
                f"Kırılma noktası: {bp}. Uzun dönem ilişki bu tarihte rejim değişimiyle devam etmektedir."
            ) if anl else (
                f"**Eşbütünleşme tespit edilemedi** (ADF min={stat_f}, Krit.5%={c5}). "
                f"Kırılma noktası {bp} dahil uzun dönem ilişki kanıtlanamadı."
            )
            lines.append(f"- **{pair} / {model}:** {sonuc}")
        lines.append("")

    elif test_name == "ARDL Bounds":
        lines.append("**ARDL Bounds Testi — İstatistiksel Yorum**\n")
        lines.append("> F-istatistiği I(0) alt sınırının altındaysa ilişki yok, I(1) üst sınırının üstündeyse uzun dönem ilişki vardır, aradaysa sonuçsuz bölgedir.\n")
        for _, row in df_res.iterrows():
            total += 1
            y     = row.get("Y (Bağımlı)","")
            x     = row.get("X (Bağımsız)","")
            f     = row.get("F-stat","")
            p     = row.get("p-value","")
            i0    = row.get("CV I(0) 5%","")
            i1    = row.get("CV I(1) 5%","")
            karar = str(row.get("Karar",""))
            try: f_f = f"{float(f):.4f}"
            except: f_f = str(f)
            try: p_f = f"{float(p):.4f}"
            except: p_f = str(p)
            if "Var" in karar:
                sig_count += 1
                sonuc = f"**Uzun dönem ilişki mevcuttur** (F={f_f} > I(1) sınırı={i1}). ARDL uzun dönem katsayıları tahmin edilebilir."
            elif "Yok" in karar:
                sonuc = f"**Uzun dönem ilişki yoktur** (F={f_f} < I(0) sınırı={i0}). Seviyede eşbütünleşme kanıtlanamadı."
            else:
                sonuc = f"**Sonuçsuz bölge** (F={f_f}, I(0)={i0}, I(1)={i1}). Entegrasyon derecesi net belirlenmeden kesin karar verilemez."
            lines.append(f"- **{y} ~ {x}** (p={p_f}): {sonuc}")
        lines.append("")

    # ── Genel özet ──────────────────────────────────────────────
    if total > 0:
        oran = sig_count / total * 100
        lines.append("---")
        lines.append(
            f"**Özet:** {total} testin {sig_count} adedinde (%{oran:.0f}) H₀ reddedildi "
            f"(α = {alpha}). "
        )
        if oran >= 75:
            lines.append("Sonuçlar büyük ölçüde tutarlı ve istatistiksel olarak güçlüdür.")
        elif oran >= 40:
            lines.append("Sonuçlar karma bir tablo ortaya koymaktadır; model spesifikasyonu ve veri kalitesi gözden geçirilmelidir.")
        else:
            lines.append("Hipotezlerin büyük çoğunluğu reddedilememiştir; seri özellikleri veya örneklem büyüklüğü yetersiz olabilir.")

    return "\n".join(lines)


def show_interpretation(test_name: str, df_res: pd.DataFrame):
    """Test sonuçlarını kural tabanlı yorum motoruna gönderir."""
    yorum = interpret_results(test_name, df_res, alpha)
    st.markdown("---")
    st.markdown("#### 📝 İstatistiksel Yorum")
    st.markdown(yorum)
    st.markdown("---")



warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# SAYFA AYARLARI
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Ekonometrik Test Paketi",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Ekonometrik Test Paketi")
st.caption("Durağanlık · Nedensellik · Eşbütünleşme")

# ─────────────────────────────────────────
# AYARLAR (sidebar)
# ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Ayarlar")
    uploaded = st.file_uploader("Excel Dosyası Yükle", type=["xlsx", "xls"])
    max_lag  = st.slider("Maksimum Gecikme (MAX_LAG)", 1, 20, 12)
    alpha    = st.selectbox("Anlamlılık Düzeyi (α)", [0.01, 0.05, 0.10], index=1)
    st.markdown("---")
    st.markdown("**Kütüphane kurulumu:**")
    st.code("python.exe -m pip install streamlit\npython.exe -m pip install statsmodels\npython.exe -m pip install arch\npython.exe -m pip install scipy\npython.exe -m pip install xlsxwriter", language="bash")

# ─────────────────────────────────────────
# VERİ YÜKLEME
# ─────────────────────────────────────────
if uploaded is None:
    st.info("Sol panelden Excel dosyanızı yükleyin. 1. sütun dönem/yıl (indeks), diğerleri test edilecek seriler olmalıdır.")
    st.stop()

try:
    df_raw = pd.read_excel(uploaded, index_col=0)
    df_raw = df_raw.select_dtypes(include=[np.number])
    if df_raw.empty:
        st.error("Dosyada sayısal sütun bulunamadı.")
        st.stop()
except Exception as e:
    st.error(f"Dosya okunamadı: {e}")
    st.stop()

VARIABLES = {col: df_raw[col] for col in df_raw.columns}
VAR_NAMES = list(VARIABLES.keys())

with st.sidebar:
    st.markdown("---")
    st.markdown(f"**Yüklenen veri:** `{uploaded.name}`")
    st.markdown(f"**Gözlem sayısı:** {len(df_raw)}")
    st.markdown(f"**Değişkenler:** {', '.join(VAR_NAMES)}")

# ─────────────────────────────────────────
# YARDIMCI: KARAR
# ─────────────────────────────────────────
def karar_badge(p, reverse=False):
    if reverse:
        return "🔴 Durağan Değil" if p < alpha else "🟢 Durağan"
    return "🟢 Ret (Anlamlı)" if p <= alpha else "🔴 Red Edilemedi"

def karar_badge_coint(p):
    return "🟢 Ret (Eşbütünleşme Var)" if p <= alpha else "🔴 Red Edilemedi"

def entegrasyon(series, fn):
    for d, s in enumerate([series, series.diff().dropna(), series.diff().diff().dropna()]):
        try:
            if fn(s) <= alpha: return f"I({d})"
        except: pass
    return "I(?)"

def get_integration_order(series):
    from statsmodels.tsa.stattools import adfuller
    for d, s in enumerate([series, series.diff().dropna(), series.diff().diff().dropna()]):
        try:
            _, p, _, _, _, _ = adfuller(s.dropna(), autolag="AIC", regression="c")
            if p <= alpha: return d
        except: pass
    return 2

# ─────────────────────────────────────────
# EXCEL EXPORT
# ─────────────────────────────────────────
def df_to_excel(results_dict: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        wb = writer.book
        fmt_header = wb.add_format({"bold": True, "bg_color": "#1F4E79", "font_color": "white", "border": 1})
        fmt_green  = wb.add_format({"bg_color": "#E2EFDA", "border": 1})
        fmt_red    = wb.add_format({"bg_color": "#FCE4D6", "border": 1})
        fmt_normal = wb.add_format({"border": 1})
        fmt_title  = wb.add_format({"bold": True, "font_size": 12, "font_color": "#1F4E79"})

        for sheet_name, df in results_dict.items():
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False, startrow=1)
            ws = writer.sheets[safe_name]
            ws.write(0, 0, sheet_name, fmt_title)
            for col_num, col in enumerate(df.columns):
                ws.write(1, col_num, col, fmt_header)
                ws.set_column(col_num, col_num, max(len(str(col)) + 4, 14))
            for row_num in range(len(df)):
                karar_val = str(df.iloc[row_num].get("Karar", ""))
                fmt = fmt_green if "Ret" in karar_val or "Var" in karar_val or "Durağan" in karar_val and "Değil" not in karar_val else fmt_red if "Değil" in karar_val or "Edilemedi" in karar_val else fmt_normal
                for col_num in range(len(df.columns)):
                    ws.write(row_num + 2, col_num, df.iloc[row_num, col_num], fmt)
    return buf.getvalue()

# ─────────────────────────────────────────
# KATEGORİ SEKMELERİ
# ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📐 Durağanlık Testleri", "🔗 Nedensellik Testleri", "⚖️ Eşbütünleşme Testleri"])

# ═══════════════════════════════════════════════════════
# TAB 1: DURAĞANLIK
# ═══════════════════════════════════════════════════════
with tab1:
    from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews

    test_d = st.selectbox("Test Seç", ["ADF", "PP", "KPSS", "DF-GLS", "Zivot-Andrews"], key="dur_test")
    run_d  = st.button("▶ Testi Çalıştır", key="run_dur")

    if run_d:
        results_export = {}

        if test_d == "ADF":
            rows = []
            with st.spinner("ADF hesaplanıyor..."):
                for var, s in VARIABLES.items():
                    for model, reg in [("Sabit", "c"), ("Sabit+Trend", "ct")]:
                        stat, p, lag, n_obs, crit, _ = adfuller(s.dropna(), maxlag=max_lag, autolag="AIC", regression=reg)
                        rows.append({"Değişken": var, "Model": model,
                                     "İstatistik": round(stat,4), "p-value": round(p,4),
                                     "n": n_obs, "Optimal Lag": lag,
                                     "Krit.1%": round(crit["1%"],3), "Krit.5%": round(crit["5%"],3),
                                     "Krit.10%": round(crit["10%"],3), "Karar": karar_badge(p)})
            df_res = pd.DataFrame(rows)
            st.subheader("ADF — Augmented Dickey-Fuller")
            st.caption(f"H0: Birim kök vardır  |  Gecikme: AIC  |  α = {alpha}")
            st.dataframe(df_res, use_container_width=True, hide_index=True)
            show_interpretation("ADF", df_res)
            # Entegrasyon özeti
            io_rows = []
            for var, s in VARIABLES.items():
                deg = entegrasyon(s, lambda x: adfuller(x, autolag="AIC", regression="c")[1])
                io_rows.append({"Değişken": var, "Entegrasyon Derecesi": deg})
            st.subheader("Entegrasyon Derecesi Özeti")
            st.dataframe(pd.DataFrame(io_rows), use_container_width=True, hide_index=True)
            results_export["ADF"] = df_res
            results_export["ADF_Entegrasyon"] = pd.DataFrame(io_rows)

        elif test_d == "PP":
            try:
                from arch.unitroot import PhillipsPerron
                rows = []
                with st.spinner("PP hesaplanıyor..."):
                    for var, s in VARIABLES.items():
                        for model, trend in [("Sabit", "c"), ("Sabit+Trend", "ct")]:
                            pp = PhillipsPerron(s.dropna(), trend=trend, lags=max_lag)
                            rows.append({"Değişken": var, "Model": model,
                                         "İstatistik": round(pp.stat,4), "p-value": round(pp.pvalue,4),
                                         "n": pp.nobs,
                                         "Krit.1%": round(pp.critical_values["1%"],3),
                                         "Krit.5%": round(pp.critical_values["5%"],3),
                                         "Krit.10%": round(pp.critical_values["10%"],3),
                                         "Karar": karar_badge(pp.pvalue)})
                df_res = pd.DataFrame(rows)
                st.subheader("PP — Phillips-Perron")
                st.caption(f"H0: Birim kök vardır  |  Kernel: Bartlett-Newey-West  |  α = {alpha}")
                st.dataframe(df_res, use_container_width=True, hide_index=True)
                show_interpretation("PP", df_res)
                results_export["PP"] = df_res
            except ImportError:
                st.error("PP için `arch` kütüphanesi gerekli: `python.exe -m pip install arch`")

        elif test_d == "KPSS":
            rows = []
            with st.spinner("KPSS hesaplanıyor..."):
                for var, s in VARIABLES.items():
                    for model, reg in [("Sabit", "c"), ("Sabit+Trend", "ct")]:
                        stat, p, lag, crit = kpss(s.dropna(), regression=reg, nlags="auto")
                        rows.append({"Değişken": var, "Model": model,
                                     "İstatistik": round(stat,4), "p-value": round(p,4),
                                     "n": len(s.dropna()), "Lag": lag,
                                     "Krit.1%": round(crit["1%"],3), "Krit.5%": round(crit["5%"],3),
                                     "Krit.10%": round(crit["10%"],3),
                                     "Karar": karar_badge(p, reverse=True)})
            df_res = pd.DataFrame(rows)
            st.subheader("KPSS — Kwiatkowski-Phillips-Schmidt-Shin")
            st.caption(f"H0: Seri DURAĞANDIR (ADF'nin tersi)  |  α = {alpha}")
            st.info("⚠️ KPSS'de H0 tersine çevrilmiştir: p < α → Durağan Değil")
            st.dataframe(df_res, use_container_width=True, hide_index=True)
            show_interpretation("KPSS", df_res)
            results_export["KPSS"] = df_res

        elif test_d == "DF-GLS":
            try:
                from arch.unitroot import DFGLS
                rows = []
                with st.spinner("DF-GLS hesaplanıyor..."):
                    for var, s in VARIABLES.items():
                        for model, trend in [("Sabit", "c"), ("Sabit+Trend", "ct")]:
                            d = DFGLS(s.dropna(), trend=trend, max_lags=max_lag)
                            rows.append({"Değişken": var, "Model": model,
                                         "İstatistik": round(d.stat,4), "p-value": round(d.pvalue,4),
                                         "n": d.nobs, "Optimal Lag": d.lags,
                                         "Krit.1%": round(d.critical_values["1%"],3),
                                         "Krit.5%": round(d.critical_values["5%"],3),
                                         "Krit.10%": round(d.critical_values["10%"],3),
                                         "Karar": karar_badge(d.pvalue)})
                df_res = pd.DataFrame(rows)
                st.subheader("DF-GLS — Elliott, Rothenberg & Stock (1996)")
                st.caption(f"H0: Birim kök vardır  |  GLS detrending  |  α = {alpha}")
                st.dataframe(df_res, use_container_width=True, hide_index=True)
                show_interpretation("DF-GLS", df_res)
                results_export["DF-GLS"] = df_res
            except ImportError:
                st.error("DF-GLS için `arch` kütüphanesi gerekli.")

        elif test_d == "Zivot-Andrews":
            rows = []
            with st.spinner("Zivot-Andrews hesaplanıyor (yavaş olabilir)..."):
                for var, s in VARIABLES.items():
                    for model, reg in [("Sabit Kırılma","c"),("Trend Kırılma","t"),("Her İkisi","ct")]:
                        try:
                            stat, p, _, bp, crit = zivot_andrews(s.dropna(), maxlag=max_lag, regression=reg)
                            try: bp_l = str(s.dropna().index[bp])[:10]
                            except: bp_l = str(bp)
                            rows.append({"Değişken": var, "Model": model,
                                         "İstatistik": round(stat,4), "p-value": round(p,4),
                                         "n": len(s.dropna()), "Kırılma Noktası": bp_l,
                                         "Krit.1%": round(crit["1%"],3),
                                         "Krit.5%": round(crit["5%"],3),
                                         "Krit.10%": round(crit["10%"],3),
                                         "Karar": karar_badge(p)})
                        except Exception as e:
                            rows.append({"Değişken": var, "Model": model, "Karar": f"Hata: {e}"})
            df_res = pd.DataFrame(rows)
            st.subheader("Zivot-Andrews (1992) — Yapısal Kırılmalı")
            st.caption(f"H0: Birim kök vardır  |  Kırılma tarihi endojen  |  α = {alpha}")
            st.dataframe(df_res, use_container_width=True, hide_index=True)
            show_interpretation("Zivot-Andrews", df_res)
            results_export["Zivot-Andrews"] = df_res

        if results_export:
            excel_bytes = df_to_excel(results_export)
            st.download_button("⬇️ Excel'e Aktar", data=excel_bytes,
                               file_name=f"duraganlık_{test_d}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ═══════════════════════════════════════════════════════
# TAB 2: NEDENSELLİK
# ═══════════════════════════════════════════════════════
with tab2:
    from statsmodels.tsa.stattools import adfuller, grangercausalitytests, coint
    from statsmodels.tsa.vector_ar.var_model import VAR
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant

    test_n = st.selectbox("Test Seç", ["Granger", "Toda-Yamamoto", "Hsiao", "Engle-Granger ECM", "Breitung-Candelon"], key="ned_test")

    col1, col2 = st.columns(2)
    with col1:
        x_sec = st.selectbox("X (Bağımsız)", ["Tüm çiftler"] + VAR_NAMES, key="x_ned")
    with col2:
        y_opts = [v for v in VAR_NAMES if v != x_sec] if x_sec != "Tüm çiftler" else VAR_NAMES
        y_sec = st.selectbox("Y (Bağımlı)", ["Tüm çiftler"] + VAR_NAMES if x_sec == "Tüm çiftler" else y_opts, key="y_ned")

    if x_sec == "Tüm çiftler":
        pairs = list(permutations(VAR_NAMES, 2))
    else:
        pairs = [(x_sec, y_sec)]

    run_n = st.button("▶ Testi Çalıştır", key="run_ned")

    if run_n:
        results_export = {}

        if test_n == "Granger":
            rows = []
            with st.spinner("Granger hesaplanıyor..."):
                for x_name, y_name in pairs:
                    x = VARIABLES[x_name]; y = VARIABLES[y_name]
                    data = pd.concat([y, x], axis=1).dropna()
                    best_lag, best_p, best_f = 1, 1.0, 0.0
                    for lag in range(1, min(max_lag+1, len(data)//4)):
                        try:
                            res = grangercausalitytests(data, maxlag=lag, verbose=False)
                            p = res[lag]["ssr_ftest"][1]; f = res[lag]["ssr_ftest"][0]
                            if p < best_p: best_lag, best_p, best_f = lag, p, f
                        except: pass
                    n_obs = len(data) - best_lag
                    rows.append({"X → Y": f"{x_name} → {y_name}", "Lag": best_lag,
                                 "F-stat": round(best_f,4), "p-value": round(best_p,4),
                                 "df2": n_obs-best_lag-1, "Karar": karar_badge(best_p)})
            df_res = pd.DataFrame(rows)
            st.subheader("Granger Nedensellik Testi")
            st.caption(f"H0: X, Y'nin Granger nedeni değildir  |  α = {alpha}")
            st.dataframe(df_res, use_container_width=True, hide_index=True)
            show_interpretation("Granger", df_res)
            results_export["Granger"] = df_res

        elif test_n == "Toda-Yamamoto":
            rows = []
            with st.spinner("Toda-Yamamoto hesaplanıyor..."):
                for x_name, y_name in pairs:
                    x = VARIABLES[x_name]; y = VARIABLES[y_name]
                    d_max = max(get_integration_order(x), get_integration_order(y))
                    data = pd.concat([y, x], axis=1).dropna()
                    try:
                        model = VAR(data)
                        p_opt = max(1, model.select_order(maxlags=min(max_lag,8)).aic)
                        fit = model.fit(p_opt + d_max)
                        restr = np.zeros((p_opt, fit.params.shape[0]))
                        for i in range(p_opt):
                            restr[i, 1 + i*2 + 1] = 1
                        wt = fit.wald_test(restr, equation=0)
                        chi2 = float(wt.statistic); wp = float(wt.pvalue)
                    except: chi2, wp, p_opt, d_max = np.nan, np.nan, 0, 0
                    rows.append({"X → Y": f"{x_name} → {y_name}", "p_opt": p_opt, "d_max": d_max,
                                 "Chi2": round(chi2,4) if not np.isnan(chi2) else "—",
                                 "p-value": round(wp,4) if not np.isnan(wp) else "—",
                                 "Karar": karar_badge(wp) if not np.isnan(wp) else "—"})
            df_res = pd.DataFrame(rows)
            st.subheader("Toda-Yamamoto Nedensellik Testi")
            st.caption(f"H0: X, Y'nin Granger nedeni değildir  |  Entegrasyondan bağımsız  |  α = {alpha}")
            st.dataframe(df_res, use_container_width=True, hide_index=True)
            show_interpretation("Toda-Yamamoto", df_res)
            results_export["Toda-Yamamoto"] = df_res

        elif test_n == "Hsiao":
            rows = []
            with st.spinner("Hsiao hesaplanıyor..."):
                for x_name, y_name in pairs:
                    x = VARIABLES[x_name]; y = VARIABLES[y_name]
                    data = pd.concat([y, x], axis=1).dropna()
                    best_m, best_aic = 1, np.inf
                    for m in range(1, min(max_lag+1, len(data)//4)):
                        try:
                            res = grangercausalitytests(data, maxlag=m, verbose=False)
                            aic = res[m]["params_ftest"][0]
                            if aic < best_aic: best_m, best_aic = m, aic
                        except: pass
                    try:
                        res = grangercausalitytests(data, maxlag=best_m, verbose=False)
                        f = res[best_m]["ssr_ftest"][0]; p = res[best_m]["ssr_ftest"][1]
                        df1 = res[best_m]["ssr_ftest"][2]; df2 = int(res[best_m]["ssr_ftest"][3])
                    except: f, p, df1, df2 = np.nan, np.nan, 0, 0
                    rows.append({"X → Y": f"{x_name} → {y_name}", "Optimal Lag": best_m,
                                 "F-stat": round(f,4) if not np.isnan(f) else "—",
                                 "p-value": round(p,4) if not np.isnan(p) else "—",
                                 "df1": int(df1), "df2": df2,
                                 "Karar": karar_badge(p) if not np.isnan(p) else "—"})
            df_res = pd.DataFrame(rows)
            st.subheader("Hsiao Nedensellik Testi")
            st.caption(f"H0: X, Y'nin Granger nedeni değildir  |  AIC ile optimal gecikme  |  α = {alpha}")
            st.dataframe(df_res, use_container_width=True, hide_index=True)
            show_interpretation("Hsiao", df_res)
            results_export["Hsiao"] = df_res

        elif test_n == "Engle-Granger ECM":
            rows = []
            with st.spinner("ECM Nedensellik hesaplanıyor..."):
                for x_name, y_name in pairs:
                    x = VARIABLES[x_name]; y = VARIABLES[y_name]
                    try:
                        _, p_coint, _ = coint(y.dropna(), x.dropna())
                        reg = OLS(y, add_constant(x)).fit()
                        ecm_resid = reg.resid
                        dy = y.diff().dropna(); dx = x.diff().dropna()
                        ecm_lag = ecm_resid.shift(1).dropna()
                        idx = dy.index.intersection(dx.index).intersection(ecm_lag.index)
                        endog = dy.loc[idx]
                        exog = add_constant(pd.concat([dx.loc[idx], ecm_lag.loc[idx]], axis=1))
                        fit = OLS(endog, exog).fit()
                        t = fit.tvalues.iloc[-1]; p = fit.pvalues.iloc[-1]
                        rows.append({"X → Y": f"{x_name} → {y_name}",
                                     "ECM t-stat": round(t,4), "p-value (ECM)": round(p,4),
                                     "p-value (Coint)": round(p_coint,4), "n": len(idx),
                                     "Karar": karar_badge(p)})
                    except Exception as e:
                        rows.append({"X → Y": f"{x_name} → {y_name}", "Karar": f"Hata: {e}"})
            df_res = pd.DataFrame(rows)
            st.subheader("Engle-Granger ECM Nedensellik Testi")
            st.caption(f"H0: Uzun dönem nedensellik yoktur  |  α = {alpha}")
            st.dataframe(df_res, use_container_width=True, hide_index=True)
            show_interpretation("ECM Nedensellik", df_res)
            results_export["ECM_Nedensellik"] = df_res

        elif test_n == "Breitung-Candelon":
            try:
                from scipy.stats import chi2 as chi2_dist
                freqs = [np.pi/6, np.pi/4, np.pi/2, np.pi]
                freq_labels = ["π/6 (uzun dönem)", "π/4 (orta dönem)", "π/2 (kısa dönem)", "π (çok kısa)"]
                rows = []
                with st.spinner("Breitung-Candelon hesaplanıyor..."):
                    for x_name, y_name in pairs:
                        x = VARIABLES[x_name]; y = VARIABLES[y_name]
                        try:
                            var_m = VAR(pd.concat([y,x],axis=1).dropna())
                            p_opt = max(1, var_m.select_order(maxlags=min(max_lag,8)).aic)
                        except: p_opt = 2
                        dy = y.diff().dropna(); dx = x.diff().dropna()
                        T2 = min(len(dy), len(dx))
                        dy = dy.iloc[-T2:]; dx = dx.iloc[-T2:]
                        for freq, flabel in zip(freqs, freq_labels):
                            try:
                                cos_t = np.cos(freq * np.arange(1, T2+1))
                                sin_t = np.sin(freq * np.arange(1, T2+1))
                                lags_y = np.column_stack([dy.values[p_opt-k:T2-k] for k in range(1,p_opt+1)])
                                lags_x = np.column_stack([dx.values[p_opt-k:T2-k] for k in range(1,p_opt+1)])
                                cos_r = cos_t[p_opt:]; sin_r = sin_t[p_opt:]
                                endog = dy.values[p_opt:]
                                exog_r = add_constant(np.column_stack([lags_y, lags_x]))
                                exog_u = add_constant(np.column_stack([lags_y, lags_x, cos_r, sin_r]))
                                rss_r = OLS(endog, exog_r).fit().ssr
                                rss_u = OLS(endog, exog_u).fit().ssr
                                n_obs = len(endog)
                                chi2 = n_obs * (rss_r - rss_u) / rss_u
                                p_val = 1 - chi2_dist.cdf(chi2, df=2)
                                rows.append({"X → Y": f"{x_name} → {y_name}", "Frekans": flabel,
                                             "Chi2": round(chi2,4), "p-value": round(p_val,4),
                                             "Lag": p_opt, "Karar": karar_badge(p_val)})
                            except Exception as e:
                                rows.append({"X → Y": f"{x_name} → {y_name}", "Frekans": flabel, "Karar": f"Hata: {e}"})
                df_res = pd.DataFrame(rows)
                st.subheader("Breitung-Candelon Frekans Alanı Nedensellik")
                st.caption(f"H0: ω frekansında X, Y'nin nedeni değildir  |  α = {alpha}")
                st.dataframe(df_res, use_container_width=True, hide_index=True)
                show_interpretation("Breitung-Candelon", df_res)
                results_export["Breitung-Candelon"] = df_res
            except ImportError:
                st.error("scipy kütüphanesi gerekli: `python.exe -m pip install scipy`")

        if results_export:
            excel_bytes = df_to_excel(results_export)
            st.download_button("⬇️ Excel'e Aktar", data=excel_bytes,
                               file_name=f"nedensellik_{test_n}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ═══════════════════════════════════════════════════════
# TAB 3: EŞBÜTÜNLEŞme
# ═══════════════════════════════════════════════════════
with tab3:
    from statsmodels.tsa.stattools import coint, adfuller
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant

    test_e = st.selectbox("Test Seç", ["Engle-Granger", "Johansen", "Phillips-Ouliaris", "Gregory-Hansen", "ARDL Bounds"], key="esb_test")

    if test_e == "Johansen":
        selected_vars = st.multiselect("Değişkenleri Seç (en az 2)", VAR_NAMES, default=VAR_NAMES)
    else:
        col1, col2 = st.columns(2)
        with col1:
            x_esb = st.selectbox("X (Bağımsız)", ["Tüm çiftler"] + VAR_NAMES, key="x_esb")
        with col2:
            y_esb_opts = [v for v in VAR_NAMES if v != x_esb] if x_esb != "Tüm çiftler" else VAR_NAMES
            y_esb = st.selectbox("Y (Bağımlı)", ["Tüm çiftler"] + VAR_NAMES if x_esb == "Tüm çiftler" else y_esb_opts, key="y_esb")
        pairs_e = list(permutations(VAR_NAMES, 2)) if x_esb == "Tüm çiftler" else [(x_esb, y_esb)]

    run_e = st.button("▶ Testi Çalıştır", key="run_esb")

    if run_e:
        results_export = {}

        if test_e == "Engle-Granger":
            rows = []
            with st.spinner("Engle-Granger hesaplanıyor..."):
                for x_name, y_name in pairs_e:
                    x = VARIABLES[x_name]; y = VARIABLES[y_name]
                    try:
                        stat, p, crit = coint(y.dropna(), x.dropna())
                        rows.append({"Y ~ X": f"{y_name} ~ {x_name}",
                                     "İstatistik": round(stat,4), "p-value": round(p,4),
                                     "Krit.1%": round(crit[0],3), "Krit.5%": round(crit[1],3),
                                     "Krit.10%": round(crit[2],3), "Karar": karar_badge_coint(p)})
                    except Exception as e:
                        rows.append({"Y ~ X": f"{y_name} ~ {x_name}", "Karar": f"Hata: {e}"})
            df_res = pd.DataFrame(rows)
            st.subheader("Engle-Granger Eşbütünleşme Testi")
            st.caption(f"H0: Eşbütünleşme yoktur (artıklar I(1))  |  α = {alpha}")
            st.dataframe(df_res, use_container_width=True, hide_index=True)
            show_interpretation("Engle-Granger Eşbütünleşme", df_res)
            results_export["Engle-Granger"] = df_res

        elif test_e == "Johansen":
            from statsmodels.tsa.vector_ar.vecm import coint_johansen
            if len(selected_vars) < 2:
                st.error("En az 2 değişken seçin.")
            else:
                with st.spinner("Johansen hesaplanıyor..."):
                    try:
                        data = df_raw[selected_vars].dropna()
                        res = coint_johansen(data, det_order=0, k_ar_diff=1)
                        rows = []
                        for r in range(res.lr1.shape[0]):
                            trace = res.lr1[r]; maxeig = res.lr2[r]
                            cv_tr = res.cvt[r,1]; cv_mx = res.cvm[r,1]
                            rows.append({"H0 (r ≤)": r,
                                         "Trace Stat": round(trace,4), "CV Trace 5%": round(cv_tr,3),
                                         "Trace Karar": "🟢 Ret" if trace > cv_tr else "🔴 Red Edilemedi",
                                         "MaxEig Stat": round(maxeig,4), "CV MaxEig 5%": round(cv_mx,3),
                                         "MaxEig Karar": "🟢 Ret" if maxeig > cv_mx else "🔴 Red Edilemedi"})
                        df_res = pd.DataFrame(rows)
                        st.subheader("Johansen Eşbütünleşme Testi")
                        st.caption(f"H0: En fazla r adet koentegrasyon vektörü var  |  α = {alpha}")
                        st.dataframe(df_res, use_container_width=True, hide_index=True)
                        show_interpretation("Johansen", df_res)
                        results_export["Johansen"] = df_res
                    except Exception as e:
                        st.error(f"Hata: {e}")

        elif test_e == "Phillips-Ouliaris":
            rows = []
            with st.spinner("Phillips-Ouliaris hesaplanıyor..."):
                for x_name, y_name in pairs_e:
                    x = VARIABLES[x_name]; y = VARIABLES[y_name]
                    try:
                        stat, p, crit = coint(y.dropna(), x.dropna(), trend="c")
                        rows.append({"Y ~ X": f"{y_name} ~ {x_name}",
                                     "İstatistik": round(stat,4), "p-value": round(p,4),
                                     "Krit.1%": round(crit[0],3), "Krit.5%": round(crit[1],3),
                                     "Krit.10%": round(crit[2],3), "Karar": karar_badge_coint(p)})
                    except Exception as e:
                        rows.append({"Y ~ X": f"{y_name} ~ {x_name}", "Karar": f"Hata: {e}"})
            df_res = pd.DataFrame(rows)
            st.subheader("Phillips-Ouliaris Eşbütünleşme Testi")
            st.caption(f"H0: Eşbütünleşme yoktur  |  Artık tabanlı  |  α = {alpha}")
            st.dataframe(df_res, use_container_width=True, hide_index=True)
            show_interpretation("Phillips-Ouliaris", df_res)
            results_export["Phillips-Ouliaris"] = df_res

        elif test_e == "Gregory-Hansen":
            GH_CRIT = {
                "C":   {"1%": -5.13, "5%": -4.61, "10%": -4.34},
                "C/T": {"1%": -5.45, "5%": -4.99, "10%": -4.72},
                "C/S": {"1%": -5.47, "5%": -4.95, "10%": -4.68},
            }
            rows = []
            with st.spinner("Gregory-Hansen hesaplanıyor (yavaş olabilir)..."):
                for x_name, y_name in pairs_e:
                    x = VARIABLES[x_name]; y = VARIABLES[y_name]
                    for model_label in ["C", "C/T", "C/S"]:
                        try:
                            idx = y.dropna().index.intersection(x.dropna().index)
                            y_s = y.loc[idx].values; x_s = x.loc[idx].values
                            T = len(y_s); trim = int(0.15*T)
                            min_stat, min_bp = np.inf, trim
                            for bp in range(trim, T-trim):
                                du = np.zeros(T); du[bp:] = 1
                                if model_label == "C":
                                    X = add_constant(np.column_stack([x_s, du]))
                                elif model_label == "C/T":
                                    X = add_constant(np.column_stack([x_s, du, np.arange(T)]))
                                else:
                                    X = add_constant(np.column_stack([x_s, du, x_s*du]))
                                resid = OLS(y_s, X).fit().resid
                                adf_s = adfuller(resid, maxlag=max_lag, autolag="AIC", regression="nc")[0]
                                if adf_s < min_stat: min_stat, min_bp = adf_s, bp
                            crit = GH_CRIT[model_label]
                            k = "🟢 Ret (Kırılmalı Eşbütünleşme Var)" if min_stat < crit["5%"] else "🔴 Red Edilemedi"
                            try: bp_label = str(idx[min_bp])[:10]
                            except: bp_label = str(min_bp)
                            rows.append({"Y ~ X": f"{y_name} ~ {x_name}", "Model": model_label,
                                         "ADF Min Stat": round(min_stat,4), "Kırılma Noktası": bp_label,
                                         "Krit.1%": crit["1%"], "Krit.5%": crit["5%"],
                                         "Krit.10%": crit["10%"], "Karar": k})
                        except Exception as e:
                            rows.append({"Y ~ X": f"{y_name} ~ {x_name}", "Model": model_label, "Karar": f"Hata: {e}"})
            df_res = pd.DataFrame(rows)
            st.subheader("Gregory-Hansen Eşbütünleşme Testi (1996)")
            st.caption(f"H0: Kırılmalı eşbütünleşme yoktur  |  C=sabit | C/T=trend | C/S=rejim  |  α = {alpha}")
            st.dataframe(df_res, use_container_width=True, hide_index=True)
            show_interpretation("Gregory-Hansen", df_res)
            results_export["Gregory-Hansen"] = df_res

        elif test_e == "ARDL Bounds":
            PSS = {"1%": {"I0":6.84,"I1":7.84}, "5%": {"I0":4.94,"I1":5.73}, "10%": {"I0":4.04,"I1":4.78}}
            from scipy.stats import f as f_dist
            rows = []
            with st.spinner("ARDL Bounds hesaplanıyor..."):
                for x_name, y_name in pairs_e:
                    x = VARIABLES[x_name]; y = VARIABLES[y_name]
                    try:
                        dy = y.diff().dropna(); dx = x.diff().dropna()
                        y_lag = y.shift(1).dropna(); x_lag = x.shift(1).dropna()
                        idx = dy.index.intersection(dx.index).intersection(y_lag.index).intersection(x_lag.index)
                        endog = dy.loc[idx]
                        p_opt = min(4, len(idx)//8)
                        cols = [y_lag.loc[idx], x_lag.loc[idx]]
                        for p in range(1, p_opt+1):
                            cols.append(dy.shift(p).loc[idx])
                            cols.append(dx.shift(p).loc[idx])
                        exog = add_constant(pd.concat(cols, axis=1).dropna())
                        endog = endog.loc[exog.index]
                        fit_u = OLS(endog, exog).fit()
                        fit_r = OLS(endog, exog.iloc[:,2:]).fit()
                        f_stat = ((fit_r.ssr - fit_u.ssr)/2) / (fit_u.ssr/fit_u.df_resid)
                        p_val = 1 - f_dist.cdf(f_stat, 2, fit_u.df_resid)
                        cv_i0 = PSS["5%"]["I0"]; cv_i1 = PSS["5%"]["I1"]
                        if f_stat > cv_i1: k = "🟢 Eşbütünleşme Var"
                        elif f_stat < cv_i0: k = "🔴 Eşbütünleşme Yok"
                        else: k = "🟡 Sonuçsuz Bölge"
                        rows.append({"Y (Bağımlı)": y_name, "X (Bağımsız)": x_name,
                                     "F-stat": round(f_stat,4), "p-value": round(p_val,4),
                                     "CV I(0) 5%": cv_i0, "CV I(1) 5%": cv_i1, "Karar": k})
                    except Exception as e:
                        rows.append({"Y (Bağımlı)": y_name, "X (Bağımsız)": x_name, "Karar": f"Hata: {e}"})
            df_res = pd.DataFrame(rows)
            st.subheader("ARDL Bounds Testi — Pesaran, Shin & Smith (2001)")
            st.caption(f"H0: Seviyede uzun dönem ilişki yoktur  |  α = {alpha}")
            st.dataframe(df_res, use_container_width=True, hide_index=True)
            show_interpretation("ARDL Bounds", df_res)
            results_export["ARDL_Bounds"] = df_res

        if results_export:
            excel_bytes = df_to_excel(results_export)
            st.download_button("⬇️ Excel'e Aktar", data=excel_bytes,
                               file_name=f"esbütünlesme_{test_e}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")