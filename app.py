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




# ─────────────────────────────────────────
# ZAMaN SERİSİ GRAFİĞİ
# ─────────────────────────────────────────
import plotly.graph_objects as go
import plotly.express as px

PALETTE = ["#4fc3f7", "#1a6a7a", "#81c784", "#f3b235", "#d264c8", "#e67832"]

def show_series_charts(variables: dict, selected: list = None):
    """Seçili veya tüm değişkenler için zaman serisi grafiği gösterir."""
    cols_to_plot = selected if selected else list(variables.keys())
    if not cols_to_plot:
        return
    fig = go.Figure()
    for i, var in enumerate(cols_to_plot):
        s = variables[var].dropna()
        fig.add_trace(go.Scatter(
            x=s.index.astype(str),
            y=s.values,
            name=var,
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.8),
            mode="lines",
            hovertemplate=f"<b>{var}</b><br>%{{x}}<br>%{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor="#0b1520",
        plot_bgcolor="#0f1d2b",
        font=dict(color="#e8f0f8", size=12),
        legend=dict(
            bgcolor="#132233",
            bordercolor="rgba(79,195,247,0.2)",
            borderwidth=1,
            font=dict(size=11, color="#c8dce8"),
        ),
        xaxis=dict(
            gridcolor="rgba(79,195,247,0.08)",
            linecolor="rgba(79,195,247,0.2)",
            tickfont=dict(size=10, color="#7ab8d4"),
            title=None,
        ),
        yaxis=dict(
            gridcolor="rgba(79,195,247,0.08)",
            linecolor="rgba(79,195,247,0.2)",
            tickfont=dict(size=10, color="#7ab8d4"),
            title=None,
        ),
        margin=dict(l=48, r=16, t=16, b=40),
        height=280,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────
# KORELASYON MATRİSİ
# ─────────────────────────────────────────
def show_correlation_heatmap(variables: dict):
    """Değişkenler arası korelasyon matrisini heatmap olarak gösterir."""
    if len(variables) < 2:
        return
    df_corr = pd.DataFrame(variables).corr().round(3)
    vars_list = list(df_corr.columns)

    fig = go.Figure(data=go.Heatmap(
        z=df_corr.values,
        x=vars_list,
        y=vars_list,
        colorscale=[
            [0.0,  "#0b1520"],
            [0.25, "#0e4d5c"],
            [0.5,  "#1a6a7a"],
            [0.75, "#2a7aaa"],
            [1.0,  "#4fc3f7"],
        ],
        zmid=0,
        text=df_corr.values,
        texttemplate="%{text}",
        textfont=dict(size=11, color="#e8f0f8"),
        hovertemplate="%{y} × %{x}<br>Korelasyon: %{z:.3f}<extra></extra>",
        showscale=True,
        colorbar=dict(
            tickfont=dict(color="#7ab8d4", size=10),
            outlinecolor="rgba(79,195,247,0.2)",
            outlinewidth=1,
            thickness=12,
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#0b1520",
        plot_bgcolor="#0f1d2b",
        font=dict(color="#e8f0f8", size=11),
        xaxis=dict(tickfont=dict(size=10, color="#7ab8d4"), side="bottom"),
        yaxis=dict(tickfont=dict(size=10, color="#7ab8d4"), autorange="reversed"),
        margin=dict(l=16, r=16, t=16, b=16),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────
# TEST İSTATİSTİĞİ GÖRSELLEŞTİRME
# ─────────────────────────────────────────
def show_test_stat_chart(df_res: pd.DataFrame, test_name: str):
    """
    Test istatistiklerini kritik değer çizgileriyle birlikte bar grafiği olarak gösterir.
    Desteklenen testler: ADF, PP, KPSS, DF-GLS, Granger, Toda-Yamamoto, Hsiao,
                         Engle-Granger, Phillips-Ouliaris, ARDL Bounds
    """
    # Sütun tespiti
    stat_col  = next((c for c in df_res.columns if "İstatistik" in c or "stat" in c.lower() or "Chi2" in c or "F-stat" in c), None)
    crit5_col = next((c for c in df_res.columns if "5%" in c), None)
    crit1_col = next((c for c in df_res.columns if "1%" in c), None)
    label_col = next((c for c in ["Değişken","X → Y","Y ~ X","Y (Bağımlı)"] if c in df_res.columns), None)
    model_col = "Model" if "Model" in df_res.columns else None

    if stat_col is None:
        return

    # Etiket oluştur
    def make_label(row):
        parts = []
        if label_col: parts.append(str(row.get(label_col, "")))
        if model_col: parts.append(str(row.get(model_col, "")))
        return " / ".join(p for p in parts if p and p != "nan") or str(row.name)

    df_plot = df_res.copy()
    df_plot["_label"] = df_plot.apply(make_label, axis=1)

    try:
        df_plot["_stat"] = pd.to_numeric(df_plot[stat_col], errors="coerce")
    except Exception:
        return

    df_plot = df_plot.dropna(subset=["_stat"])
    if df_plot.empty:
        return

    # Karar rengi
    def bar_color(row):
        karar = str(row.get("Karar",""))
        if "🟢" in karar or "Ret" in karar or "Var" in karar:
            return "#1a6a7a"
        elif "🔴" in karar or "Değil" in karar or "Yok" in karar:
            return "#3a5a6a"
        return "#2a7aaa"

    colors = [bar_color(row) for _, row in df_plot.iterrows()]

    fig = go.Figure()

    # Bar: test istatistikleri
    fig.add_trace(go.Bar(
        x=df_plot["_label"],
        y=df_plot["_stat"],
        marker_color=colors,
        marker_line_color="rgba(79,195,247,0.3)",
        marker_line_width=1,
        name="Test İstatistiği",
        hovertemplate="%{x}<br>İstatistik: %{y:.4f}<extra></extra>",
    ))

    # Kritik değer çizgileri
    if crit5_col and df_plot[crit5_col].notna().any():
        try:
            cv5 = pd.to_numeric(df_plot[crit5_col], errors="coerce").median()
            fig.add_hline(
                y=cv5,
                line_color="#f3b235",
                line_dash="dash",
                line_width=1.5,
                annotation_text=f"Krit. %5 ({cv5:.3f})",
                annotation_font=dict(color="#f3b235", size=10),
                annotation_position="top right",
            )
        except Exception:
            pass

    if crit1_col and df_plot[crit1_col].notna().any():
        try:
            cv1 = pd.to_numeric(df_plot[crit1_col], errors="coerce").median()
            fig.add_hline(
                y=cv1,
                line_color="#e67832",
                line_dash="dot",
                line_width=1.5,
                annotation_text=f"Krit. %1 ({cv1:.3f})",
                annotation_font=dict(color="#e67832", size=10),
                annotation_position="bottom right",
            )
        except Exception:
            pass

    fig.update_layout(
        paper_bgcolor="#0b1520",
        plot_bgcolor="#0f1d2b",
        font=dict(color="#e8f0f8", size=11),
        xaxis=dict(
            tickfont=dict(size=9, color="#7ab8d4"),
            gridcolor="rgba(79,195,247,0.06)",
            linecolor="rgba(79,195,247,0.15)",
            tickangle=-30,
        ),
        yaxis=dict(
            gridcolor="rgba(79,195,247,0.06)",
            linecolor="rgba(79,195,247,0.15)",
            tickfont=dict(size=10, color="#7ab8d4"),
            title=dict(text="Test İstatistiği", font=dict(size=10, color="#7ab8d4")),
            zeroline=True,
            zerolinecolor="rgba(79,195,247,0.2)",
            zerolinewidth=1,
        ),
        legend=dict(
            bgcolor="#132233",
            bordercolor="rgba(79,195,247,0.2)",
            borderwidth=1,
            font=dict(size=10, color="#c8dce8"),
        ),
        margin=dict(l=48, r=16, t=16, b=80),
        height=300,
        bargap=0.3,
    )
    st.plotly_chart(fig, use_container_width=True)


warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# SAYFA AYARLARI
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Ekonometrik Test Paketi | M. Sabri Doğruyol",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# ÖZEL CSS — Site temasına uygun
# ─────────────────────────────────────────
st.markdown("""
<style>
/* ── GENEL ARKAPLAN ── */
.stApp {
    background-color: #0b1520;
    color: #e8f0f8;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background-color: #0f1d2b !important;
    border-right: 1px solid rgba(79,195,247,0.15);
}
[data-testid="stSidebar"] * {
    color: #e8f0f8 !important;
}
[data-testid="stSidebar"] .stSlider > div > div {
    background: rgba(79,195,247,0.2) !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #132233 !important;
    border-color: rgba(79,195,247,0.3) !important;
}

/* ── TAB STİLİ ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #0f1d2b;
    border-bottom: 1px solid rgba(79,195,247,0.2);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #7ab8d4 !important;
    border: 1px solid transparent;
    border-radius: 4px 4px 0 0;
    padding: 8px 20px;
    font-size: 13px;
    letter-spacing: 0.5px;
}
.stTabs [aria-selected="true"] {
    background-color: #132233 !important;
    color: #4fc3f7 !important;
    border-color: rgba(79,195,247,0.3) !important;
    border-bottom-color: #132233 !important;
}

/* ── DATAFRAME / TABLO ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(79,195,247,0.2);
    border-radius: 4px;
}
[data-testid="stDataFrame"] table {
    background-color: #0f1d2b !important;
}
[data-testid="stDataFrame"] th {
    background-color: #132233 !important;
    color: #4fc3f7 !important;
    border-bottom: 1px solid rgba(79,195,247,0.3) !important;
    font-size: 12px;
    letter-spacing: 0.5px;
}
[data-testid="stDataFrame"] td {
    color: #e8f0f8 !important;
    background-color: #0f1d2b !important;
    border-color: rgba(79,195,247,0.1) !important;
    font-size: 13px;
}
[data-testid="stDataFrame"] tr:hover td {
    background-color: #132233 !important;
}

/* ── BUTONLAR ── */
.stButton > button {
    background-color: #1a6a7a;
    color: #e8f0f8;
    border: 1px solid rgba(79,195,247,0.3);
    border-radius: 3px;
    font-size: 12px;
    letter-spacing: 1px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background-color: #4fc3f7;
    color: #0b1520;
    border-color: #4fc3f7;
}

/* ── DOWNLOAD BUTONU ── */
[data-testid="stDownloadButton"] > button {
    background-color: transparent;
    color: #4fc3f7;
    border: 1px solid rgba(79,195,247,0.4);
    border-radius: 3px;
    font-size: 12px;
    letter-spacing: 1px;
}
[data-testid="stDownloadButton"] > button:hover {
    background-color: rgba(79,195,247,0.1);
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background-color: #132233;
    border: 1px dashed rgba(79,195,247,0.3);
    border-radius: 4px;
}

/* ── SELECT BOX / SLIDER ── */
.stSelectbox > div > div {
    background-color: #132233 !important;
    border-color: rgba(79,195,247,0.3) !important;
    color: #e8f0f8 !important;
}
.stSlider [data-testid="stThumbValue"] {
    color: #4fc3f7 !important;
}

/* ── SPINNER ── */
.stSpinner > div {
    border-top-color: #4fc3f7 !important;
}

/* ── BAŞLIK & CAPTION ── */
h1, h2, h3, h4 {
    color: #e8f0f8 !important;
    font-weight: 500;
}
.stCaption {
    color: #7ab8d4 !important;
}

/* ── MARKDOWN YORUMLAR ── */
.stMarkdown p, .stMarkdown li {
    color: #c8dce8 !important;
    line-height: 1.8;
}
.stMarkdown strong {
    color: #e8f0f8 !important;
}
.stMarkdown blockquote {
    border-left: 3px solid #1a6a7a;
    background: rgba(26,106,122,0.1);
    padding: 8px 16px;
    border-radius: 0 4px 4px 0;
    color: #7ab8d4 !important;
}
.stMarkdown code {
    background: #132233;
    color: #4fc3f7;
    padding: 2px 6px;
    border-radius: 3px;
}
hr {
    border-color: rgba(79,195,247,0.15) !important;
}

/* ── INFO / WARNING / ERROR ── */
[data-testid="stAlert"] {
    background-color: #132233 !important;
    border-color: rgba(79,195,247,0.3) !important;
    color: #e8f0f8 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# BAŞLIK ALANI
# ─────────────────────────────────────────
import base64, os

def get_profile_img():
    """Profil fotoğrafını base64 olarak döndürür."""
    img_path = "profil_daire.png"
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

_prof_b64 = get_profile_img()
_prof_html = (
    f'<img src="data:image/png;base64,{_prof_b64}" style="' +
    'width:52px;height:52px;border-radius:50%;border:2px solid rgba(79,195,247,0.4);' +
    'object-fit:cover;object-position:center top;">' 
) if _prof_b64 else '<div style="width:52px;height:52px;border-radius:50%;background:#1a6a7a;border:2px solid rgba(79,195,247,0.4);"></div>'

st.markdown(f"""
<div style="
    padding: 32px 0 24px 0;
    border-bottom: 1px solid rgba(79,195,247,0.15);
    margin-bottom: 24px;
">
    <div style="
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 8px;
    ">
        {_prof_html}
        <div>
            <div style="font-size: 22px; font-weight: 600; color: #e8f0f8; letter-spacing: -0.3px;">
                Ekonometrik Test Paketi
            </div>
            <div style="font-size: 12px; color: #7ab8d4; letter-spacing: 2px; text-transform: uppercase; margin-top: 2px;">
                M. Sabri Doğruyol &nbsp;·&nbsp; msabridogruyol.com
            </div>
        </div>
    </div>
    <div style="
        display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap;
    ">
        <span style="font-size:11px; color:#4fc3f7; border:1px solid rgba(79,195,247,0.25);
                     padding:3px 10px; border-radius:2px; letter-spacing:1px;">
            Durağanlık
        </span>
        <span style="font-size:11px; color:#4fc3f7; border:1px solid rgba(79,195,247,0.25);
                     padding:3px 10px; border-radius:2px; letter-spacing:1px;">
            Nedensellik
        </span>
        <span style="font-size:11px; color:#4fc3f7; border:1px solid rgba(79,195,247,0.25);
                     padding:3px 10px; border-radius:2px; letter-spacing:1px;">
            Eşbütünleşme
        </span>
        <span style="font-size:11px; color:#7ab8d4; border:1px solid rgba(79,195,247,0.15);
                     padding:3px 10px; border-radius:2px; letter-spacing:1px;">
            15 Test
        </span>
        <span style="font-size:11px; color:#7ab8d4; border:1px solid rgba(79,195,247,0.15);
                     padding:3px 10px; border-radius:2px; letter-spacing:1px;">
            Otomatik Yorum
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# AYARLAR (sidebar)
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 16px 0 8px 0; border-bottom: 1px solid rgba(79,195,247,0.15); margin-bottom: 16px;">
        <div style="font-size:13px; font-weight:600; color:#e8f0f8; letter-spacing:0.5px;">
            📊 Ekonometrik Test Paketi
        </div>
        <div style="font-size:10px; color:#7ab8d4; letter-spacing:2px; margin-top:3px;">
            M. SABRI DOĞRUYOL
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Veri Yükle**")
    uploaded = st.file_uploader("Excel dosyası (.xlsx)", type=["xlsx", "xls"],
                                 label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Test Parametreleri**")
    max_lag = st.slider("Maksimum Gecikme", 1, 20, 12,
                        help="ADF, KPSS ve diğer testler için maksimum gecikme sayısı")
    alpha   = st.selectbox("Anlamlılık Düzeyi (α)", [0.01, 0.05, 0.10], index=1,
                            help="H0 reddedilmesi için eşik p-değeri")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px; color:#3a5a6a; line-height:1.8;">
        <div style="color:#4fc3f7; font-size:10px; letter-spacing:2px; margin-bottom:8px;">
            KULLANIM
        </div>
        1. Excel dosyasını yükle<br>
        2. Sekme seç (Durağanlık / Nedensellik / Eşbütünleşme)<br>
        3. Test ve değişken seç<br>
        4. ▶ Çalıştır butonuna bas
    </div>
    """, unsafe_allow_html=True)



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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📐 Durağanlık", "🔗 Nedensellik", "⚖️ Eşbütünleşme", "📈 Tahmin Modelleri", "🗂️ Panel Veri"])

# ═══════════════════════════════════════════════════════
# TAB 1: DURAĞANLIK
# ═══════════════════════════════════════════════════════
with tab1:
    from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews

    test_d = st.selectbox("Test Seç", ["ADF", "PP", "KPSS", "DF-GLS", "Zivot-Andrews", "Fourier FADF/FKPSS"], key="dur_test")
    if test_d == "Fourier FADF/FKPSS":
        fourier_k_d = st.slider("Fourier Frekans Sayısı (k)", 1, 5, 1, key="fk_dur")
    run_d  = st.button("▶ Testi Çalıştır", key="run_dur")

    if run_d:
        results_export = {}
        show_series_charts(VARIABLES)

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
            show_test_stat_chart(df_res, "ADF")
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
                show_test_stat_chart(df_res, "PP")
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
            show_test_stat_chart(df_res, "KPSS")
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
                show_test_stat_chart(df_res, "DF-GLS")
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
            show_test_stat_chart(df_res, "Zivot-Andrews")
            show_interpretation("Zivot-Andrews", df_res)
            results_export["Zivot-Andrews"] = df_res

        elif test_d == "Fourier FADF/FKPSS":
            from statsmodels.regression.linear_model import OLS
            from statsmodels.tools.tools import add_constant
            show_series_charts(VARIABLES)
            show_correlation_heatmap(VARIABLES)
            rows_fadf, rows_fkpss = [], []
            with st.spinner("Fourier testleri hesaplanıyor..."):
                for var, series in VARIABLES.items():
                    s = series.dropna().values
                    T = len(s)
                    t = np.arange(1, T+1)
                    cos_t = np.cos(2 * np.pi * fourier_k_d * t / T)
                    sin_t = np.sin(2 * np.pi * fourier_k_d * t / T)
                    X = add_constant(np.column_stack([t, cos_t, sin_t]))
                    try:
                        resid = OLS(s, X).fit().resid
                        stat, p, lag, nobs, crit, _ = adfuller(resid, maxlag=max_lag, autolag="AIC", regression="nc")
                        rows_fadf.append({"Değişken": var, "k": fourier_k_d,
                            "İstatistik": round(stat,4), "p-value": round(p,4),
                            "Optimal Lag": lag, "n": nobs,
                            "Krit.1%": round(crit["1%"],3), "Krit.5%": round(crit["5%"],3),
                            "Krit.10%": round(crit["10%"],3),
                            "Karar": karar_badge(p)})
                    except Exception as e:
                        rows_fadf.append({"Değişken": var, "Karar": f"Hata: {e}"})
                    try:
                        resid_k = OLS(s, X).fit().resid
                        stat_k, p_k, lag_k, crit_k = kpss(resid_k, regression="c", nlags="auto")
                        rows_fkpss.append({"Değişken": var, "k": fourier_k_d,
                            "İstatistik": round(stat_k,4), "p-value": round(p_k,4), "Lag": lag_k,
                            "Krit.1%": round(crit_k["1%"],3), "Krit.5%": round(crit_k["5%"],3),
                            "Krit.10%": round(crit_k["10%"],3),
                            "Karar": karar_badge(p_k, reverse=True)})
                    except Exception as e:
                        rows_fkpss.append({"Değişken": var, "Karar": f"Hata: {e}"})

            df_fadf  = pd.DataFrame(rows_fadf)
            df_fkpss = pd.DataFrame(rows_fkpss)
            st.subheader(f"Fourier ADF (FADF) — k={fourier_k_d}")
            st.caption(f"H0: Birim kök vardır  |  Fourier detrending  |  α = {alpha}")
            st.dataframe(df_fadf, use_container_width=True, hide_index=True)
            show_test_stat_chart(df_fadf, "ADF")
            show_interpretation("ADF", df_fadf)
            st.subheader(f"Fourier KPSS (FKPSS) — k={fourier_k_d}")
            st.caption(f"H0: Seri durağandır  |  p < α → Durağan değil  |  α = {alpha}")
            st.dataframe(df_fkpss, use_container_width=True, hide_index=True)
            show_test_stat_chart(df_fkpss, "KPSS")
            show_interpretation("KPSS", df_fkpss)
            results_export["FADF"]  = df_fadf
            results_export["FKPSS"] = df_fkpss

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
            show_test_stat_chart(df_res, "Granger")
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
            show_test_stat_chart(df_res, "Toda-Yamamoto")
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
            show_test_stat_chart(df_res, "Hsiao")
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
            show_test_stat_chart(df_res, "ECM Nedensellik")
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
                show_test_stat_chart(df_res, "Breitung-Candelon")
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
            show_test_stat_chart(df_res, "Engle-Granger Eşbütünleşme")
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
                        show_test_stat_chart(df_res, "Johansen")
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
            show_test_stat_chart(df_res, "Phillips-Ouliaris")
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
            show_test_stat_chart(df_res, "Gregory-Hansen")
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
            show_test_stat_chart(df_res, "ARDL Bounds")
            show_interpretation("ARDL Bounds", df_res)
            results_export["ARDL_Bounds"] = df_res

        if results_export:
            excel_bytes = df_to_excel(results_export)
            st.download_button("⬇️ Excel'e Aktar", data=excel_bytes,
                               file_name=f"esbütünlesme_{test_e}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ═══════════════════════════════════════════════════════
# TAB 4: TAHMİN MODELLERİ
# ═══════════════════════════════════════════════════════
with tab4:
    st.markdown("**Model Seç**")
    test_t = st.selectbox("", [
        "ARIMA / SARIMA",
        "VAR (Vektör Otoregresyon)",
        "VECM (Hata Düzeltme Modeli)",
        "ARCH / GARCH",
    ], key="tah_test", label_visibility="collapsed")

    if test_t in ("ARIMA / SARIMA",):
        var_arima = st.selectbox("Değişken", VAR_NAMES, key="arima_var")
        col1, col2, col3 = st.columns(3)
        with col1: p_order = st.slider("p (AR)", 0, 5, 1)
        with col2: d_order = st.slider("d (Fark)", 0, 2, 1)
        with col3: q_order = st.slider("q (MA)", 0, 5, 1)
        sarima_on = st.checkbox("SARIMA (Mevsimsel)", value=False)
        if sarima_on:
            col4, col5, col6, col7 = st.columns(4)
            with col4: P_s = st.slider("P", 0, 2, 1)
            with col5: D_s = st.slider("D", 0, 1, 1)
            with col6: Q_s = st.slider("Q", 0, 2, 1)
            with col7: m_s = st.slider("m (Periyot)", 2, 52, 12)
        else:
            P_s, D_s, Q_s, m_s = 0, 0, 0, 0
        forecast_h = st.slider("Öngörü Ufku (dönem)", 1, 36, 12)

    elif test_t == "VAR (Vektör Otoregresyon)":
        selected_var_vars = st.multiselect("Değişkenler (en az 2)", VAR_NAMES, default=VAR_NAMES[:min(3,len(VAR_NAMES))], key="var_vars")
        max_lag_var = st.slider("Maksimum Gecikme", 1, 12, 4, key="var_lag")
        forecast_h_var = st.slider("Öngörü Ufku (dönem)", 1, 24, 8, key="var_fh")

    elif test_t == "VECM (Hata Düzeltme Modeli)":
        selected_vecm_vars = st.multiselect("Değişkenler (en az 2)", VAR_NAMES, default=VAR_NAMES[:min(3,len(VAR_NAMES))], key="vecm_vars")
        vecm_k = st.slider("Gecikme (k)", 1, 8, 2, key="vecm_k")
        vecm_r = st.slider("Koentegrasyon Rankı (r)", 1, max(1,len(VAR_NAMES)-1), 1, key="vecm_r")
        forecast_h_vecm = st.slider("Öngörü Ufku (dönem)", 1, 24, 8, key="vecm_fh")

    elif test_t == "ARCH / GARCH":
        var_garch = st.selectbox("Değişken", VAR_NAMES, key="garch_var")
        col1, col2 = st.columns(2)
        with col1: garch_p = st.slider("p (GARCH)", 1, 3, 1, key="gp")
        with col2: garch_q = st.slider("q (ARCH)", 1, 3, 1, key="gq")
        garch_dist = st.selectbox("Hata Dağılımı", ["normal","t","skewt"], key="gdist")


    run_t = st.button("▶ Modeli Çalıştır", key="run_tah")

    if run_t:
        results_export = {}

        # ── ARIMA / SARIMA ──
        if test_t == "ARIMA / SARIMA":
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            series = VARIABLES[var_arima].dropna()
            show_series_charts({var_arima: series})
            try:
                with st.spinner("Model tahmin ediliyor..."):
                    if sarima_on:
                        model = SARIMAX(series, order=(p_order,d_order,q_order),
                                        seasonal_order=(P_s,D_s,Q_s,m_s),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
                        model_name = f"SARIMA({p_order},{d_order},{q_order})({P_s},{D_s},{Q_s})[{m_s}]"
                    else:
                        model = ARIMA(series, order=(p_order,d_order,q_order))
                        model_name = f"ARIMA({p_order},{d_order},{q_order})"

                    fit = model.fit()
                    forecast = fit.get_forecast(steps=forecast_h)
                    fc_mean = forecast.predicted_mean
                    fc_ci   = forecast.conf_int()

                # Model özeti
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("AIC", f"{fit.aic:.2f}")
                col2.metric("BIC", f"{fit.bic:.2f}")
                col3.metric("Log-Likelihood", f"{fit.llf:.2f}")
                col4.metric("Model", model_name)

                # Öngörü grafiği
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=series.index.astype(str), y=series.values,
                    name="Gözlem", line=dict(color="#4fc3f7", width=1.5)
                ))
                fc_idx = [str(i) for i in range(len(series), len(series)+forecast_h)]
                fig.add_trace(go.Scatter(
                    x=fc_idx, y=fc_mean.values,
                    name="Öngörü", line=dict(color="#f3b235", width=2, dash="dash")
                ))
                fig.add_trace(go.Scatter(
                    x=fc_idx + fc_idx[::-1],
                    y=list(fc_ci.iloc[:,1].values) + list(fc_ci.iloc[:,0].values[::-1]),
                    fill="toself", fillcolor="rgba(243,178,53,0.1)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="%95 Güven Aralığı"
                ))
                fig.update_layout(
                    paper_bgcolor="#0b1520", plot_bgcolor="#0f1d2b",
                    font=dict(color="#e8f0f8", size=11),
                    xaxis=dict(gridcolor="rgba(79,195,247,0.06)", tickfont=dict(size=9, color="#7ab8d4")),
                    yaxis=dict(gridcolor="rgba(79,195,247,0.06)", tickfont=dict(size=10, color="#7ab8d4")),
                    legend=dict(bgcolor="#132233", bordercolor="rgba(79,195,247,0.2)", borderwidth=1),
                    margin=dict(l=48,r=16,t=16,b=40), height=320,
                )
                st.subheader(f"{model_name} — Öngörü")
                st.plotly_chart(fig, use_container_width=True)

                # Artık grafiği
                resid = fit.resid
                fig2 = go.Figure(go.Scatter(
                    x=series.index.astype(str), y=resid.values,
                    mode="lines", line=dict(color="#1a6a7a", width=1),
                    name="Artıklar"
                ))
                fig2.add_hline(y=0, line_color="rgba(79,195,247,0.3)", line_dash="dash")
                fig2.update_layout(
                    paper_bgcolor="#0b1520", plot_bgcolor="#0f1d2b",
                    font=dict(color="#e8f0f8",size=11),
                    xaxis=dict(gridcolor="rgba(79,195,247,0.06)", tickfont=dict(size=9, color="#7ab8d4")),
                    yaxis=dict(gridcolor="rgba(79,195,247,0.06)", tickfont=dict(size=10, color="#7ab8d4")),
                    margin=dict(l=48,r=16,t=16,b=40), height=200, title=None
                )
                st.subheader("Artıklar")
                st.plotly_chart(fig2, use_container_width=True)

                # Öngörü tablosu
                fc_df = pd.DataFrame({
                    "Dönem": range(1, forecast_h+1),
                    "Öngörü": fc_mean.values.round(4),
                    "Alt %95": fc_ci.iloc[:,0].values.round(4),
                    "Üst %95": fc_ci.iloc[:,1].values.round(4),
                })
                st.subheader("Öngörü Tablosu")
                st.dataframe(fc_df, use_container_width=True, hide_index=True)
                results_export["ARIMA_Öngörü"] = fc_df

            except Exception as e:
                st.error(f"Model hatası: {e}")

        # ── VAR ──
        elif test_t == "VAR (Vektör Otoregresyon)":
            if len(selected_var_vars) < 2:
                st.error("En az 2 değişken seçin.")
            else:
                from statsmodels.tsa.vector_ar.var_model import VAR as VARModel
                data_var = df_raw[selected_var_vars].dropna()
                show_series_charts({v: data_var[v] for v in selected_var_vars})
                try:
                    with st.spinner("VAR modeli tahmin ediliyor..."):
                        model = VARModel(data_var)
                        lag_res = model.select_order(maxlags=max_lag_var)
                        opt_lag = max(1, lag_res.aic)
                        fit = model.fit(opt_lag)
                        forecast = fit.forecast(data_var.values[-opt_lag:], steps=forecast_h_var)
                        fc_df = pd.DataFrame(forecast, columns=selected_var_vars)

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Optimal Lag (AIC)", opt_lag)
                    col2.metric("AIC", f"{fit.aic:.2f}")
                    col3.metric("Değişken Sayısı", len(selected_var_vars))

                    # Öngörü grafiği
                    fig = go.Figure()
                    for i, var in enumerate(selected_var_vars):
                        s = data_var[var]
                        fig.add_trace(go.Scatter(
                            x=s.index.astype(str), y=s.values,
                            name=var, line=dict(color=PALETTE[i % len(PALETTE)], width=1.5)
                        ))
                        fc_idx = [str(j) for j in range(len(s), len(s)+forecast_h_var)]
                        fig.add_trace(go.Scatter(
                            x=fc_idx, y=fc_df[var].values,
                            name=f"{var} (öngörü)",
                            line=dict(color=PALETTE[i % len(PALETTE)], width=2, dash="dash")
                        ))
                    fig.update_layout(
                        paper_bgcolor="#0b1520", plot_bgcolor="#0f1d2b",
                        font=dict(color="#e8f0f8",size=11),
                        legend=dict(bgcolor="#132233", bordercolor="rgba(79,195,247,0.2)", borderwidth=1, font=dict(size=10)),
                        xaxis=dict(gridcolor="rgba(79,195,247,0.06)", tickfont=dict(size=9, color="#7ab8d4")),
                        yaxis=dict(gridcolor="rgba(79,195,247,0.06)", tickfont=dict(size=10, color="#7ab8d4")),
                        margin=dict(l=48,r=16,t=16,b=40), height=320,
                    )
                    st.subheader(f"VAR({opt_lag}) — Öngörü")
                    st.plotly_chart(fig, use_container_width=True)
                    st.subheader("Öngörü Tablosu")
                    st.dataframe(fc_df.round(4), use_container_width=True, hide_index=True)
                    results_export["VAR_Öngörü"] = fc_df.round(4)

                except Exception as e:
                    st.error(f"VAR hatası: {e}")

        # ── VECM ──
        elif test_t == "VECM (Hata Düzeltme Modeli)":
            if len(selected_vecm_vars) < 2:
                st.error("En az 2 değişken seçin.")
            else:
                from statsmodels.tsa.vector_ar.vecm import VECM
                data_vecm = df_raw[selected_vecm_vars].dropna()
                show_series_charts({v: data_vecm[v] for v in selected_vecm_vars})
                try:
                    with st.spinner("VECM tahmin ediliyor..."):
                        model = VECM(data_vecm, k_ar_diff=vecm_k, coint_rank=vecm_r)
                        fit = model.fit()
                        forecast = fit.predict(steps=forecast_h_vecm)
                        fc_df = pd.DataFrame(forecast, columns=selected_vecm_vars)

                    col1, col2 = st.columns(2)
                    col1.metric("Koentegrasyon Rankı", vecm_r)
                    col2.metric("Gecikme (k)", vecm_k)

                    # Hata düzeltme katsayıları
                    alpha_df = pd.DataFrame(
                        fit.alpha,
                        index=selected_vecm_vars,
                        columns=[f"ECT_{i+1}" for i in range(vecm_r)]
                    ).round(4)
                    st.subheader("Hata Düzeltme Katsayıları (α)")
                    st.dataframe(alpha_df, use_container_width=True)

                    # Öngörü grafiği
                    fig = go.Figure()
                    for i, var in enumerate(selected_vecm_vars):
                        s = data_vecm[var]
                        fig.add_trace(go.Scatter(
                            x=s.index.astype(str), y=s.values,
                            name=var, line=dict(color=PALETTE[i % len(PALETTE)], width=1.5)
                        ))
                        fc_idx = [str(j) for j in range(len(s), len(s)+forecast_h_vecm)]
                        fig.add_trace(go.Scatter(
                            x=fc_idx, y=fc_df[var].values,
                            name=f"{var} (öngörü)",
                            line=dict(color=PALETTE[i % len(PALETTE)], width=2, dash="dash")
                        ))
                    fig.update_layout(
                        paper_bgcolor="#0b1520", plot_bgcolor="#0f1d2b",
                        font=dict(color="#e8f0f8",size=11),
                        legend=dict(bgcolor="#132233", bordercolor="rgba(79,195,247,0.2)", borderwidth=1, font=dict(size=10)),
                        xaxis=dict(gridcolor="rgba(79,195,247,0.06)", tickfont=dict(size=9, color="#7ab8d4")),
                        yaxis=dict(gridcolor="rgba(79,195,247,0.06)", tickfont=dict(size=10, color="#7ab8d4")),
                        margin=dict(l=48,r=16,t=16,b=40), height=320,
                    )
                    st.subheader(f"VECM — Öngörü")
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(fc_df.round(4), use_container_width=True, hide_index=True)
                    results_export["VECM_Öngörü"] = fc_df.round(4)

                except Exception as e:
                    st.error(f"VECM hatası: {e}")

        # ── ARCH / GARCH ──
        elif test_t == "ARCH / GARCH":
            from arch import arch_model
            series = VARIABLES[var_garch].dropna()
            returns = series.pct_change().dropna() * 100
            show_series_charts({f"{var_garch} (getiri %)": returns})
            try:
                with st.spinner("GARCH modeli tahmin ediliyor..."):
                    am = arch_model(returns, vol="Garch", p=garch_p, q=garch_q, dist=garch_dist)
                    res = am.fit(disp="off")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("AIC", f"{res.aic:.2f}")
                col2.metric("BIC", f"{res.bic:.2f}")
                col3.metric("Log-Likelihood", f"{res.loglikelihood:.2f}")
                col4.metric("Model", f"GARCH({garch_p},{garch_q})")

                # Koşullu varyans grafiği
                cond_vol = res.conditional_volatility
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=returns.index.astype(str), y=returns.values,
                    name="Getiri", line=dict(color="#4fc3f7", width=1), opacity=0.6
                ))
                fig.add_trace(go.Scatter(
                    x=cond_vol.index.astype(str), y=cond_vol.values,
                    name="Koşullu Volatilite", line=dict(color="#f3b235", width=2),
                    yaxis="y2"
                ))
                fig.update_layout(
                    paper_bgcolor="#0b1520", plot_bgcolor="#0f1d2b",
                    font=dict(color="#e8f0f8",size=11),
                    legend=dict(bgcolor="#132233", bordercolor="rgba(79,195,247,0.2)", borderwidth=1),
                    xaxis=dict(gridcolor="rgba(79,195,247,0.06)", tickfont=dict(size=9, color="#7ab8d4")),
                    yaxis=dict(title="Getiri (%)", gridcolor="rgba(79,195,247,0.06)", tickfont=dict(size=10, color="#4fc3f7")),
                    yaxis2=dict(title="Volatilite", overlaying="y", side="right", tickfont=dict(size=10, color="#f3b235"), gridcolor="rgba(0,0,0,0)"),
                    margin=dict(l=48,r=64,t=16,b=40), height=340,
                )
                st.subheader(f"GARCH({garch_p},{garch_q}) — Koşullu Volatilite")
                st.plotly_chart(fig, use_container_width=True)

                # Parametre tablosu
                params_df = pd.DataFrame({
                    "Parametre": res.params.index,
                    "Katsayı": res.params.values.round(6),
                    "Std. Hata": res.std_err.values.round(6),
                    "t-stat": res.tvalues.values.round(4),
                    "p-value": res.pvalues.values.round(4),
                })
                st.subheader("Model Parametreleri")
                st.dataframe(params_df, use_container_width=True, hide_index=True)
                results_export["GARCH_Parametreler"] = params_df

            except Exception as e:
                st.error(f"GARCH hatası: {e}")


        if results_export:
            excel_bytes = df_to_excel(results_export)
            st.download_button("⬇️ Excel'e Aktar", data=excel_bytes,
                               file_name="tahmin_modelleri.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ═══════════════════════════════════════════════════════
# TAB 5: PANEL VERİ
# ═══════════════════════════════════════════════════════
with tab5:
    from statsmodels.tsa.stattools import adfuller, coint
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
    st.info("**Panel Veri Formatı:** Excel dosyasında 1. sütun dönem (yıl), 2. sütun birim (ülke/firma ID), 3.+ sütunlar değişkenler olmalıdır.")

    test_p = st.selectbox("Test Seç", [
        "LLC — Levin-Lin-Chu Birim Kök",
        "IPS — Im-Pesaran-Shin Birim Kök",
        "Fisher-ADF Birim Kök",
        "Pedroni Eşbütünleşme",
        "Kao Eşbütünleşme",
    ], key="panel_test")

    run_p = st.button("▶ Testi Çalıştır", key="run_panel")

    if run_p:
        results_export = {}

        # Panel veri formatı kontrolü
        if len(df_raw.columns) < 2:
            st.error("Panel veri için en az 2 sütun gereklidir (birim + değişken).")
        else:
            # Panel veri yapısı: multi-index veya geniş format
            # Geniş format: her sütun bir birim, satırlar dönemler
            # Her sütunu ayrı birim olarak değerlendir
            panel_vars = list(df_raw.columns)
            n_units = len(panel_vars)
            T_obs   = len(df_raw)

            show_series_charts(VARIABLES)

            if test_p == "LLC — Levin-Lin-Chu Birim Kök":
                # LLC: ortak birim kök varsayımı
                # Her birim için ADF t-istatistiği, sonra standartlaştırılmış test
                rows = []
                t_stats = []
                with st.spinner("LLC hesaplanıyor..."):
                    for var in panel_vars:
                        s = df_raw[var].dropna()
                        try:
                            stat, p, lag, nobs, crit, _ = adfuller(s, maxlag=max_lag, autolag="AIC", regression="c")
                            t_stats.append(stat)
                            rows.append({"Birim": var, "ADF Stat": round(stat,4),
                                         "p-value": round(p,4), "Lag": lag, "n": nobs})
                        except Exception as e:
                            rows.append({"Birim": var, "Karar": f"Hata: {e}"})

                if t_stats:
                    # LLC panel istatistiği (basitleştirilmiş): ortalama t / std
                    llc_stat = np.mean(t_stats) * np.sqrt(n_units)
                    from scipy.stats import norm
                    llc_p    = norm.cdf(llc_stat)
                    k_llc    = "🟢 Panel Durağan (H0 Reddedildi)" if llc_p <= alpha else "🔴 Panel Durağan Değil"

                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("LLC Panel İstatistiği", f"{llc_stat:.4f}")
                    col2.metric("p-value", f"{llc_p:.4f}")
                    col3.metric("Karar", k_llc.replace("🟢","").replace("🔴","").strip())

                df_res = pd.DataFrame(rows)
                st.subheader("LLC — Birim Bazlı ADF Sonuçları")
                st.caption("H0: Tüm birimlerde birim kök vardır (ortak birim kök)")
                st.dataframe(df_res, use_container_width=True, hide_index=True)
                show_test_stat_chart(df_res.rename(columns={"ADF Stat":"İstatistik","Birim":"Değişken"}), "LLC")
                results_export["LLC"] = df_res

            elif test_p == "IPS — Im-Pesaran-Shin Birim Kök":
                # IPS: bireysel birim kök varsayımı
                rows = []
                t_bar_list = []
                with st.spinner("IPS hesaplanıyor..."):
                    for var in panel_vars:
                        s = df_raw[var].dropna()
                        try:
                            stat, p, lag, nobs, crit, _ = adfuller(s, maxlag=max_lag, autolag="AIC", regression="c")
                            t_bar_list.append(stat)
                            rows.append({"Birim": var, "ADF Stat": round(stat,4),
                                         "p-value": round(p,4), "Lag": lag,
                                         "Karar": "🟢 Durağan" if p<=alpha else "🔴 Durağan Değil"})
                        except Exception as e:
                            rows.append({"Birim": var, "Karar": f"Hata: {e}"})

                if t_bar_list:
                    t_bar = np.mean(t_bar_list)
                    # IPS W-stat standardizasyonu (yaklaşık)
                    E_t = -1.503  # E[t-bar] için tablo değeri (yaklaşık, lag=1)
                    V_t =  0.745  # Var[t-bar] için tablo değeri (yaklaşık)
                    W_ips = np.sqrt(n_units) * (t_bar - E_t) / np.sqrt(V_t)
                    from scipy.stats import norm
                    ips_p = norm.cdf(W_ips)
                    k_ips = "🟢 En az bir birim durağan (H0 Reddedildi)" if ips_p <= alpha else "🔴 Panel Durağan Değil"

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("t-bar", f"{t_bar:.4f}")
                    col2.metric("W-stat (IPS)", f"{W_ips:.4f}")
                    col3.metric("p-value", f"{ips_p:.4f}")
                    col4.metric("Birim Sayısı", n_units)

                df_res = pd.DataFrame(rows)
                st.subheader("IPS — Birim Bazlı Sonuçlar")
                st.caption("H0: Tüm birimlerde birim kök vardır  |  H1: En az bir birim durağandır")
                st.dataframe(df_res, use_container_width=True, hide_index=True)
                show_test_stat_chart(df_res.rename(columns={"ADF Stat":"İstatistik","Birim":"Değişken"}), "IPS")
                results_export["IPS"] = df_res

            elif test_p == "Fisher-ADF Birim Kök":
                # Fisher: -2 * Σ ln(p_i) ~ Chi2(2N)
                from scipy.stats import chi2 as chi2_dist
                rows = []
                p_vals = []
                with st.spinner("Fisher-ADF hesaplanıyor..."):
                    for var in panel_vars:
                        s = df_raw[var].dropna()
                        try:
                            stat, p, lag, nobs, crit, _ = adfuller(s, maxlag=max_lag, autolag="AIC", regression="c")
                            p_vals.append(p)
                            rows.append({"Birim": var, "ADF Stat": round(stat,4),
                                         "p-value": round(p,4), "Lag": lag,
                                         "Karar": "🟢 Durağan" if p<=alpha else "🔴 Durağan Değil"})
                        except Exception as e:
                            rows.append({"Birim": var, "Karar": f"Hata: {e}"})

                if p_vals:
                    fisher_stat = -2 * np.sum(np.log(np.clip(p_vals, 1e-10, 1)))
                    fisher_p    = 1 - chi2_dist.cdf(fisher_stat, df=2*len(p_vals))
                    k_f = "🟢 H0 Reddedildi — En az bir birim durağan" if fisher_p<=alpha else "🔴 H0 Reddedilemedi"

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Fisher Chi2 İstatistiği", f"{fisher_stat:.4f}")
                    col2.metric("p-value", f"{fisher_p:.4f}")
                    col3.metric("df", 2*len(p_vals))

                df_res = pd.DataFrame(rows)
                st.subheader("Fisher-ADF Panel Birim Kök")
                st.caption("H0: Tüm birimlerde birim kök vardır  |  Test: -2·Σln(pᵢ) ~ χ²(2N)")
                st.dataframe(df_res, use_container_width=True, hide_index=True)
                show_test_stat_chart(df_res.rename(columns={"ADF Stat":"İstatistik","Birim":"Değişken"}), "Fisher-ADF")
                results_export["Fisher_ADF"] = df_res

            elif test_p == "Pedroni Eşbütünleşme":
                if len(panel_vars) < 2:
                    st.error("Pedroni için en az 2 değişken gereklidir.")
                else:
                    from statsmodels.tsa.stattools import coint
                    # Pedroni: her birim çifti için artık tabanlı koentegrasyon
                    dep_var  = st.selectbox("Bağımlı Değişken (Y)", panel_vars, key="ped_y")
                    indep_vars = [v for v in panel_vars if v != dep_var]
                    rows = []
                    with st.spinner("Pedroni hesaplanıyor..."):
                        for xvar in indep_vars:
                            try:
                                stat, p, crit = coint(df_raw[dep_var].dropna(), df_raw[xvar].dropna())
                                rows.append({
                                    "Y ~ X": f"{dep_var} ~ {xvar}",
                                    "İstatistik": round(stat,4), "p-value": round(p,4),
                                    "Krit.1%": round(crit[0],3), "Krit.5%": round(crit[1],3),
                                    "Krit.10%": round(crit[2],3),
                                    "Karar": "🟢 Eşbütünleşme Var" if p<=alpha else "🔴 Eşbütünleşme Yok"
                                })
                            except Exception as e:
                                rows.append({"Y ~ X": f"{dep_var} ~ {xvar}", "Karar": f"Hata: {e}"})

                    df_res = pd.DataFrame(rows)
                    st.subheader("Pedroni Panel Eşbütünleşme")
                    st.caption("H0: Eşbütünleşme yoktur  |  Artık tabanlı panel koentegrasyon")
                    st.dataframe(df_res, use_container_width=True, hide_index=True)
                    show_test_stat_chart(df_res, "Pedroni")
                    results_export["Pedroni"] = df_res

            elif test_p == "Kao Eşbütünleşme":
                if len(panel_vars) < 2:
                    st.error("Kao için en az 2 değişken gereklidir.")
                else:
                    from statsmodels.tsa.stattools import coint
                    from statsmodels.regression.linear_model import OLS
                    from statsmodels.tools.tools import add_constant
                    dep_var_kao = st.selectbox("Bağımlı Değişken (Y)", panel_vars, key="kao_y")
                    indep_kao   = [v for v in panel_vars if v != dep_var_kao]
                    rows = []
                    e_stats = []
                    with st.spinner("Kao hesaplanıyor..."):
                        for xvar in indep_kao:
                            try:
                                y = df_raw[dep_var_kao].dropna()
                                x = df_raw[xvar].dropna()
                                idx = y.index.intersection(x.index)
                                resid = OLS(y.loc[idx], add_constant(x.loc[idx])).fit().resid
                                stat, p, lag, nobs, crit, _ = adfuller(resid, maxlag=max_lag, autolag="AIC", regression="nc")
                                e_stats.append(stat)
                                rows.append({
                                    "Y ~ X": f"{dep_var_kao} ~ {xvar}",
                                    "ADF Stat (artık)": round(stat,4), "p-value": round(p,4),
                                    "Lag": lag,
                                    "Krit.5%": round(crit["5%"],3),
                                    "Karar": "🟢 Eşbütünleşme Var" if p<=alpha else "🔴 Eşbütünleşme Yok"
                                })
                            except Exception as e:
                                rows.append({"Y ~ X": f"{dep_var_kao} ~ {xvar}", "Karar": f"Hata: {e}"})

                    if e_stats:
                        from scipy.stats import norm
                        kao_stat = np.mean(e_stats) * np.sqrt(len(e_stats))
                        kao_p    = norm.cdf(kao_stat)
                        col1, col2 = st.columns(2)
                        col1.metric("Kao Panel İstatistiği", f"{kao_stat:.4f}")
                        col2.metric("p-value", f"{kao_p:.4f}")

                    df_res = pd.DataFrame(rows)
                    st.subheader("Kao Panel Eşbütünleşme")
                    st.caption("H0: Eşbütünleşme yoktur  |  Panel ADF artık tabanlı test")
                    st.dataframe(df_res, use_container_width=True, hide_index=True)
                    show_test_stat_chart(df_res.rename(columns={"ADF Stat (artık)":"İstatistik"}), "Kao")
                    results_export["Kao"] = df_res

        if results_export:
            excel_bytes = df_to_excel(results_export)
            st.download_button("⬇️ Excel'e Aktar", data=excel_bytes,
                               file_name="panel_veri.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
