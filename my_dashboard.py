"""날씨 × 카드 소비 대시보드 (수강생용).

실행: python -m streamlit run my_dashboard.py

- 조인 키: (date, region). 비 오는 날 = 일강수량(rainfall) 3mm 이상.
- 소비지수: 지역 · 업종별 기간(2025년 전체) 평균 금액 = 100.
- 비 오는 날 효과는 단순 비교와 '같은 지역 · 같은 달 · 같은 평일/주말' 통제 비교를 함께 보여준다.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from make_card_data import HOLIDAYS_2025  # 카드 데이터 생성과 같은 주말 · 공휴일 기준
from regions import REGIONS

HERE = Path(__file__).parent
RAIN_MM = 3.0

# 색 (dataviz 기본 팔레트 · 라이트 테마)
BLUE, RED, ORANGE, GRAY = "#257de0", "#ec4647", "#f47230", "#9c9c9c"
INK, INK2, GRID = "#191919", "#666666", "#e7e7e7"
DIVERGING = [[0, RED], [0.5, "#efefef"], [1, BLUE]]  # 100 미만 빨강 · 100 회색 · 100 초과 파랑

st.set_page_config(page_title="날씨 × 카드 소비", page_icon="🌧️", layout="wide")


# ───────────────────────── 데이터 ─────────────────────────
@st.cache_data
def load() -> pd.DataFrame:
    w = pd.read_csv(HERE / "data" / "weather_daily.csv")
    c = pd.read_csv(HERE / "data" / "card_daily.csv")
    df = c.merge(w, on=["date", "region"], how="inner", validate="many_to_one")
    df["date"] = pd.to_datetime(df["date"])
    df["rain"] = df["rainfall"] >= RAIN_MM
    df["month"] = df["date"].dt.month
    df["weekend"] = (df["date"].dt.day_of_week >= 5) | df["date"].isin(HOLIDAYS_2025)
    df["idx"] = df["amount"] / df.groupby(["region", "industry"])["amount"].transform("mean") * 100
    return df, len(w), len(c)


def rain_effect(d: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """by 그룹별 비 오는 날 증감률(%): 단순 비교 + 통제 비교(지역 · 월 · 평일/주말 셀 매칭)."""
    rows = []
    for key, g in d.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        r, o = g.loc[g.rain, "idx"], g.loc[~g.rain, "idx"]
        simple = (r.mean() / o.mean() - 1) * 100 if len(r) and len(o) else np.nan
        t = g.groupby(["region", "month", "weekend", "rain"])["idx"].agg(["mean", "size"]).unstack("rain")
        ctrl = np.nan
        if ("mean", True) in t.columns and ("mean", False) in t.columns:
            t = t.dropna()
            if len(t):
                wgt = t[("size", True)]
                ctrl = ((t[("mean", True)] / t[("mean", False)] - 1) * wgt).sum() / wgt.sum() * 100
        rows.append({**dict(zip(by, key)), "단순": simple, "통제": ctrl,
                     "비오는날": int(g.rain.sum()), "그외날": int((~g.rain).sum())})
    return pd.DataFrame(rows)


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white",
                      paper_bgcolor="white", font=dict(color=INK), hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK2)))
    fig.update_xaxes(gridcolor=GRID, linecolor="#c2c2c2", tickfont=dict(color=INK2), title_font=dict(color=INK2))
    fig.update_yaxes(gridcolor=GRID, linecolor="#c2c2c2", tickfont=dict(color=INK2), title_font=dict(color=INK2))
    return fig


df, n_weather, n_card = load()
ALL_REGIONS = list(REGIONS)
ALL_IND = sorted(df["industry"].unique())

# ───────────────────────── 사이드바 ─────────────────────────
with st.sidebar:
    st.header("필터")
    regions = st.multiselect("지역", ALL_REGIONS, default=ALL_REGIONS)
    industries = st.multiselect("업종 (KPI · 막대 · 주간 차트)", ALL_IND, default=ALL_IND)
    focus = st.selectbox("선택 업종 (지도 · 산점도)", industries or ALL_IND,
                         index=(industries or ALL_IND).index("배달") if "배달" in (industries or ALL_IND) else 0)
    d0, d1 = df["date"].min().date(), df["date"].max().date()
    period = st.slider("기간", min_value=d0, max_value=d1, value=(d0, d1), format="YYYY-MM-DD")
    st.divider()
    st.subheader("실측 날씨 연동")
    weather = st.radio("소비를 볼 날씨 조건", ["전체", "비 오는 날", "맑은 날"], horizontal=True,
                       help=f"비 오는 날 = 일강수량 {RAIN_MM:g}mm 이상, 맑은 날 = {RAIN_MM:g}mm 미만. "
                            "KPI 1~3과 산점도에 적용된다.")
    st.caption(f"날씨 출처: {', '.join(df['source'].unique())} (재분석 자료)")

if not regions or not industries:
    st.warning("지역과 업종을 하나 이상 선택하세요.")
    st.stop()

base = df[df.region.isin(regions) & df.date.dt.date.between(*period)]
sel = base[base.industry.isin(industries)]
wmask = {"전체": slice(None), "비 오는 날": sel.rain, "맑은 날": ~sel.rain}[weather]
sel_w = sel[wmask] if weather != "전체" else sel

# ───────────────────────── 헤더 ─────────────────────────
st.title("🌧️ 날씨 × 카드 소비 대시보드")
st.caption(
    f"조인 {len(df):,}행 (날씨 {n_weather:,}행 · 카드 {n_card:,}행, 키 (date, region)) · "
    f"현재 필터: {period[0]} ~ {period[1]} · 지역 {len(regions)}개 · 업종 {len(industries)}개 · "
    f"{len(sel):,}행 (날씨 조건 '{weather}' 적용 시 {len(sel_w):,}행)"
)

# ───────────────────────── KPI ─────────────────────────
eff_ind = rain_effect(sel, ["industry"]).dropna(subset=["통제"])
k1, k2, k3, k4 = st.columns(4)
amt, tx = sel_w["amount"].sum(), sel_w["tx_count"].sum()
k1.metric("총 소비금액", f"{amt / 1e12:,.1f}조 원" if amt >= 1e12 else f"{amt / 1e8:,.0f}억 원",
          help="가상 데이터 · 실제 시장 규모 아님")
k2.metric("승인 건수", f"{tx / 1e8:,.1f}억 건" if tx >= 1e8 else f"{tx / 1e4:,.0f}만 건")
k3.metric("객단가 (건당)", f"{amt / tx:,.0f}원" if tx else "-")
if len(eff_ind):
    top = eff_ind.loc[eff_ind["통제"].abs().idxmax()]
    k4.metric("비 오는 날 증감률 1위 업종", top["industry"], f"{top['통제']:+.1f}% (통제)",
              help=f"같은 지역·월·평일/주말 비교. 단순 비교 {top['단순']:+.1f}%")
else:
    k4.metric("비 오는 날 증감률 1위 업종", "-", help="선택 범위에 비 오는 날이 없습니다.")

# ───────────────────────── 지도 ─────────────────────────
st.subheader(f"📍 시도별 {focus} 소비지수")
mode = st.radio("날씨", ["맑은 날", "비 오는 날"], horizontal=True, key="map_mode",
                help="지역·업종 연평균 = 100. 선택한 날씨의 날만 평균낸 지수.")
f = base[base.industry == focus]
m = (f[f.rain == (mode == "비 오는 날")].groupby("region")
     .agg(지수=("idx", "mean"), 일수=("idx", "size")).reset_index())
m = m.merge(rain_effect(f, ["region"])[["region", "통제"]], on="region", how="left")
m["lat"] = m.region.map(lambda r: REGIONS[r]["lat"])
m["lon"] = m.region.map(lambda r: REGIONS[r]["lon"])
if len(m):
    span = max(abs(m["지수"] - 100).max(), 5)
    fig = px.scatter_map(
        m, lat="lat", lon="lon", color="지수", text="region", size=np.full(len(m), 18), size_max=18,
        color_continuous_scale=DIVERGING, range_color=(100 - span, 100 + span),
        hover_name="region", hover_data={"지수": ":.1f", "일수": True, "통제": ":+.1f", "lat": False, "lon": False},
        labels={"통제": "비 효과(통제, %)"}, zoom=5.6, center=dict(lat=35.9, lon=127.8),
        map_style="carto-positron",
    )
    fig.update_traces(textposition="top center", textfont=dict(color=INK, size=12))
    fig.update_layout(coloraxis_colorbar=dict(title="지수"))
    st.plotly_chart(style(fig, 560), width="stretch")
    st.caption(f"{mode} 기준 · 지역별 {focus} 평균 소비지수 (100 = 연평균). 마우스를 올리면 통제 비교 증감률이 보입니다.")
else:
    st.info(f"선택 범위에 {mode}이 없습니다.")

# ───────────────────────── 주간 소비 · 강수 ─────────────────────────
st.subheader("📈 주간 소비와 주간 강수량")
wk = sel.assign(week=sel.date.dt.to_period("W").dt.start_time)
wk_amt = wk.groupby("week")["amount"].sum() / 1e8
wk_rain = (base.drop_duplicates(["date", "region"]).assign(week=lambda x: x.date.dt.to_period("W").dt.start_time)
           .groupby(["week", "region"])["rainfall"].sum().groupby("week").mean())
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.06,
                    subplot_titles=("주간 소비금액 (억 원)", "주간 강수량 (지역 평균, mm)"))
fig.add_trace(go.Scatter(x=wk_amt.index, y=wk_amt.values, mode="lines", line=dict(color=BLUE, width=2),
                         name="소비금액", hovertemplate="%{x|%Y-%m-%d} 주<br>%{y:,.0f}억 원<extra></extra>"), 1, 1)
fig.add_trace(go.Bar(x=wk_rain.index, y=wk_rain.values, marker_color=BLUE, marker_line_width=0, opacity=0.55,
                     name="강수량", hovertemplate="%{x|%Y-%m-%d} 주<br>%{y:.1f}mm<extra></extra>"), 2, 1)
fig.update_layout(showlegend=False, hovermode="x unified", bargap=0.15)
fig.update_annotations(font=dict(color=INK2, size=13), x=0, xanchor="left")
st.plotly_chart(style(fig, 460).update_layout(hovermode="x unified"), width="stretch")
st.caption("두 값의 단위가 달라 y축을 둘로 겹치지 않고 위아래로 나눴습니다. 첫 주 · 마지막 주는 7일이 안 될 수 있습니다.")

# ───────────────────────── 산점도 · 막대 ─────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.subheader(f"🔍 날씨 변수와 {focus} 소비지수")
    var_map = {"강수량 (mm)": "rainfall", "기온 (℃)": "temp", "습도 (%)": "humidity"}
    vlabel = st.selectbox("날씨 변수", list(var_map))
    v = var_map[vlabel]
    s = sel_w[sel_w.industry == focus] if focus in industries else base[base.industry == focus]
    if len(s) > 1:
        fig = go.Figure(go.Scattergl(
            x=s[v], y=s["idx"], mode="markers", name="지역-날짜",
            marker=dict(color=BLUE, size=8, opacity=0.35, line=dict(color="white", width=1)),
            customdata=np.stack([s.region, s.date.dt.strftime("%Y-%m-%d")], axis=1),
            hovertemplate="%{customdata[0]} · %{customdata[1]}<br>" + vlabel + " %{x}<br>지수 %{y:.1f}<extra></extra>"))
        a, b = np.polyfit(s[v], s["idx"], 1)
        xs = np.linspace(s[v].min(), s[v].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=a * xs + b, mode="lines", line=dict(color=ORANGE, width=2),
                                 name="추세선", hoverinfo="skip"))
        if v == "rainfall":
            fig.add_vline(x=RAIN_MM, line=dict(color=GRAY, dash="dot", width=1),
                          annotation_text="3mm", annotation_font_color=INK2)
        fig.update_layout(xaxis_title=vlabel, yaxis_title="소비지수 (100 = 연평균)")
        st.plotly_chart(style(fig), width="stretch")
        r = np.corrcoef(s[v], s["idx"])[0, 1]
        st.caption(f"{len(s):,}개 지역-날짜 · 날씨 조건 '{weather}' · 상관계수 r = {r:+.2f} "
                   "(계절·요일 미통제 단순 상관)")
    else:
        st.info("점이 부족합니다.")

with c2:
    st.subheader("🌂 업종별 비 오는 날 증감률")
    if len(eff_ind):
        e = eff_ind.sort_values("통제")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=e.industry, x=e["단순"], orientation="h", name="단순 비교", marker_color=GRAY,
                             hovertemplate="%{y} 단순 %{x:+.1f}%<extra></extra>"))
        fig.add_trace(go.Bar(y=e.industry, x=e["통제"], orientation="h", name="통제 비교 (지역·월·평일/주말)",
                             marker_color=BLUE, text=e["통제"].map("{:+.1f}%".format), textposition="outside",
                             textfont=dict(color=INK), hovertemplate="%{y} 통제 %{x:+.1f}%<extra></extra>"))
        fig.add_vline(x=0, line=dict(color="#c2c2c2", width=1))
        fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08, xaxis_title="비 오는 날 증감률 (%)",
                          xaxis=dict(range=[min(e[["단순", "통제"]].min().min(), 0) * 1.25,
                                            max(e[["단순", "통제"]].max().max(), 0) * 1.25]))
        st.plotly_chart(style(fig), width="stretch")
        st.caption(f"비 오는 날 {int(eff_ind['비오는날'].iloc[0]):,}행 vs 그 외 {int(eff_ind['그외날'].iloc[0]):,}행 (업종당) · "
                   "통제 = 같은 지역·같은 달·같은 평일/주말 셀끼리 비교 후 비 오는 날 수로 가중평균")
    else:
        st.info("선택 범위에 비 오는 날이 없어 계산할 수 없습니다.")

# ───────────────────────── 추천 액션 ─────────────────────────
st.subheader("✅ 추천 액션")
if len(eff_ind) >= 2:
    eff_reg = rain_effect(sel, ["industry", "region"]).dropna(subset=["통제"])
    up = eff_ind.loc[eff_ind["통제"].idxmax()]
    dn = eff_ind.loc[eff_ind["통제"].idxmin()]
    gap = eff_ind.assign(g=(eff_ind["통제"] - eff_ind["단순"]).abs()).sort_values("g").iloc[-1]

    def consistency(ind: str, positive: bool) -> str:
        r = eff_reg[eff_reg.industry == ind]
        same = int((r["통제"] > 0).sum() if positive else (r["통제"] < 0).sum())
        return f"{same}/{len(r)}개 시도 같은 방향"

    best_reg = eff_reg[eff_reg.industry == up.industry].sort_values("통제").iloc[-1]
    rain_share = base.drop_duplicates(["date", "region"]).rain.mean() * 100
    a1, a2, a3 = st.columns(3)
    with a1, st.container(border=True):
        st.markdown(f"**1. 비 예보 날 {up.industry} 프로모션 강화**")
        st.markdown(f"[확인된 사실] 비 오는 날 {up.industry} 소비 **{up['통제']:+.1f}%** (통제, 단순 {up['단순']:+.1f}%), "
                    f"{consistency(up.industry, up['통제'] > 0)}. 효과가 가장 큰 곳은 **{best_reg.region} "
                    f"({best_reg['통제']:+.1f}%)**.")
        st.markdown(f"[추정] 강수 예보 전날 푸시 · 쿠폰을 {best_reg.region}부터 시험하면 반응이 클 것.")
    with a2, st.container(border=True):
        st.markdown(f"**2. 비 오는 날 {dn.industry} 이탈 방어**")
        st.markdown(f"[확인된 사실] 비 오는 날 {dn.industry} 소비 **{dn['통제']:+.1f}%** (통제, 단순 {dn['단순']:+.1f}%), "
                    f"{consistency(dn.industry, dn['통제'] > 0)}. 선택 범위의 비 오는 날 비중 {rain_share:.1f}%.")
        st.markdown("[추정] 매장 방문이 줄어드는 날이므로 온라인 주문 · 테이크아웃 · 배달 연계 혜택으로 수요를 옮겨 잡는다.")
    with a3, st.container(border=True):
        st.markdown("**3. 날씨 캠페인 성과는 같은 달 · 같은 요일끼리 측정**")
        st.markdown(f"[확인된 사실] {gap.industry}은(는) 단순 비교 {gap['단순']:+.1f}% vs 통제 비교 "
                    f"**{gap['통제']:+.1f}%** — 차이 {gap['통제'] - gap['단순']:+.1f}%p. 비가 특정 계절 · 주말에 몰려서 생긴 착시.")
        st.markdown("[추정] 캠페인 효과를 단순 '비 온 날 vs 안 온 날'로 보면 과대 · 과소평가할 수 있다.")
else:
    st.info("추천 액션은 업종 2개 이상, 비 오는 날이 있는 범위에서 계산됩니다.")

st.divider()
st.caption("한계: 카드 소비는 실습용 가상 데이터로 금액을 실제 시장 규모로 해석하지 않는다. "
           "날씨는 Open-Meteo 재분석 자료(모델 추정치)라 기상청 관측값과 다를 수 있다.")
