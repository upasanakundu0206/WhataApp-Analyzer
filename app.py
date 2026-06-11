import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

import preprocessor
import helper

matplotlib.use('Agg')

st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg,#0f0f1a 0%,#1a1a2e 50%,#16213e 100%); min-height:100vh; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0d1117 0%,#161b22 100%) !important; border-right:1px solid #25D36640; }
[data-testid="stMetric"] { background:linear-gradient(135deg,#1a1a2e,#16213e); border:1px solid #25D36630; border-radius:16px; padding:20px !important; transition:transform 0.2s,box-shadow 0.2s; }
[data-testid="stMetric"]:hover { transform:translateY(-3px); box-shadow:0 8px 30px rgba(37,211,102,0.15); border-color:#25D366; }
[data-testid="stMetricLabel"] { color:#8b949e !important; font-size:12px !important; font-weight:500 !important; text-transform:uppercase; letter-spacing:0.05em; }
[data-testid="stMetricValue"] { color:#25D366 !important; font-size:2rem !important; font-weight:700 !important; }
.stTabs [data-baseweb="tab-list"] { background:#161b22; border-radius:12px; padding:4px; gap:4px; border:1px solid #30363d; }
.stTabs [data-baseweb="tab"] { background:transparent; border-radius:8px; color:#8b949e; font-weight:500; font-size:13px; padding:8px 16px; border:none; transition:all 0.2s; }
.stTabs [data-baseweb="tab"]:hover { background:#1f2937; color:#ffffff; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#25D366,#128C7E) !important; color:white !important; font-weight:600 !important; }
.stTabs [data-baseweb="tab-highlight"] { display:none; }
.stButton > button { background:linear-gradient(135deg,#25D366,#128C7E); color:white; border:none; border-radius:12px; font-weight:600; font-size:14px; padding:12px 24px; transition:all 0.3s; box-shadow:0 4px 15px rgba(37,211,102,0.3); width:100%; }
.stButton > button:hover { transform:translateY(-2px); box-shadow:0 6px 25px rgba(37,211,102,0.4); }
[data-testid="stFileUploader"] { background:#161b22; border:2px dashed #25D36650; border-radius:16px; padding:20px; transition:border-color 0.3s; }
[data-testid="stFileUploader"]:hover { border-color:#25D366; }
[data-testid="stSelectbox"] > div > div { background:#161b22 !important; border:1px solid #30363d !important; border-radius:10px !important; color:white !important; }
.section-title { font-size:1.1rem; font-weight:600; color:#ffffff; margin:1.5rem 0 0.8rem; padding:10px 16px; background:linear-gradient(90deg,#25D36620,transparent); border-left:3px solid #25D366; border-radius:0 8px 8px 0; }
.stat-card { background:linear-gradient(135deg,#1a1a2e,#16213e); border:1px solid #25D36630; border-radius:16px; padding:20px; text-align:center; transition:all 0.3s; }
.stat-card:hover { border-color:#25D366; transform:translateY(-3px); box-shadow:0 8px 25px rgba(37,211,102,0.15); }
.stat-card .icon { font-size:2rem; margin-bottom:8px; }
.stat-card .value { font-size:2rem; font-weight:700; color:#25D366; line-height:1; }
.stat-card .label { font-size:12px; color:#8b949e; text-transform:uppercase; letter-spacing:0.05em; margin-top:6px; }
.predict-card { background:linear-gradient(135deg,#25D366,#128C7E); border-radius:16px; padding:24px; text-align:center; color:white; margin:16px 0; box-shadow:0 8px 30px rgba(37,211,102,0.25); }
.predict-card h3 { font-size:1.6rem; margin:0 0 8px; }
.predict-card p { margin:0; opacity:0.9; font-size:14px; }
.insight-box { background:#161b22; border:1px solid #25D36640; border-radius:12px; padding:16px; text-align:center; transition:all 0.2s; }
.insight-box:hover { border-color:#25D366; box-shadow:0 4px 20px rgba(37,211,102,0.1); }
.hero-banner { background:linear-gradient(135deg,#0d2818 0%,#1a3a2a 50%,#0d2818 100%); border:1px solid #25D36640; border-radius:20px; padding:40px; text-align:center; margin-bottom:24px; }
.hero-banner h1 { color:#25D366; font-size:2.5rem; margin:0 0 8px; }
.hero-banner p { color:#8b949e; font-size:1rem; margin:0; }
[data-testid="stAlert"] { border-radius:12px; border-left-width:4px; }
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; border:1px solid #30363d; }
[data-testid="stExpander"] { background:#161b22; border:1px solid #30363d; border-radius:12px; }
[data-testid="stDownloadButton"] > button { background:linear-gradient(135deg,#1f6feb,#388bfd); box-shadow:0 4px 15px rgba(31,111,235,0.3); }
.sidebar-badge { background:linear-gradient(135deg,#25D36620,#128C7E20); border:1px solid #25D36640; border-radius:10px; padding:12px 16px; margin:8px 0; font-size:13px; color:#c9d1d9; }
.sidebar-badge b { color:#25D366; }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(37,211,102,0.4);} 70%{box-shadow:0 0 0 10px rgba(37,211,102,0);} 100%{box-shadow:0 0 0 0 rgba(37,211,102,0);} }
.wa-logo { animation:pulse 2s infinite; border-radius:50%; display:inline-block; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark layout helper ─────────────────────────────────────────────────
def dk(height=360, **kw):
    base = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(22,27,34,0.8)',
        font=dict(color='#c9d1d9', family='Inter'),
        xaxis=dict(gridcolor='#21262d', linecolor='#30363d'),
        yaxis=dict(gridcolor='#21262d', linecolor='#30363d'),
        legend=dict(bgcolor='rgba(22,27,34,0.8)', bordercolor='#30363d'),
        margin=dict(t=40, b=40, l=10, r=10),
        height=height,
    )
    base.update(kw)
    return base

GREEN_SCALE  = [[0,'#0d2818'],[0.5,'#25D366'],[1,'#dcfce7']]
PURPLE_SCALE = [[0,'#1a103d'],[0.5,'#7c3aed'],[1,'#e9d5ff']]
BLUE_SCALE   = [[0,'#0c1a3d'],[0.5,'#1f6feb'],[1,'#bfdbfe']]
ORANGE_SCALE = [[0,'#2d1a00'],[0.5,'#f59e0b'],[1,'#fef3c7']]
TEAL_SCALE   = [[0,'#0a2a2a'],[0.5,'#14b8a6'],[1,'#ccfbf1']]

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 8px;">
        <div class="wa-logo" style="background:#25D366;width:56px;height:56px;
             border-radius:50%;display:inline-flex;align-items:center;
             justify-content:center;font-size:28px;">💬</div>
        <div style="color:#25D366;font-size:18px;font-weight:700;margin-top:10px;">WhatsApp Analyzer</div>
        <div style="color:#8b949e;font-size:12px;">ML-Powered Chat Insights</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    uploaded = st.file_uploader("📂 Upload WhatsApp export (.txt)", type=['txt'],
        help="WhatsApp → 3 dots → More → Export Chat → Without Media")

    if uploaded:
        raw = uploaded.read().decode('utf-8', errors='ignore')
        df  = preprocessor.preprocess(raw)
        if df.empty:
            st.error("❌ Could not parse this file.")
            st.stop()

        real_users = sorted(df[df['user'] != 'group_notification']['user'].unique())
        user_list  = ['Overall'] + real_users
        selected   = st.selectbox("👤 Analyse for", user_list)
        chat_type  = "Group Chat 👥" if len(real_users) > 2 else "Individual Chat 💑"

        st.markdown(f"""
        <div class="sidebar-badge">
            <b>{chat_type}</b><br>
            👤 {len(real_users)} participants<br>
            💬 {len(df):,} messages<br>
            📅 {df['date'].min()} → {df['date'].max()}
        </div>""", unsafe_allow_html=True)

        analyze_btn = st.button("🔍 Generate Analysis", type="primary")
        st.markdown("---")
        st.markdown("""
        <div style="color:#8b949e;font-size:11px;text-align:center;">
            🔒 Your data stays on your device.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="color:#8b949e;font-size:13px;line-height:1.8;">
        <b style="color:#25D366">Android:</b><br>
        Open chat → ⋮ → More → Export chat → Without Media<br><br>
        <b style="color:#25D366">iPhone:</b><br>
        Open chat → Contact name → Export Chat → Without Media
        </div>""", unsafe_allow_html=True)
        st.stop()

# ── WELCOME SCREEN ────────────────────────────────────────────────────────────
if not analyze_btn:
    st.markdown("""
    <div class="hero-banner">
        <div style="font-size:4rem">💬</div>
        <h1>WhatsApp Chat Analyzer</h1>
        <p>Upload your chat export and discover powerful ML-driven insights</p>
    </div>""", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col,(icon,title,desc) in zip([c1,c2,c3,c4],[
        ("📊","Statistics","Messages, words, media & links"),
        ("📅","Timelines","Monthly & daily activity trends"),
        ("🤖","ML Predictor","Predicts who messages next"),
        ("😊","Sentiment","Positive, neutral & negative vibes"),
    ]):
        col.markdown(f"""
        <div class="stat-card">
            <div class="icon">{icon}</div>
            <div style="font-weight:600;color:#fff;margin-bottom:4px">{title}</div>
            <div style="font-size:12px;color:#8b949e">{desc}</div>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ── MAIN HEADER ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;gap:16px;margin-bottom:8px;">
    <div style="font-size:2.5rem">💬</div>
    <div>
        <div style="font-size:1.8rem;font-weight:700;color:#ffffff;">
            Chat Analysis <span style="color:#25D366"> — {selected}</span>
        </div>
        <div style="color:#8b949e;font-size:13px;">
            {len(df):,} messages · {df['date'].min()} to {df['date'].max()}
        </div>
    </div>
</div>
<div style="height:2px;background:linear-gradient(90deg,#25D366,transparent);border-radius:2px;margin-bottom:24px;"></div>
""", unsafe_allow_html=True)

tabs = st.tabs(["📊 Overview","📅 Timelines","👥 Users","☁️ Words",
                "😊 Emojis","🧠 Sentiment","🤖 ML Predictor","🔗 Links & More"])

# ═══ TAB 1: OVERVIEW ══════════════════════════════════════════════════════════
with tabs[0]:
    try:
        msgs, words, media, links = helper.fetch_stats(selected, df)
        c1,c2,c3,c4 = st.columns(4)
        for col,val,label,icon in zip([c1,c2,c3,c4],
            [msgs,words,media,links],
            ["Total Messages","Total Words","Media Shared","Links Shared"],
            ["💬","📝","🖼️","🔗"]):
            col.markdown(f"""
            <div class="stat-card">
                <div class="icon">{icon}</div>
                <div class="value">{val:,}</div>
                <div class="label">{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<p class="section-title">⏰ Messages by Hour of Day</p>', unsafe_allow_html=True)
        hourly = helper.hourly_pattern(selected, df)
        fig = px.bar(hourly, x='hour', y='messages',
                     labels={'hour':'Hour (24h)','messages':'Messages'},
                     color='messages', color_continuous_scale=GREEN_SCALE)
        fig.update_layout(**dk(300, showlegend=False))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<p class="section-title">📅 Messages by Day of Week</p>', unsafe_allow_html=True)
        week = helper.week_activity(selected, df)
        fig2 = px.bar(x=week.index, y=week.values,
                      labels={'x':'Day','y':'Messages'},
                      color=week.values, color_continuous_scale=PURPLE_SCALE)
        fig2.update_layout(**dk(300, showlegend=False))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<p class="section-title">🔥 Activity Heatmap (Day × Hour)</p>', unsafe_allow_html=True)
        hmap = helper.activity_heatmap(selected, df)
        fig3, ax = plt.subplots(figsize=(14,4), facecolor='#161b22')
        ax.set_facecolor('#161b22')
        sns.heatmap(hmap, ax=ax, cmap='YlOrRd', linewidths=0.3, linecolor='#21262d',
                    cbar_kws={'label':'Messages'})
        ax.set_xlabel('Hour of Day', color='#8b949e')
        ax.set_ylabel('', color='#8b949e')
        ax.tick_params(colors='#8b949e')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        # Media breakdown
        st.markdown('<p class="section-title">🖼️ Media Type Breakdown</p>', unsafe_allow_html=True)
        media_df = helper.media_breakdown(selected, df)
        total_media = int(media_df['count'].sum()) if not media_df.empty else 0
        if media_df.empty or total_media == 0:
            st.info("No media messages detected in this chat.")
        else:
            mc1, mc2 = st.columns(2)
            with mc1:
                icons = {'Image':'🖼️','Video':'🎥','Audio':'🎵','Sticker':'🎭','Document':'📄','Other Media':'📎'}
                media_df['label'] = media_df['media_type'].apply(lambda x: f"{icons.get(x,'📎')} {x}")
                fig_m = px.pie(media_df, names='label', values='count',
                               color_discrete_sequence=px.colors.qualitative.Set2,
                               title=f"Total: {total_media} media")
                fig_m.update_traces(textposition='inside', textinfo='percent+label')
                fig_m.update_layout(**dk(360, showlegend=False))
                st.plotly_chart(fig_m, use_container_width=True)
            with mc2:
                fig_m2 = px.bar(media_df, x='media_type', y='count',
                                labels={'media_type':'Type','count':'Count'},
                                color='count', color_continuous_scale=TEAL_SCALE, text='count')
                fig_m2.update_traces(textposition='outside')
                fig_m2.update_layout(**dk(360, showlegend=False))
                st.plotly_chart(fig_m2, use_container_width=True)

            if selected == 'Overall':
                st.markdown('<p class="section-title">📤 Media Sent by User</p>', unsafe_allow_html=True)
                mbu = helper.media_by_user(df)
                if not mbu.empty:
                    fig_m3 = px.bar(mbu, x='user', y='media_count',
                                    labels={'user':'User','media_count':'Media sent'},
                                    color='media_count', color_continuous_scale=ORANGE_SCALE, text='media_count')
                    fig_m3.update_traces(textposition='outside')
                    fig_m3.update_layout(**dk(340, showlegend=False))
                    st.plotly_chart(fig_m3, use_container_width=True)

            st.markdown("""
| Type | What it is |
|---|---|
| 🖼️ Image | Photos shared |
| 🎥 Video | Video clips |
| 🎵 Audio | Voice notes & music |
| 🎭 Sticker | WhatsApp stickers |
| 📄 Document | PDFs, Word, Excel files |
            """)
    except Exception as e:
        st.error(f"Overview error: {e}")

# ═══ TAB 2: TIMELINES ═════════════════════════════════════════════════════════
with tabs[1]:
    try:
        st.markdown('<p class="section-title">📈 Monthly Timeline</p>', unsafe_allow_html=True)
        mt = helper.monthly_timeline(selected, df)
        fig = px.line(mt, x='period', y='messages', markers=True,
                      labels={'period':'Month','messages':'Messages'})
        fig.update_traces(line_color='#25D366', marker_color='#128C7E', marker_size=7)
        fig.update_layout(**dk(360, xaxis_tickangle=-45))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<p class="section-title">📉 Daily Timeline</p>', unsafe_allow_html=True)
        dt = helper.daily_timeline(selected, df)
        fig2 = px.area(dt, x='date', y='messages',
                       labels={'date':'Date','messages':'Messages'},
                       color_discrete_sequence=['#25D366'])
        fig2.update_layout(**dk(320))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<p class="section-title">📆 Messages per Month</p>', unsafe_allow_html=True)
        ma = helper.month_activity(selected, df)
        fig3 = px.bar(x=ma.index, y=ma.values,
                      labels={'x':'Month','y':'Messages'},
                      color=ma.values, color_continuous_scale=TEAL_SCALE)
        fig3.update_layout(**dk(300, showlegend=False))
        st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.error(f"Timelines error: {e}")

# ═══ TAB 3: USERS ═════════════════════════════════════════════════════════════
with tabs[2]:
    try:
        if selected != 'Overall':
            st.info("Switch to 'Overall' to see user-level comparisons.")
        else:
            counts, pct_df = helper.most_busy_users(df)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<p class="section-title">🏆 Most Active Users</p>', unsafe_allow_html=True)
                fig = px.bar(x=counts.values, y=counts.index, orientation='h',
                             labels={'x':'Messages','y':'User'},
                             color=counts.values, color_continuous_scale=GREEN_SCALE)
                fig.update_layout(**dk(400, showlegend=False))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown('<p class="section-title">📊 Message Share (%)</p>', unsafe_allow_html=True)
                fig2 = px.pie(pct_df, names='user', values='percent',
                              color_discrete_sequence=px.colors.qualitative.Set3)
                fig2.update_traces(textposition='inside', textinfo='percent+label')
                fig2.update_layout(**dk(400, showlegend=False))
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown('<p class="section-title">📏 Average Message Length</p>', unsafe_allow_html=True)
            ml = helper.message_length_stats('Overall', df)
            if not ml.empty:
                fig3 = px.bar(ml, x='user', y='avg_msg_length',
                              labels={'user':'User','avg_msg_length':'Avg characters'},
                              color='avg_msg_length', color_continuous_scale=BLUE_SCALE)
                fig3.update_layout(**dk(320, showlegend=False))
                st.plotly_chart(fig3, use_container_width=True)

            st.markdown('<p class="section-title">⏱️ Average Response Time (minutes)</p>', unsafe_allow_html=True)
            rt = helper.response_time(df)
            if not rt.empty:
                fig4 = px.bar(rt, x='user', y='avg_reply_min',
                              labels={'user':'User','avg_reply_min':'Avg reply (min)'},
                              color='avg_reply_min', color_continuous_scale=ORANGE_SCALE)
                fig4.update_layout(**dk(300, showlegend=False))
                st.plotly_chart(fig4, use_container_width=True)

            st.markdown('<p class="section-title">🚀 Conversation Starters</p>', unsafe_allow_html=True)
            cs = helper.conversation_starters(df)
            if not cs.empty:
                fig5 = px.bar(cs, x='user', y='times_started',
                              labels={'user':'User','times_started':'Times started'},
                              color='times_started', color_continuous_scale=PURPLE_SCALE)
                fig5.update_layout(**dk(300, showlegend=False))
                st.plotly_chart(fig5, use_container_width=True)
    except Exception as e:
        st.error(f"Users tab error: {e}")

# ═══ TAB 4: WORDS ═════════════════════════════════════════════════════════════
with tabs[3]:
    try:
        col1, col2 = st.columns([3,2])
        with col1:
            st.markdown('<p class="section-title">☁️ Word Cloud</p>', unsafe_allow_html=True)
            wc = helper.create_wordcloud(selected, df)
            fig, ax = plt.subplots(figsize=(10,5), facecolor='#161b22')
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout(pad=0)
            st.pyplot(fig)
            plt.close()
        with col2:
            st.markdown('<p class="section-title">🔤 Top 20 Words</p>', unsafe_allow_html=True)
            cw = helper.most_common_words(selected, df)
            if not cw.empty:
                fig2 = px.bar(cw, x='count', y='word', orientation='h',
                              labels={'count':'Count','word':'Word'},
                              color='count', color_continuous_scale=PURPLE_SCALE)
                fig2.update_layout(**dk(500, showlegend=False,
                                        yaxis={'categoryorder':'total ascending'}))
                st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.error(f"Words tab error: {e}")

# ═══ TAB 5: EMOJIS ════════════════════════════════════════════════════════════
with tabs[4]:
    try:
        emoji_df = helper.emoji_helper(selected, df)
        if emoji_df.empty:
            st.info("No emojis found in this chat.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<p class="section-title">😊 Top 20 Emojis</p>', unsafe_allow_html=True)
                fig = px.bar(emoji_df, x='count', y='emoji', orientation='h',
                             labels={'count':'Count','emoji':'Emoji'},
                             color='count', color_continuous_scale=GREEN_SCALE)
                fig.update_layout(**dk(500, showlegend=False,
                                       yaxis={'categoryorder':'total ascending'}))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown('<p class="section-title">🍩 Emoji Share</p>', unsafe_allow_html=True)
                fig2 = px.pie(emoji_df.head(10), names='emoji', values='count',
                              color_discrete_sequence=px.colors.qualitative.Pastel)
                fig2.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14)
                fig2.update_layout(**dk(500, showlegend=True))
                st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(emoji_df, use_container_width=True, height=300)
    except Exception as e:
        st.error(f"Emojis tab error: {e}")

# ═══ TAB 6: SENTIMENT ═════════════════════════════════════════════════════════
with tabs[5]:
    try:
        result = helper.sentiment_analysis(selected, df)
        if isinstance(result, pd.DataFrame) and result.empty:
            st.info("Not enough text messages for sentiment analysis.")
        else:
            overall_sent, by_user_sent = result
            colors = {'Positive':'#25D366','Neutral':'#8b949e','Negative':'#ef4444'}
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<p class="section-title">😊 Overall Sentiment</p>', unsafe_allow_html=True)
                fig = px.pie(values=overall_sent.values, names=overall_sent.index,
                             color=overall_sent.index, color_discrete_map=colors)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(**dk(360, showlegend=False))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown('<p class="section-title">📊 Breakdown</p>', unsafe_allow_html=True)
                pos = int(overall_sent.get('Positive',0))
                neu = int(overall_sent.get('Neutral',0))
                neg = int(overall_sent.get('Negative',0))
                tot = pos+neu+neg
                if tot:
                    st.metric("😊 Positive",f"{pos} ({pos*100//tot}%)")
                    st.metric("😐 Neutral", f"{neu} ({neu*100//tot}%)")
                    st.metric("😞 Negative",f"{neg} ({neg*100//tot}%)")
            if selected == 'Overall' and not by_user_sent.empty:
                st.markdown('<p class="section-title">👥 Sentiment by User</p>', unsafe_allow_html=True)
                fig3 = px.bar(by_user_sent, x='user', y='percent',
                              color='sentiment', barmode='group',
                              color_discrete_map=colors,
                              labels={'percent':'% messages','user':'User'})
                fig3.update_layout(**dk(380))
                st.plotly_chart(fig3, use_container_width=True)
    except Exception as e:
        st.error(f"Sentiment error: {e}")

# ═══ TAB 7: ML PREDICTOR ══════════════════════════════════════════════════════
with tabs[6]:
    try:
        st.markdown("## 🤖 ML Message Predictor")
        st.markdown("Uses **Naive Bayes** trained on your chat to predict who sends the next message.")

        try:
            import sklearn
        except ImportError:
            st.error("Install scikit-learn: `pip install scikit-learn`")
            st.stop()

        with st.spinner("🔄 Training model..."):
            model, le, features, accuracy, report_df = helper.build_predictor_model(df)

        if model is None:
            st.warning("Not enough data. Need 50+ messages with 2+ users.")
        else:
            c1,c2,c3 = st.columns(3)
            c1.metric("🎯 Accuracy", f"{accuracy}%")
            c2.metric("📚 Algorithm","Naive Bayes")
            c3.metric("📩 Training msgs",f"{len(df[df['user']!='group_notification']):,}")
            st.markdown("---")

            st.markdown('<p class="section-title">🔮 Live Prediction</p>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                pred_hour = st.slider("🕐 Hour of day", 0, 23, datetime.now().hour)
                pred_day  = st.selectbox("📅 Day of week",
                    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
                    index=datetime.now().weekday())
            with col2:
                pred_gap  = st.slider("⏱️ Minutes since last message", 0, 1440, 5)
                real_list = sorted(df[df['user']!='group_notification']['user'].unique())
                last_user = st.selectbox("👤 Who sent last?", real_list)

            day_map = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,
                       'Friday':4,'Saturday':5,'Sunday':6}
            pred_df = helper.predict_next_sender(
                model, le, df, hour=pred_hour,
                day_of_week=day_map[pred_day],
                gap_minutes=pred_gap, last_user=last_user)

            if not pred_df.empty:
                top_user = pred_df.iloc[0]['user']
                top_prob = pred_df.iloc[0]['probability']
                st.markdown(f"""
                <div class="predict-card">
                    <h3>🏆 Most likely: {top_user}</h3>
                    <p>Predicted with {top_prob:.1f}% confidence</p>
                </div>""", unsafe_allow_html=True)
                fig = px.bar(pred_df, x='user', y='probability',
                             labels={'user':'User','probability':'Probability (%)'},
                             color='probability', color_continuous_scale=GREEN_SCALE, text='probability')
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(**dk(380, showlegend=False, yaxis=dict(range=[0,110])))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown('<p class="section-title">📋 Model Performance</p>', unsafe_allow_html=True)
            if report_df is not None and not report_df.empty:
                st.dataframe(report_df.style.background_gradient(subset=['f1-score'], cmap='Greens'),
                             use_container_width=True, height=300)
                col1, col2 = st.columns(2)
                with col1:
                    fig2 = px.bar(x=report_df.index, y=report_df['f1-score'],
                                  labels={'x':'User','y':'F1-Score'},
                                  color=report_df['f1-score'], color_continuous_scale=TEAL_SCALE,
                                  text=report_df['f1-score'])
                    fig2.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                    fig2.update_layout(**dk(360, showlegend=False, yaxis=dict(range=[0,1.15])))
                    st.plotly_chart(fig2, use_container_width=True)
                with col2:
                    fig3 = go.Figure()
                    fig3.add_trace(go.Bar(name='Precision', x=report_df.index,
                                         y=report_df['precision'], marker_color='#25D366'))
                    fig3.add_trace(go.Bar(name='Recall', x=report_df.index,
                                         y=report_df['recall'], marker_color='#128C7E'))
                    fig3.update_layout(**dk(360, barmode='group', yaxis=dict(range=[0,1.15])))
                    st.plotly_chart(fig3, use_container_width=True)

            st.markdown("---")
            st.markdown('<p class="section-title">💡 Insights</p>', unsafe_allow_html=True)
            insights = helper.get_predictor_insights(model, le, df)
            ic1,ic2,ic3 = st.columns(3)
            if 'morning_person' in insights:
                ic1.markdown(f"""<div class="insight-box">
                    <div class="ins-icon">🌅</div>
                    <div class="ins-title">Morning Starter</div>
                    <div class="ins-value">{insights['morning_person']}</div>
                    <div class="ins-sub">Most active 6–10am</div>
                </div>""", unsafe_allow_html=True)
            if 'night_owl' in insights:
                ic2.markdown(f"""<div class="insight-box">
                    <div class="ins-icon">🦉</div>
                    <div class="ins-title">Night Owl</div>
                    <div class="ins-value">{insights['night_owl']}</div>
                    <div class="ins-sub">Most active 10pm–4am</div>
                </div>""", unsafe_allow_html=True)
            ic3.markdown(f"""<div class="insight-box">
                <div class="ins-icon">🎯</div>
                <div class="ins-title">Model Accuracy</div>
                <div class="ins-value">{accuracy}%</div>
                <div class="ins-sub">On test messages</div>
            </div>""", unsafe_allow_html=True)

            with st.expander("📖 How does this ML model work?"):
                st.markdown("""
**Algorithm:** Multinomial Naive Bayes

**Features used:**
| Feature | Description |
|---|---|
| Hour of day | What time is it? (0–23) |
| Day of week | Monday–Sunday |
| Is weekend | 1 if Sat/Sun |
| Gap since last message | Minutes since previous message |
| Previous sender | Who sent just before |
| Message length | How long the previous message was |

**Metrics:**
- **Precision** — when it predicts User X, how often is it right?
- **Recall** — of all User X messages, how many did it catch?
- **F1-Score** — balance of both (higher = better, max = 1.0)
                """)
    except Exception as e:
        st.error(f"ML Predictor error: {e}")

# ═══ TAB 8: LINKS & MORE ══════════════════════════════════════════════════════
with tabs[7]:
    try:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="section-title">🔗 Top Domains Shared</p>', unsafe_allow_html=True)
            ld = helper.link_analysis(selected, df)
            if ld.empty:
                st.info("No links found in this chat.")
            else:
                fig = px.bar(ld, x='count', y='domain', orientation='h',
                             labels={'count':'Times shared','domain':'Domain'},
                             color='count', color_continuous_scale=BLUE_SCALE)
                fig.update_layout(**dk(380, showlegend=False,
                                       yaxis={'categoryorder':'total ascending'}))
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown('<p class="section-title">📊 Message Types</p>', unsafe_allow_html=True)
            type_counts = df[df['user']!='group_notification']['message_type'].value_counts()
            fig2 = px.pie(values=type_counts.values, names=type_counts.index,
                          color_discrete_sequence=px.colors.qualitative.Set2)
            fig2.update_traces(textposition='inside', textinfo='percent+label')
            fig2.update_layout(**dk(380, showlegend=False))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<p class="section-title">📋 Raw Data Preview</p>', unsafe_allow_html=True)
        d_show = df[['date_time','user','message','message_type']].copy()
        if selected != 'Overall':
            d_show = d_show[d_show['user']==selected]
        st.dataframe(d_show.tail(200), use_container_width=True, height=320)

        st.markdown('<p class="section-title">💾 Download Data</p>', unsafe_allow_html=True)
        csv = d_show.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download as CSV", csv, "whatsapp_chat_data.csv",
                           "text/csv", use_container_width=True)
    except Exception as e:
        st.error(f"Links tab error: {e}")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#8b949e;font-size:12px;margin-top:40px;padding:20px;
     border-top:1px solid #21262d;">
    💬 WhatsApp Chat Analyzer · Built with Streamlit & scikit-learn ·
    🔒 Your data never leaves your device
</div>""", unsafe_allow_html=True)