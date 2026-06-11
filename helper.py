import re
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import emoji
import numpy as np


STOP_WORDS = {
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'is','it','this','that','was','are','be','as','i','you','he','she','we',
    'they','my','your','his','her','our','their','me','him','us','them',
    'so','if','do','did','not','no','yes','ok','okay','hi','hey','hello',
    'have','had','will','just','from','by','about','up','out','what','how',
    'when','where','who','which','all','been','has','its','than','then',
    'too','very','can','get','got','lol','haha','na','ha','ya','yea','yeah',
    'like','know','think','see','good','great','nice','well','also','now',
    'media','omitted','message','deleted','https','http','www'
}

URL_RE = re.compile(r'https?://\S+')


# ── basic stats ───────────────────────────────────────────────────────────────
def fetch_stats(user, df):
    d = df if user == 'Overall' else df[df['user'] == user]
    msgs  = len(d)
    words = d['message'].apply(lambda x: len(x.split())).sum()
    media = (d['message_type'] == 'media').sum()
    links = d['message'].apply(lambda x: len(URL_RE.findall(x))).sum()
    return int(msgs), int(words), int(media), int(links)


# ── timelines ─────────────────────────────────────────────────────────────────
def monthly_timeline(user, df):
    d = df if user == 'Overall' else df[df['user'] == user]
    t = (d.groupby(['year','month_num','month']).size()
          .reset_index(name='messages')
          .sort_values(['year','month_num']))
    t['period'] = t['month'] + ' ' + t['year'].astype(str)
    return t

def daily_timeline(user, df):
    d = df if user == 'Overall' else df[df['user'] == user]
    return d.groupby('date').size().reset_index(name='messages')


# ── activity maps ─────────────────────────────────────────────────────────────
_DAY_ORDER = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

def week_activity(user, df):
    d = df if user == 'Overall' else df[df['user'] == user]
    return d['day_name'].value_counts().reindex(_DAY_ORDER).fillna(0)

def month_activity(user, df):
    d = df if user == 'Overall' else df[df['user'] == user]
    return d['month'].value_counts()

def activity_heatmap(user, df):
    d = df if user == 'Overall' else df[df['user'] == user]
    pivot = d.pivot_table(index='day_name', columns='hour',
                          values='message', aggfunc='count').reindex(_DAY_ORDER)
    return pivot.fillna(0)


# ── most busy users ───────────────────────────────────────────────────────────
def most_busy_users(df):
    real = df[df['user'] != 'group_notification']
    counts = real['user'].value_counts().head(10)
    pct = (counts / counts.sum() * 100).round(1).reset_index()
    pct.columns = ['user','percent']
    return counts, pct


# ── word cloud ────────────────────────────────────────────────────────────────
def create_wordcloud(user, df):
    d = df if user == 'Overall' else df[df['user'] == user]
    text = ' '.join(
        w for msg in d[d['message_type'] == 'text']['message']
        for w in msg.lower().split()
        if w not in STOP_WORDS and len(w) > 2 and w.isalpha()
    )
    if not text.strip():
        text = "no text messages found"
    wc = WordCloud(width=800, height=400, background_color='white',
                   max_words=150, colormap='viridis').generate(text)
    return wc


# ── most common words ─────────────────────────────────────────────────────────
def most_common_words(user, df, n=20):
    d = df if user == 'Overall' else df[df['user'] == user]
    words = [
        w for msg in d[d['message_type'] == 'text']['message']
        for w in msg.lower().split()
        if w not in STOP_WORDS and len(w) > 2 and w.isalpha()
    ]
    mc = Counter(words).most_common(n)
    return pd.DataFrame(mc, columns=['word','count'])


# ── emoji analysis ────────────────────────────────────────────────────────────
def emoji_helper(user, df):
    d = df if user == 'Overall' else df[df['user'] == user]
    emojis = []
    for msg in d['message']:
        emojis += [ch for ch in msg if ch in emoji.EMOJI_DATA]
    if not emojis:
        return pd.DataFrame(columns=['emoji','count'])
    counts = Counter(emojis).most_common(20)
    return pd.DataFrame(counts, columns=['emoji','count'])


# ── sentiment ─────────────────────────────────────────────────────────────────
def sentiment_analysis(user, df):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        return pd.DataFrame()
    d = df if user == 'Overall' else df[df['user'] == user]
    d = d[d['message_type'] == 'text'].copy()
    if d.empty:
        return pd.DataFrame()
    sia = SentimentIntensityAnalyzer()
    d['compound'] = d['message'].apply(lambda x: sia.polarity_scores(x)['compound'])
    d['sentiment'] = d['compound'].apply(
        lambda c: 'Positive' if c >= 0.05 else ('Negative' if c <= -0.05 else 'Neutral'))
    by_user = (d.groupby('user')['sentiment']
                .value_counts(normalize=True)
                .mul(100).round(1)
                .reset_index(name='percent'))
    overall = d['sentiment'].value_counts()
    return overall, by_user


# ── response time ─────────────────────────────────────────────────────────────
def response_time(df):
    d = df[df['user'] != 'group_notification'].copy()
    d = d.sort_values('date_time')
    d['prev_user'] = d['user'].shift(1)
    d['prev_time'] = d['date_time'].shift(1)
    d['resp_sec']  = (d['date_time'] - d['prev_time']).dt.total_seconds()
    replies = d[(d['user'] != d['prev_user']) & (d['resp_sec'] > 0) & (d['resp_sec'] < 86400)]
    avg = replies.groupby('user')['resp_sec'].mean().div(60).round(1).reset_index()
    avg.columns = ['user','avg_reply_min']
    return avg.sort_values('avg_reply_min').head(10)


# ── link domain breakdown ─────────────────────────────────────────────────────
def link_analysis(user, df):
    try:
        d = df if user == 'Overall' else df[df['user'] == user]
        # get all messages including link type
        all_msgs = d[d['message_type'].isin(['text','link'])]
        urls = [u for msg in all_msgs['message'] for u in URL_RE.findall(msg)]
        if not urls:
            return pd.DataFrame(columns=['domain','count'])
        domains = []
        for u in urls:
            m = re.search(r'https?://(?:www\.)?([^/?\s]+)', u)
            if m:
                domains.append(m.group(1))
        counts = Counter(domains).most_common(10)
        return pd.DataFrame(counts, columns=['domain','count'])
    except Exception:
        return pd.DataFrame(columns=['domain','count'])


# ── conversation starters ─────────────────────────────────────────────────────
def conversation_starters(df):
    d = df[df['user'] != 'group_notification'].copy().sort_values('date_time')
    d['gap_hours'] = (d['date_time'] - d['date_time'].shift(1)).dt.total_seconds() / 3600
    starters = d[(d['gap_hours'] > 4) | (d['gap_hours'].isna())]
    counts = starters['user'].value_counts().head(10).reset_index()
    counts.columns = ['user','times_started']
    return counts


# ── message length stats ──────────────────────────────────────────────────────
def message_length_stats(user, df):
    d = df if user == 'Overall' else df[df['user'] == user]
    d = d[d['message_type'] == 'text'].copy()
    if d.empty:
        return pd.DataFrame(columns=['user','avg_msg_length'])
    d['length'] = d['message'].apply(len)
    stats = d.groupby('user')['length'].mean().round(1).reset_index()
    stats.columns = ['user','avg_msg_length']
    return stats.sort_values('avg_msg_length', ascending=False).head(10)


# ── hourly pattern ────────────────────────────────────────────────────────────
def hourly_pattern(user, df):
    d = df if user == 'Overall' else df[df['user'] == user]
    return d.groupby('hour').size().reset_index(name='messages')


# ── media breakdown ───────────────────────────────────────────────────────────
def media_breakdown(user, df):
    d = df if user == 'Overall' else df[df['user'] == user]
    media_df = d[d['message_type'] == 'media'].copy()
    if media_df.empty:
        return pd.DataFrame(columns=['media_type','count'])
    if 'media_subtype' not in media_df.columns:
        return pd.DataFrame(columns=['media_type','count'])
    counts = media_df['media_subtype'].fillna('Other Media').value_counts().reset_index()
    counts.columns = ['media_type','count']
    return counts


def media_by_user(df):
    d = df[df['message_type'] == 'media']
    if d.empty:
        return pd.DataFrame(columns=['user','media_count'])
    counts = d['user'].value_counts().reset_index()
    counts.columns = ['user','media_count']
    return counts.head(10)


# ══════════════════════════════════════════════════════════════════════════════
# ── MESSAGE PREDICTOR (ML) ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def build_predictor_model(df):
    try:
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.preprocessing import LabelEncoder, MinMaxScaler
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score
    except ImportError:
        return None, None, None, None, None

    d = df[df['user'] != 'group_notification'].copy().sort_values('date_time')
    d = d[d['message_type'].isin(['text','link','media'])].reset_index(drop=True)

    if len(d) < 50:
        return None, None, None, None, None

    # ── feature engineering ───────────────────────────────────────────────────
    d['prev_user']   = d['user'].shift(1).fillna('unknown')
    d['gap_minutes'] = (d['date_time'] - d['date_time'].shift(1)).dt.total_seconds().div(60).fillna(0).clip(0, 1440)
    d['hour']        = d['date_time'].dt.hour
    d['day_of_week'] = d['date_time'].dt.dayofweek
    d['is_weekend']  = (d['day_of_week'] >= 5).astype(int)
    d['msg_len']     = d['message'].apply(len).clip(0, 500)

    le_prev = LabelEncoder()
    d['prev_user_enc'] = le_prev.fit_transform(d['prev_user'])

    features = ['hour','day_of_week','is_weekend','gap_minutes','prev_user_enc','msg_len']
    X = d[features].values
    y = d['user'].values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    if len(le.classes_) < 2:
        return None, None, None, None, None

    # stratify only if every class has >= 2 samples
    counts = Counter(y_enc)
    can_stratify = all(v >= 2 for v in counts.values())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42,
        stratify=y_enc if can_stratify else None
    )

    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # ── FIXED: only use labels that appear in y_test ──────────────────────────
    test_labels      = sorted(set(y_test))
    test_label_names = [le.classes_[i] for i in test_labels]

    report = classification_report(
        y_test, y_pred,
        labels=test_labels,
        target_names=test_label_names,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report).T
    # keep only per-user rows (drop accuracy / macro avg / weighted avg)
    report_df = report_df[report_df.index.isin(test_label_names)]
    report_df = report_df[['precision','recall','f1-score','support']].round(2)

    model._scaler   = scaler
    model._le_prev  = le_prev
    model._features = features

    return model, le, features, round(accuracy * 100, 1), report_df


def predict_next_sender(model, le, df, hour, day_of_week, gap_minutes, last_user):
    if model is None:
        return pd.DataFrame()
    try:
        try:
            prev_enc = model._le_prev.transform([last_user])[0]
        except ValueError:
            prev_enc = 0

        is_weekend = 1 if day_of_week >= 5 else 0
        msg_len    = 50

        X_raw    = np.array([[hour, day_of_week, is_weekend, gap_minutes, prev_enc, msg_len]])
        X_scaled = model._scaler.transform(X_raw)

        probs  = model.predict_proba(X_scaled)[0]
        result = pd.DataFrame({
            'user':        le.classes_,
            'probability': (probs * 100).round(1)
        }).sort_values('probability', ascending=False).reset_index(drop=True)
        return result
    except Exception:
        return pd.DataFrame()


def get_predictor_insights(model, le, df):
    if model is None:
        return {}
    d = df[df['user'] != 'group_notification'].copy()
    insights = {}

    hour_by_user = d.groupby(['user','hour']).size().reset_index(name='count')
    peak = hour_by_user.loc[hour_by_user.groupby('user')['count'].idxmax()]
    insights['peak_hours'] = peak[['user','hour']].reset_index(drop=True)

    morning = d[d['hour'].between(6, 10)]['user'].value_counts()
    if not morning.empty:
        insights['morning_person'] = morning.index[0]

    night = d[d['hour'].isin(list(range(22,24)) + list(range(0,4)))]['user'].value_counts()
    if not night.empty:
        insights['night_owl'] = night.index[0]

    return insights