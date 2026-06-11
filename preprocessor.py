import re
import pandas as pd


def detect_format(data):
    if re.search(r'^\[\d{1,2}/\d{1,2}/\d{2,4},', data, re.MULTILINE):
        return 'ios'
    return 'android'


def preprocess(data):
    fmt = detect_format(data)
    if fmt == 'ios':
        pattern = r'\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]\s'
    else:
        pattern = r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?:\s?[APap][Mm])?)\s-\s'

    parts = re.split(pattern, data)[1:]
    if len(parts) < 3:
        return pd.DataFrame()

    dates   = parts[::3]
    times   = parts[1::3]
    content = parts[2::3]

    records = []
    for d, t, body in zip(dates, times, content):
        body = body.strip()
        if ': ' in body:
            user, message = body.split(': ', 1)
        else:
            user, message = 'group_notification', body
        records.append({'date_str': d.strip(), 'time_str': t.strip(),
                        'user': user.strip(), 'message': message.strip()})

    df = pd.DataFrame(records)
    df['date_time'] = _parse_datetimes(df['date_str'], df['time_str'], fmt)
    df.dropna(subset=['date_time'], inplace=True)

    df['date']      = df['date_time'].dt.date
    df['year']      = df['date_time'].dt.year
    df['month_num'] = df['date_time'].dt.month
    df['month']     = df['date_time'].dt.strftime('%B')
    df['day']       = df['date_time'].dt.day
    df['day_name']  = df['date_time'].dt.strftime('%A')
    df['hour']      = df['date_time'].dt.hour
    df['minute']    = df['date_time'].dt.minute

    # classify message type AND media subtype
    df['message_type']  = df['message'].apply(_classify)
    df['media_subtype'] = df['message'].apply(_media_subtype)

    df.reset_index(drop=True, inplace=True)
    return df


def _parse_datetimes(date_s, time_s, fmt):
    combined = date_s + ' ' + time_s
    formats = (
        ['%d/%m/%y %I:%M:%S %p','%d/%m/%Y %I:%M:%S %p',
         '%m/%d/%y %I:%M:%S %p','%d/%m/%y %H:%M:%S','%d/%m/%Y %H:%M:%S']
        if fmt == 'ios' else
        ['%d/%m/%Y %H:%M','%d/%m/%y %H:%M','%m/%d/%Y %H:%M','%m/%d/%y %H:%M',
         '%d/%m/%Y %I:%M %p','%d/%m/%y %I:%M %p','%m/%d/%Y %I:%M %p','%m/%d/%y %I:%M %p']
    )
    for f in formats:
        try:
            parsed = pd.to_datetime(combined, format=f)
            if parsed.notna().sum() > len(combined) * 0.85:
                return parsed
        except Exception:
            continue
    return pd.to_datetime(combined, infer_datetime_format=True, errors='coerce')


# ── comprehensive media pattern ───────────────────────────────────────────────
_MEDIA_RE = re.compile(
    r'(<media omitted>|<image omitted>|<video omitted>|<audio omitted>|<sticker omitted>|'
    r'image omitted|video omitted|audio omitted|sticker omitted|gif omitted|'
    r'document omitted|contact card omitted|'
    r'\b(IMG|VID|AUD|PTT|STK)-\d{8}-WA\d+\.|'
    r'\.(jpg|jpeg|png|mp4|mp3|opus|aac|pdf|docx|xlsx|pptx|webp|gif)\s*\(file attached\))',
    re.I
)

# ── image patterns ────────────────────────────────────────────────────────────
_IMAGE_RE = re.compile(
    r'(image omitted|<image omitted>|'
    r'IMG-\d{8}-WA\d+\.|'
    r'\.(jpg|jpeg|png|webp|gif)\s*\(file attached\))',
    re.I
)

# ── video patterns ────────────────────────────────────────────────────────────
_VIDEO_RE = re.compile(
    r'(video omitted|<video omitted>|gif omitted|'
    r'VID-\d{8}-WA\d+\.|'
    r'\.(mp4|mkv|avi)\s*\(file attached\))',
    re.I
)

# ── audio patterns ────────────────────────────────────────────────────────────
_AUDIO_RE = re.compile(
    r'(audio omitted|<audio omitted>|'
    r'(PTT|AUD)-\d{8}-WA\d+\.|'
    r'\.(mp3|opus|aac|ogg|wav)\s*\(file attached\))',
    re.I
)

# ── sticker patterns ──────────────────────────────────────────────────────────
_STICKER_RE = re.compile(r'(sticker omitted|<sticker omitted>|STK-\d{8}-WA\d+\.)', re.I)

# ── document patterns ─────────────────────────────────────────────────────────
_DOC_RE = re.compile(
    r'(document omitted|contact card omitted|'
    r'\.(pdf|docx|xlsx|pptx|doc|xls|ppt|txt|csv|zip)\s*\(file attached\))',
    re.I
)

_DELETED  = re.compile(r'(This message was deleted|You deleted this message)', re.I)
_URL      = re.compile(r'https?://\S+')
_NOTIFY   = re.compile(
    r'(end-to-end encrypted|changed the subject|added|removed|left|'
    r'joined using|created group|changed their phone|pinned a message|'
    r'turned on disappearing|turned off disappearing|changed the group)', re.I
)


def _classify(msg):
    if _MEDIA_RE.search(msg):   return 'media'
    if _DELETED.search(msg):    return 'deleted'
    if _NOTIFY.search(msg):     return 'notification'
    if _URL.search(msg):        return 'link'
    return 'text'


def _media_subtype(msg):
    """Returns the specific type of media for media messages."""
    if not _MEDIA_RE.search(msg):
        return None
    if _STICKER_RE.search(msg): return 'Sticker'
    if _IMAGE_RE.search(msg):   return 'Image'
    if _VIDEO_RE.search(msg):   return 'Video'
    if _AUDIO_RE.search(msg):   return 'Audio'
    if _DOC_RE.search(msg):     return 'Document'
    return 'Other Media'