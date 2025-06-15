import os
import tempfile
import streamlit as st
import whisper
import yt_dlp
import imageio_ffmpeg as iio_ffmpeg
from urllib.parse import urlparse, parse_qs
import re
import time
from datetime import datetime
import logging

# استخدم ffmpeg المضمّن عبر imageio-ffmpeg
FFMPEG_BIN = iio_ffmpeg.get_ffmpeg_exe()

# إعداد logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="Video Transcriber (Local Whisper)",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🎥 مفرغ الفيديوهات بـ Whisper المحلي 🚀")
st.markdown("أدخل رابط فيديو YouTube لتفريغه نصياً بدون الحاجة لـ OpenAI API")

# تحديد مدة كل جزء بالدقائق
chunk_minutes = st.number_input(
    "⏳ مدة كل جزء بالدقائق", min_value=1, max_value=60, value=5,
    help="يتم تجميع المقاطع الزمنية ضمن هذه المدة لكل جزء"
)

# دوال مساعدة
def check_dependencies():
    missing = []
    try:
        import whisper
    except ImportError:
        missing.append("openai-whisper")
    try:
        import yt_dlp
    except ImportError:
        missing.append("yt-dlp")
    try:
        _ = FFMPEG_BIN
    except Exception:
        missing.append("ffmpeg (imageio-ffmpeg)")
    return missing

def display_installation_guide(missing):
    st.error("❌ المكتبات التالية غير مثبتة:")
    for dep in missing:
        st.code(f"pip install {dep}")
    st.stop()

def sanitize_youtube_url(url: str) -> str:
    url = url.strip()
    if "youtu.be" in url:
        vid = url.split("/")[-1].split("?")[0]
        return f"https://www.youtube.com/watch?v={vid}"
    m = re.search(r'/shorts/([\w-]+)', url)
    if m:
        return f"https://www.youtube.com/watch?v={m.group(1)}"
    parsed = urlparse(url)
    if parsed.netloc in ["www.youtube.com","youtube.com","m.youtube.com"]:
        q = parse_qs(parsed.query)
        if "v" in q:
            return f"https://www.youtube.com/watch?v={q['v'][0]}"
    return url

def validate_youtube_url(url: str) -> bool:
    patterns = [
        r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'https?://(?:www\.)?youtube\.com/shorts/[\w-]+',
        r'https?://youtu\.be/[\w-]+',
        r'https?://m\.youtube\.com/watch\?v=[\w-]+'
    ]
    return any(re.match(p, url) for p in patterns)

@st.cache_data(ttl=300)
def get_video_info(url: str):
    try:
        opts = {'quiet': True, 'no_warnings': True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title','غير متوفر'),
                'duration': info.get('duration',0),
                'uploader': info.get('uploader','غير متوفر'),
                'view_count': info.get('view_count',0),
                'thumbnail': info.get('thumbnail','')
            }
    except Exception as e:
        logger.error(f"Error extracting video info: {e}")
        return None

def download_audio(url: str, out_path: str, progress_cb=None) -> str:
    class Hook:
        def __init__(self, cb): self.cb = cb
        def __call__(self, d):
            if d.get('status')=='downloading' and self.cb and d.get('total_bytes'):
                pct = d['downloaded_bytes'] / d['total_bytes'] * 100
                self.cb(pct)

    opts = {
        'format': 'bestaudio/best',
        'outtmpl': out_path.rsplit('.',1)[0]+'.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'ffmpeg_location': FFMPEG_BIN,
        'postprocessors': [{
            'key':'FFmpegExtractAudio',
            'preferredcodec':'wav',
            'preferredquality':'192'
        }]
    }
    if progress_cb:
        opts['progress_hooks'] = [Hook(progress_cb)]

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        base = out_path.rsplit('.',1)[0]
        for ext in ['.wav','.m4a','.webm','.mp3']:
            f = base + ext
            if os.path.exists(f) and os.path.getsize(f)>0:
                if f != out_path:
                    os.replace(f, out_path)
                return out_path
    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
    return None

@st.cache_resource
def load_model(size: str):
    try:
        return whisper.load_model(size)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def transcribe_audio(model, path: str, lang: str="auto") -> dict:
    opts = {
        'task':'transcribe','fp16':False,
        'temperature':0.0,'compression_ratio_threshold':2.4,
        'no_speech_threshold':0.6
    }
    if lang != 'auto': opts['language'] = lang
    return model.transcribe(path, **opts)

# تحقق من الاعتمادات
miss = check_dependencies()
if miss:
    display_installation_guide(miss)

# واجهة المستخدم
col1, col2, col3 = st.columns([2,1,1])
with col1:
    video_url = st.text_input("🔗 رابط الفيديو")
with col2:
    model_size = st.selectbox("🤖 حجم النموذج", ["tiny","base","small","medium","large"], index=2)
with col3:
    language = st.selectbox("🌍 اللغة", ["auto","ar","en","fr","es","de","it","pt","ru","ja","ko","zh"], index=0)

# عرض معلومات الفيديو إذا كان الرابط صحيحاً
if video_url and validate_youtube_url(video_url):
    clean = sanitize_youtube_url(video_url)
    info = get_video_info(clean)
    if info:
        dm, ds = divmod(info['duration'],60)
        st.success(f"📹 {info['title']}")
        st.info(f"⏱️ {dm:02d}:{ds:02d} — 👤 {info['uploader']} — 👁️ {info['view_count']}")
        if info['thumbnail']:
            st.image(info['thumbnail'], width=200)

# زر التفريغ
if st.button("🚀 بدء التفريغ النصي", use_container_width=True):
    if not video_url.strip():
        st.warning("⚠️ ادخل رابط فيديو")
        st.stop()
    if not validate_youtube_url(video_url):
        st.error("❌ رابط غير صالح")
        st.stop()

    clean = sanitize_youtube_url(video_url)

    # تحميل الصوت
    with st.spinner("⏳ تحميل الصوت..."):
        progress = st.progress(0)
        status = st.empty()
        def upd(p): 
            progress.progress(min(p,100)/100)
            status.text(f"{p:.1f}%")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        audio_path = download_audio(clean, tmp.name, upd)
        if not audio_path:
            st.error("❌ فشل التحميل")
            st.stop()
        st.success("✅ تم التحميل")

    # تحميل النموذج
    with st.spinner(f"🤖 تحميل نموذج {model_size}..."):
        model = load_model(model_size)
        if not model:
            st.error("❌ فشل تحميل النموذج")
            st.stop()
        st.success("✅ جاهز")

    # التفريغ
    with st.spinner("📝 التفريغ..."):
        t0 = time.time()
        res = transcribe_audio(model, audio_path, language)
        if not res.get('segments'):
            st.warning("⚠️ لم يتم العثور على نص")
            st.stop()
        dt = time.time() - t0
        st.success(f"✅ تم التفريغ ({dt:.1f}s)")

    # تقسيم النص إلى أجزاء
    sec_chunk = chunk_minutes * 60
    segs = res['segments']
    chunks, cur, boundary = [], [], sec_chunk
    for s in segs:
        if s['start'] >= boundary:
            chunks.append(" ".join(x['text'].strip() for x in cur))
            cur, boundary = [], boundary + sec_chunk
        cur.append(s)
    if cur:
        chunks.append(" ".join(x['text'].strip() for x in cur))

    # العرض والتنزيل
    for idx, text in enumerate(chunks, 1):
        with st.expander(f"جزء {idx}"):
            st.text_area("", text, height=200)
            fname = f"part{idx}_{datetime.now():%Y%m%d_%H%M%S}.txt"
            st.download_button(f"💾 تحميل جزء {idx}", text, file_name=fname)

    # تنظيف مؤقت
    try:
        os.remove(audio_path)
    except:
        pass
