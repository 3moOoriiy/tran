import os
import tempfile
import streamlit as st
import whisper
import yt_dlp
from urllib.parse import urlparse, parse_qs
import re
import time
from datetime import datetime
import logging
import subprocess
import sys

# إعداد الـ logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# إعداد الصفحة
st.set_page_config(
    page_title="Video Transcriber (Local Whisper)",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🎥 مفرغ الفيديوهات بـ Whisper المحلي 🚀")
st.markdown("أدخل رابط فيديو YouTube لتفريغه نصياً بدون الحاجة لـ OpenAI API")

# اختيار مدة كل جزء بالدقائق
chunk_minutes = st.number_input(
    "⏳ مدة كل جزء بالدقائق", min_value=1, max_value=60, value=5,
    help="يتم تجميع المقاطع الزمنية ضمن هذه المدة لكل جزء"
)

# -------------------- دوال مساعدة -------------------- #
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
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("ffmpeg")
    return missing


def display_installation_guide(missing_deps):
    st.error("❌ مكتبات مطلوبة غير مثبتة:")
    for dep in missing_deps:
        st.code(f"pip install {dep}")
    if "ffmpeg" in missing_deps:
        st.markdown("""
        **لتثبيت FFmpeg:**
        - **Windows:** قم بتحميله من [ffmpeg.org](https://ffmpeg.org/download.html)
        - **macOS:** `brew install ffmpeg`
        - **Linux:** `sudo apt install ffmpeg` أو `sudo yum install ffmpeg`
        """)
    st.stop()


def sanitize_youtube_url(url: str) -> str:
    url = url.strip()
    if "youtu.be" in url:
        video_id = url.split("/")[-1].split("?")[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    shorts_match = re.search(r'/shorts/([\w-]+)', url)
    if shorts_match:
        video_id = shorts_match.group(1)
        return f"https://www.youtube.com/watch?v={video_id}"
    parsed = urlparse(url)
    if parsed.netloc in ["www.youtube.com", "youtube.com", "m.youtube.com"]:
        qs = parse_qs(parsed.query)
        if "v" in qs and qs["v"]:
            return f"https://www.youtube.com/watch?v={qs['v'][0]}"
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
                'title': info.get('title', 'غير متوفر'),
                'duration': info.get('duration', 0),
                'uploader': info.get('uploader', 'غير متوفر'),
                'view_count': info.get('view_count', 0),
                'thumbnail': info.get('thumbnail', '')
            }
    except Exception as e:
        logger.error(f"خطأ في استخراج معلومات الفيديو: {e}")
        return None


def download_audio(url: str, output_path: str, progress_callback=None) -> str:
    class Hook:
        def __init__(self, cb): self.cb = cb
        def __call__(self, d):
            if d['status']=='downloading' and self.cb and 'total_bytes' in d:
                pct = d['downloaded_bytes']/d['total_bytes']*100
                self.cb(pct)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path.rsplit('.',1)[0]+'.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{ 'key':'FFmpegExtractAudio','preferredcodec':'wav','preferredquality':'192'}]
    }
    if progress_callback: ydl_opts['progress_hooks']=[Hook(progress_callback)]
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([url])
        base = output_path.rsplit('.',1)[0]
        for ext in ['.wav','.m4a','.webm','.mp3']:
            f = base+ext
            if os.path.exists(f) and os.path.getsize(f)>0:
                if f!=output_path:
                    os.replace(f, output_path)
                return output_path
    except Exception as e:
        logger.error(f"خطأ في تحميل الصوت: {e}")
    return None

@st.cache_resource
def load_model(size: str):
    try:
        return whisper.load_model(size)
    except Exception as e:
        logger.error(f"خطأ في تحميل النموذج: {e}")
        return None


def transcribe_audio(model, audio_path: str, language: str="auto") -> dict:
    opts = {'task':'transcribe','fp16':False,'temperature':0.0,'compression_ratio_threshold':2.4,'no_speech_threshold':0.6}
    if language!='auto': opts['language']=language
    return model.transcribe(audio_path, **opts)

# تحقق من الاعتمادات
missing = check_dependencies()
if missing: display_installation_guide(missing)

# مدخل URL ونموذج واللغة
col1,col2,col3 = st.columns([2,1,1])
with col1:
    video_url = st.text_input("🔗 رابط الفيديو", placeholder="https://youtu.be/...", help="يدعم YouTube و Shorts و youtu.be")
with col2:
    model_size = st.selectbox("🤖 حجم النموذج", ["tiny","base","small","medium","large"], index=2)
with col3:
    language = st.selectbox("🌍 اللغة", ["auto","ar","en","fr","es","de","it","pt","ru","ja","ko","zh"], index=0)

# عرض معلومات عند صحة URL
if video_url and validate_youtube_url(video_url):
    clean = sanitize_youtube_url(video_url)
    info = get_video_info(clean)
    if info:
        dmin, dsec = divmod(info['duration'],60)
        st.success(f"📹 {info['title']}")
        st.info(f"⏱️ {dmin:02d}:{dsec:02d} — 👤 {info['uploader']} — 👁️ {info['view_count']}")
        if info['thumbnail']: st.image(info['thumbnail'], width=200)

# زر التفريغ
if st.button("🚀 بدء التفريغ النصي", use_container_width=True):
    if not video_url.strip(): st.warning("⚠️ ادخل رابط فيديو"); st.stop()
    if not validate_youtube_url(video_url): st.error("❌ رابط غير صالح"); st.stop()
    clean = sanitize_youtube_url(video_url)
    # تحميل الصوت
    with st.spinner("⏳ تحميل الصوت..."):
        progress = st.progress(0)
        status = st.empty()
        def upd(p): progress.progress(min(p,100)/100); status.text(f"{p:.1f}%")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); tmp.close()
        audio = download_audio(clean, tmp.name, upd)
        if not audio: st.error("❌ فشل التحميل"); st.stop()
        st.success("✅ تم التحميل")
    # تحميل النموذج
    with st.spinner(f"🤖 تحميل نموذج {model_size}..."):
        model = load_model(model_size)
        if not model: st.error("❌ فشل التحميل"); st.stop()
        st.success("✅ جاهز")
    # التفريغ
    with st.spinner("📝 التفريغ..."):
        start = time.time()
        res = transcribe_audio(model, audio, language)
        if not res or not res.get("segments"): st.warning("⚠️ لا نص"); st.stop()
        elapsed = time.time()-start
        st.success(f"✅ تم ({elapsed:.1f}s)")
    # تقسيم وعرض
    secs = chunk_minutes*60
    segs = res['segments']
    chunks=[]; cur=[]; bound=secs
    for s in segs:
        if s['start']>=bound:
            chunks.append(" ".join([x['text'].strip() for x in cur])); cur=[]; bound+=secs
        cur.append(s)
    if cur: chunks.append(" ".join([x['text'].strip() for x in cur]))
    for i, txt in enumerate(chunks,1):
        with st.expander(f"جزء {i}"): 
            st.text_area("", txt, height=200)
            fname=f"part{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            st.download_button(f"💾 تحميل جزء {i}", txt, file_name=fname)
    # تنظيف
    try: os.unlink(audio)
    except: pass
