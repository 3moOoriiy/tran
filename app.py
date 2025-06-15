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

# إضافة اختيار مدة كل جزء
chunk_minutes = st.number_input(
    "⏳ مدة كل جزء بالدقائق", min_value=1, max_value=60, value=5,
    help="يتم تجميع المقاطع الزمنية ضمن هذه المدة لكل جزء"
)

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
    import subprocess
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("ffmpeg")
    return missing

# بقية الدوال كما في الأصل...

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

# ... sanitize_youtube_url, validate_youtube_url, get_video_info, download_audio_robust,
# load_whisper_model, transcribe_audio, format_time, create_srt_content بدون تغيير

# التحقق من المتطلبات
missing_deps = check_dependencies()
if missing_deps:
    display_installation_guide(missing_deps)

# واجهة المستخدم الرئيسية
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    video_url = st.text_input(
        "🔗 رابط الفيديو", placeholder="https://www.youtube.com/watch?v=...",
        help="يدعم روابط YouTube العادية و Shorts و youtu.be"
    )

with col2:
    model_size = st.selectbox(
        "🤖 حجم النموذج",
        ["tiny", "base", "small", "medium", "large"],
        index=2,
        help="النماذج الأكبر أدق لكن أبطأ"
    )

with col3:
    language = st.selectbox(
        "🌍 اللغة",
        {"auto": "كشف تلقائي", "ar": "العربية", "en": "الإنجليزية", "fr": "الفرنسية", 
         "es": "الإسبانية", "de": "الألمانية", "it": "الإيطالية",
         "pt": "البرتغالية", "ru": "الروسية", "ja": "اليابانية",
         "ko": "الكورية", "zh": "الصينية"},
        index=1
    )

# معلومات الفيديو
if video_url and validate_youtube_url(video_url):
    clean_url = sanitize_youtube_url(video_url)
    video_info = get_video_info(clean_url)
    # ... عرض المعلومات والتنبيهات كما في الأصل

# زر بدء التفريغ
if st.button("🚀 بدء التفريغ النصي", type="primary", use_container_width=True):
    if not video_url.strip():
        st.warning("⚠️ الرجاء إدخال رابط فيديو")
    elif not validate_youtube_url(video_url.strip()):
        st.error("❌ رابط YouTube غير صالح")
    else:
        clean_url = sanitize_youtube_url(video_url.strip())
        temp_audio_path = None
        try:
            # تحميل الصوت وتحميل النموذج كما في الأصل...
            result = transcribe_audio(model, temp_audio_path, language)
            if not result or not result.get("segments"):
                st.warning("⚠️ لم يتم العثور على نص أو مقاطع زمنية")
                st.stop()
            segments = result["segments"]
            # تقسيم إلى أجزاء بناءً على chunk_minutes
            chunks = []
            current = []
            boundary = chunk_minutes * 60
            for seg in segments:
                if seg["start"] >= boundary:
                    # finalize current
                    text = " ".join([s['text'].strip() for s in current])
                    chunks.append(text)
                    current = []
                    boundary += chunk_minutes * 60
                current.append(seg)
            if current:
                chunks.append(" ".join([s['text'].strip() for s in current]))

            # عرض كل جزء في expander مع زر تحميل
            for i, chunk in enumerate(chunks):
                with st.expander(f"جزء {i+1}"):
                    st.text_area("", chunk, height=200)
                    file_name = f"transcript_part{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    st.download_button(
                        label=f"💾 تنزيل جزء {i+1}",
                        data=chunk,
                        file_name=file_name,
                        mime="text/plain"
                    )

        except Exception as e:
            st.error(f"❌ حدث خطأ: {e}")
            logger.error(f"خطأ عام: {e}", exc_info=True)
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
