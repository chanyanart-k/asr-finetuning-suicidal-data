import streamlit as st
from audio_recorder_streamlit import audio_recorder
from config import get_model_config
from asr import load_model, transcribe


# ------------------------------------------
#  Initialize session state
# ------------------------------------------

for key in ["transcript_upload", "transcript_upload"]:
    if key not in st.session_state:
        st.session_state["key"] = ""



# ------------------------------------------
#  Load models (cached across reruns)
# ------------------------------------------

st.cache_resource(show_spinner="Loading model...")
def get_models():
    cfg = get_model_config()
    finetuned = load_model(cfg["FINETUNED_FILE_ID"], cfg["FINETUNED_MODEL_PATH"])
    base = load_model(cfg["BASE_FILE_ID"], cfg["BASE_MODEL_PATH"])
    return finetuned, base

finetuned_model, base_model = get_models()
st.success("Models ready ✅")


# ------------------------------------------
#  Tab Setup
# ------------------------------------------

tab_upload, tab_record, tab_bench = st.tabs(
        ["⬆️ Upload audio file", "🎤 Record from mic", "📊 Benchmark"]
    )


# ------------------------------------------
#  Upload Tab 
# ------------------------------------------

with tab_upload:
    st.title("Upload your Audio !") 
    uploaded = st.file_uploader(
        "Choose an audio file", 
        type=["wav", "mp3", "m4a", "flac", "ogg", "aac"],
        accept_multiple_files=False,
    )

    if uploaded:
        st.audio(uploaded.getvalue())

        if st.button("Transcribe uploaded file", type="primary"):
            try:
                with st.spinner("Transcribing..."):
                    text, _ = transcribe(uploaded, finetuned_model)
                st.session_state["transcript_upload"] = text
                st.success("Done!")
            except Exception as e:
                st.error(f"Transcription failed!: {e}")

    st.subheader("📝 Transcript")
    st.text_area("result", key="transcript_upload")
          

# ------------------------------------------
#  Record Tab 
# ------------------------------------------

with tab_record:
    st.title("Audio Recording!") 

    audio_bytes = audio_recorder(
        text="",
        recording_color="#de4b3a",
        neutral_color="#6aa36f",
        icon_name="microphone",
        icon_size="3x"
    )

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        if st.button("Transcribe recording", type="primary"):
            try:
                with st.spinner("Loading model & transcribing..."):
                    text, _ = transcribe(audio_bytes, finetuned_model)
                st.session_state["transcript_record"] = text
                st.success("Done!")
            except Exception as e:
                st.error(f"Transcription failed!: {e}")

    st.subheader("📝 Transcript")
    st.text_area("result", key="transcript_record")


# ------------------------------------------
#  Benchmark Tab 
# ------------------------------------------

BENCH_MODELS = [
        {"label": "scb10x/typhoon-asr-realtime", "model": base_model},
        {"label": "fine-tuned Typhoon model (OURS)", "model": finetuned_model},
    ]


with tab_bench:
    st.title("Run the same audio through multiple models.")

    bench_file = st.file_uploader(
        "Input audio for comparing results.",
        type=["wav", "mp3", "m4a", "flac", "ogg", "aac"],
        accept_multiple_files=False,
        key="bench_uploader",
    )

    if bench_file is None:
        st.info("Upload an audio file to start.")

    else:
        st.audio(bench_file.getvalue())
        if st.button("Run benchmark", type="primary", use_container_width=True):
            outputs = []
            with st.spinner("Transcribing with all models…"):
                for item in BENCH_MODELS:

                    try:
                        text, secs = transcribe(bench_file, item["model"])
                        outputs.append({
                            "label": item["label"],
                            "text": text,
                            "secs": secs, 
                            "err": None  
                        })

                        
                    except Exception as e:
                        outputs.append({
                            "label": item["label"],
                            "text": "",
                            "secs": 0, 
                            "err": e  
                        })
                        

                for result in outputs:
            
                    if result["err"]:
                        st.error(f"{result['label']}: {result['err']}")
                    else:
                        st.caption(f"⏱ {result['secs']:.2f}s")
                        st.text_area(result["label"], value=result["text"], height=160)