import logging
import os
import time 

import torch 
import nemo.collections.asr as nemo_asr

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from .audio import prepare_audio
from .text import clean_repeated_words

logger = logging.getLogger(__name__)


def load_model(FILE_ID, MODEL_PATH) -> nemo_asr.models.EncDecRNNTBPEModel:
    """
    Download a NeMo RNNT model.
    """

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    _remove_broken_file(MODEL_PATH)


    if not os.path.exists(MODEL_PATH):
        _download_from_gdrive(FILE_ID, MODEL_PATH)
    
    logger.info("Model size %s (%.1f MB)", os.path.getsize(MODEL_PATH) / 1e6)
    return nemo_asr.models.EncDecRNNTBPEModel.restore_from(MODEL_PATH)



def transcribe(audio_input, model: nemo_asr.models.EncDecRNNTBPEModel) -> tuple[str, float]:
    """
    Prepare audio, run inference, and return cleaned transcript + duration.
    """

    tmp_wav = prepare_audio(audio_input)
    try: 
        raw_text, duration = _run_inference(model, tmp_wav)
        return clean_repeated_words(raw_text), duration

    finally: 
        _safe_remove(tmp_wav)


    

# ------------------------------------------
#  Private helpers
# ------------------------------------------

def _run_inference(model, wav_path: str) -> tuple[str, float]:
    with torch.no_grad():
        start = time.time()
        preds = model.transcribe([wav_path], batch_size=1, num_workers=0, return_hypotheses=True)
        duration = time.time() - start
    text = preds[0][0].text if preds else ""
    return text, duration



def _safe_remove(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass



def _remove_broken_file(model_path: str)-> None: 
    """
        Remove broken files
    """
    if os.path.exists(model_path) and os.path.getsize(model_path) == 0:
        os.remove(model_path)



def _download_from_gdrive(file_id: str, dest_path: str) -> None:
    """
        Download a file from Google Drive using service account credentials.
    """
    import streamlit as st

    creds = service_account.Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )

    service = build("drive", "v3", credentials=creds)
    request = service.files().get_media(fileId=file_id)
    
    with open(dest_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
