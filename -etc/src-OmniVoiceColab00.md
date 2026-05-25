# omnivoice-colab- Source Bundle
Generated on: Tue May 26 00:06:18 WIB 2026

## Directory Structure
```
./src
├── OmniVoice_Colab.ipynb
├── README.md
├── app.py
├── colab.txt
├── hf_mirror.py
├── requirements.txt
└── subtitle.py

1 directory, 7 files
```

## File Contents

### File: src/OmniVoice_Colab.ipynb
```
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "### 🏷️ **Credits & License**\n",
        "\n",
        "* 🔗 [OmniVoice GitHub Repository](https://github.com/k2-fsa/OmniVoice)\n",
        "* 🤗 [OmniVoice on Hugging Face](https://huggingface.co/k2-fsa/OmniVoice)\n",
        "* 📄 **License**: Provided under the [Apache License 2.0\n",
        "](https://github.com/k2-fsa/OmniVoice/blob/master/LICENSE)\n",
        "\n",
        "\n",
        "\n",
        "### ⚠️ **Usage Disclaimer**\n",
        "\n",
        "Use of this voice cloning model is subject to strict ethical and legal standards. By using this tool, you agree **not to** engage in any of the following prohibited activities:\n",
        "\n",
        "* **Fraud or Deception**: Using cloned voices to create misleading or fraudulent content.\n",
        "* **Impersonation**: Replicating someone’s voice without their explicit permission, especially for malicious, harmful, or deceptive purposes.\n",
        "* **Illegal Activities**: Employing the model in any manner that violates local, national, or international laws and regulations.\n",
        "* **Harmful Content Generation**: Creating offensive, defamatory, or unethical material, including content that spreads misinformation or causes harm.\n",
        "\n",
        "> ⚖️ **Legal Responsibility**\n",
        "> The developers of this tool disclaim all liability for misuse. **Users bear full responsibility** for ensuring that their usage complies with all applicable laws, regulations, and ethical guidelines.\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "qIQGM5Ol0RcO"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "cellView": "form",
        "id": "zuzodwStx020"
      },
      "outputs": [],
      "source": [
        "#@title Install OmniVoice\n",
        "%cd /content/\n",
        "!rm -rf ./omnivoice-colab\n",
        "!git clone https://github.com/NeuralFalconYT/omnivoice-colab.git\n",
        "%cd ./omnivoice-colab\n",
        "# !git clone https://github.com/k2-fsa/OmniVoice.git\n",
        "!git clone https://github.com/NeuralFalconYT/OmniVoice.git\n",
        "!pip install -r colab.txt\n",
        "# !pip install -r requirements.txt\n",
        "from IPython.display import clear_output\n",
        "clear_output()\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#@title Run Gradio APP\n",
        "%cd /content/omnivoice-colab\n",
        "!python app.py"
      ],
      "metadata": {
        "id": "KN98TKw3y0Nx"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}```

### File: src/README.md
```markdown

# Run OmniVoice On Google Colab

Run **OmniVoice** easily on Google Colab, no complex setup required.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeuralFalconYT/omnivoice-colab/blob/main/OmniVoice_Colab.ipynb)

---

## 🧠 About

This is a **Google Colab version** of the original OmniVoice model.
It allows you to quickly generate high-quality speech from text with minimal setup.

---

## 🔹 What it can do

* Convert text → speech
* Clone voices from audio
* Support 600+ languages ([language list](https://github.com/k2-fsa/OmniVoice/blob/master/docs/languages.md))
* Add emotions using tags (`[laughter]`, etc.)
* Customize voice (pitch, accent, style)
* Fast and high-quality output
* Generate subtitles (SRT) 🆕

---

## 🖼️ Gradio APP Screenshot 

![voice\_clone](https://github.com/user-attachments/assets/ffc989b4-2a5b-478b-a34f-5ae3eb2d9776)

![voice\_design](https://github.com/user-attachments/assets/29a2dd1a-5987-4cb8-be4f-681456ddd546)

---

## 🙌 Credit

Credit goes to the original project:
👉 [https://github.com/k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)

---

## ⚠️ Disclaimer

Please use this model responsibly. Do not use it for harmful, misleading, or unethical purposes.

```

### File: src/app.py
```
# %cd /content/omnivoice-colab
import os
import sys
import logging
import tempfile
from typing import Any, Dict

import gradio as gr
import numpy as np
import torch
import scipy.io.wavfile as wavfile
import re
import os
import uuid
temp_audio_dir="./Omni_Audio"
os.makedirs(temp_audio_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Setup path to import subtitle_maker from /content/omnivoice-colab/OmniVoice/
OmniVoice_path = f"{os.getcwd()}/OmniVoice/"
sys.path.append(OmniVoice_path)
from subtitle import subtitle_maker

# Attempt to import Whisper's supported language dict to filter unsupported languages
try:
    from subtitle import LANGUAGE_CODE as WHISPER_LANGUAGE_CODE
except ImportError:
    WHISPER_LANGUAGE_CODE = None

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logging.getLogger("omnivoice").setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Model Loading (Global Scope)
# ---------------------------------------------------------------------------
print("Loading model from k2-fsa/OmniVoice to cuda ...")

from hf_mirror import download_model

try:
  model = OmniVoice.from_pretrained(
      "k2-fsa/OmniVoice",
      device_map="cuda",
      dtype=torch.float16,
      load_asr=False,
  )
except Exception as e:
  omnivoice_model_path=download_model(
    "k2-fsa/OmniVoice",
    download_folder="./OmniVoice_Model",
    redownload=False,
    workers=6,
    use_snapshot=False,
  )

  model = OmniVoice.from_pretrained(
      omnivoice_model_path,
      device_map="cuda",
      dtype=torch.float16,
      load_asr=False,
  )
sampling_rate = model.sampling_rate
print("Model loaded successfully!")

# ---------------------------------------------------------------------------
# Event Tags & JS Functions
# ---------------------------------------------------------------------------
EVENT_TAGS = [
    "[laughter]", "[sigh]", "[confirmation-en]", "[question-en]", 
    "[question-ah]", "[question-oh]", "[question-ei]", "[question-yi]",
    "[surprise-ah]", "[surprise-oh]", "[surprise-wa]", "[surprise-yo]", 
    "[dissatisfaction-hnn]"
]

# JS for Voice Clone Tab Textbox
INSERT_TAG_JS_VC = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#vc_textbox textarea');
    if (!textarea) return current_text + " " + tag_val;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    let prefix = " ";
    let suffix = " ";
    if (!current_text) return tag_val;
    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";
    if (end < current_text.length && current_text[end] === ' ') suffix = "";
    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""

# JS for Voice Design Tab Textbox
INSERT_TAG_JS_VD = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#vd_textbox textarea');
    if (!textarea) return current_text + " " + tag_val;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    let prefix = " ";
    let suffix = " ";
    if (!current_text) return tag_val;
    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";
    if (end < current_text.length && current_text[end] === ' ') suffix = "";
    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""

# ---------------------------------------------------------------------------
# UI Configurations & Language Mappings
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)

_CATEGORIES = {
    "Gender": ["Male", "Female"],
    "Age": ["Child", "Teenager", "Young Adult", "Middle-aged", "Elderly"],
    "Pitch": ["Very Low Pitch", "Low Pitch", "Moderate Pitch", "High Pitch", "Very High Pitch"],
    "Style": ["Whisper"],
    "English Accent": [
        "American Accent", "Australian Accent", "British Accent", "Chinese Accent",
        "Canadian Accent", "Indian Accent", "Korean Accent", "Portuguese Accent",
        "Russian Accent", "Japanese Accent"
    ],
    "Chinese Dialect": [
        "Henan Dialect", "Shaanxi Dialect", "Sichuan Dialect", "Guizhou Dialect",
        "Yunnan Dialect", "Guilin Dialect", "Jinan Dialect", "Shijiazhuang Dialect",
        "Gansu Dialect", "Ningxia Dialect", "Qingdao Dialect", "Northeast Dialect"
    ],
}

DIALECT_MAP = {
    "Henan Dialect": "河南话", "Shaanxi Dialect": "陕西话", "Sichuan Dialect": "四川话",
    "Guizhou Dialect": "贵州话", "Yunnan Dialect": "云南话", "Guilin Dialect": "桂林话",
    "Jinan Dialect": "济南话", "Shijiazhuang Dialect": "石家庄话", "Gansu Dialect": "甘肃话",
    "Ningxia Dialect": "宁夏话", "Qingdao Dialect": "青岛话", "Northeast Dialect": "东北话",
}

_ATTR_INFO = {
    "English Accent": "Only effective for English speech.",
    "Chinese Dialect": "Only effective for Chinese speech.",
}

# ---------------------------------------------------------------------------
# Core Logic & Helpers
# ---------------------------------------------------------------------------
def _is_whisper_supported(lang):
    """Check if the selected language is supported by Whisper to save processing time."""
    if not lang or lang == "Auto":
        return True 
    
    if WHISPER_LANGUAGE_CODE is None:
        return True 
    
    supported_langs = [str(k).lower() for k in WHISPER_LANGUAGE_CODE.keys()] + \
                      [str(v).lower() for v in WHISPER_LANGUAGE_CODE.values()]
    
    lang_lower = lang.lower()
    for w_lang in supported_langs:
        if w_lang in lang_lower or lang_lower in w_lang:
            return True
            
    return False

def generate_subtitles_if_needed(wav_path, lang, want_subs):
    """Generates Subtitles only if user requested them and language is supported."""
    if not want_subs:
        return None, None, None

    if not _is_whisper_supported(lang):
        logging.warning(f"Language '{lang}' is likely unsupported by Whisper. Skipping subtitle generation.")
        return None, None, None

    try:
        whisper_lang = lang if (lang and lang != "Auto") else None
        whisper_results = subtitle_maker(wav_path, whisper_lang)
        if whisper_results and len(whisper_results) > 3:
            return whisper_results[1], whisper_results[2], whisper_results[3] 
    except Exception as e:
        logging.warning(f"Subtitle generation failed: {e}")

    return None, None, None


def tts_file_name(text, language="en"):
    global temp_audio_dir

    # --- Clean text ---
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text)  # keep only letters + spaces
    clean_text = clean_text.lower().strip().replace(" ", "_")

    if not clean_text:
        clean_text = "audio"

    # --- Truncate ---
    truncated = clean_text[:20]

    # --- Clean language ---
    lang = re.sub(r'\s+', '_', language.strip().lower()) if language else "unknown"

    # --- Random suffix ---
    rand = uuid.uuid4().hex[:8].upper()

    # --- Final filename ---
    return f"{temp_audio_dir}/{truncated}_{lang}_{rand}.wav"


def _gen_core(
    text, language, ref_audio, instruct, num_step, guidance_scale, 
    denoise, speed, duration, preprocess_prompt, postprocess_output, mode, ref_text=None
):
    """Core Text-to-Speech Generation Logic"""
    if not text or not text.strip():
        return None, "Please enter the text to synthesize."

    if mode == "clone" and ref_audio and not ref_text:
        try:
            whisper_lang = language if (language and language != "Auto") else None
            whisper_results = subtitle_maker(ref_audio, whisper_lang)
            if whisper_results and len(whisper_results) > 7:
                ref_text = whisper_results[7]
        except Exception as e:
            logging.warning(f"Fallback transcription failed: {e}")

    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )

    lang = language if (language and language != "Auto") else None
    kw: Dict[str, Any] = dict(text=text.strip(), language=lang, generation_config=gen_config)

    if speed is not None and float(speed) != 1.0:
        kw["speed"] = float(speed)
    if duration is not None and float(duration) > 0:
        kw["duration"] = float(duration)

    if mode == "clone":
        if not ref_audio:
            return None, "Please upload a reference audio."
        kw["voice_clone_prompt"] = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
    if mode == "design":
        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

    try:
        audio = model.generate(**kw)
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"

    # waveform = audio[0].squeeze(0).numpy()
    # waveform = (waveform * 32767).astype(np.int16)
    waveform = (audio[0] * 32767).astype(np.int16)
    return (sampling_rate, waveform), "Done."

# ---------------------------------------------------------------------------
# Gradio UI Construction
# ---------------------------------------------------------------------------
theme = gr.themes.Soft(font=["Inter", "Arial", "sans-serif"])
css = """
.gradio-container {max-width: 100% !important; font-size: 16px !important;}
.gradio-container h1 {font-size: 1.5em !important;}
.gradio-container .prose {font-size: 1.1em !important;}
.compact-audio audio {height: 60px !important;}
.compact-audio .waveform {min-height: 80px !important;}

/* CSS for Event Tags */
.tag-container {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    margin-top: 5px !important;
    margin-bottom: 10px !important;
    border: none !important;
    background: transparent !important;
}
.tag-btn {
    min-width: fit-content !important;
    width: auto !important;
    height: 32px !important;
    font-size: 13px !important;
    background: #eef2ff !important;
    border: 1px solid #c7d2fe !important;
    color: #3730a3 !important;
    border-radius: 6px !important;
    padding: 0 10px !important;
    margin: 0 !important;
    box-shadow: none !important;
}
.tag-btn:hover {
    background: #c7d2fe !important;
    transform: translateY(-1px);
}
"""

def _lang_dropdown(label="Language (optional)", value="Auto"):
    return gr.Dropdown(
        label=label, choices=_ALL_LANGUAGES, value=value,
        allow_custom_value=False, interactive=True,
    )

def _gen_settings():
    with gr.Accordion("Generation Settings (optional)", open=False):
        sp = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Speed", info="1.0 = normal. >1 faster, <1 slower.")
        du = gr.Number(value=None, label="Duration (seconds)", info="Set a fixed duration to override speed.")
        ns = gr.Slider(4, 64, value=32, step=1, label="Inference Steps", info="Lower = faster, higher = better quality.")
        dn = gr.Checkbox(label="Denoise", value=True)
        gs = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Guidance Scale (CFG)")
        pp = gr.Checkbox(label="Preprocess Prompt", value=True, info="Applies silence removal and trims reference audio.")
        po = gr.Checkbox(label="Postprocess Output", value=True, info="Removes long silences from generated audio.")
    return ns, gs, dn, sp, du, pp, po

with gr.Blocks(theme=theme, css=css, title="OmniVoice Demo") as demo:
    gr.HTML("""
        <div style="text-align: center; margin: 20px auto; max-width: 800px;">
            <h1 style="font-size: 2.5em; margin-bottom: 5px;">🎙️ OmniVoice Multilingual </h1>
            <p>State-of-the-art text-to-speech model for 600+ languages, supporting Voice Clone and Voice Design.</p>
        </div>
    """)

    with gr.Tabs():
        # ==============================================================
        # Voice Clone Tab
        # ==============================================================
        with gr.TabItem("Voice Clone"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Added elem_id for JS hook
                    vc_text = gr.Textbox(label="Text to Synthesize", lines=4, placeholder="Enter the text to synthesize...", elem_id="vc_textbox")
                    
                    # Tag Buttons for Voice Clone
                    with gr.Row(elem_classes=["tag-container"]):
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes=["tag-btn"])
                            btn.click(
                                fn=None,
                                inputs=[btn, vc_text],
                                outputs=vc_text,
                                js=INSERT_TAG_JS_VC
                            )

                    with gr.Row():
                      vc_lang = _lang_dropdown("Language (optional)")
                      vc_want_subs = gr.Checkbox(label="Want Subtitles ?", value=False)
                    vc_ref_audio = gr.Audio(label="Reference Audio (3–10 seconds audio)", type="filepath", elem_classes="compact-audio")
                    
                    vc_ref_text = gr.Textbox(
                        label="Reference Text", lines=2, 
                        placeholder="Auto-transcribed upon audio upload. You can manually edit it if Whisper gets it wrong."
                    )
                                        
                    vc_btn = gr.Button("Generate", variant="primary")
                    vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po = _gen_settings()
                
                with gr.Column(scale=1):
                    vc_audio = gr.Audio(label="Output Audio", type="numpy")
                    vc_status = gr.Textbox(label="Status", lines=1)
                    
                    with gr.Accordion("Download files", open=False):
                        vc_out_wav = gr.File(label="Generated Audio (WAV)")
                        vc_out_custom_srt = gr.File(label="Sentence Level SRT")
                        vc_out_word_srt = gr.File(label="Word Level SRT")
                        vc_out_shorts_srt = gr.File(label="Shorts SRT")

            def _auto_transcribe(audio_path, lang):
                if not audio_path:
                    return gr.update(value="")
                try:
                    whisper_lang = lang if lang != "Auto" else None
                    whisper_results = subtitle_maker(audio_path, whisper_lang)
                    if whisper_results and len(whisper_results) > 7:
                        return gr.update(value=whisper_results[7])
                except Exception as e:
                    logging.warning(f"Auto-transcription failed: {e}")
                return gr.update(value="")

            vc_ref_audio.change(
                fn=_auto_transcribe,
                inputs=[vc_ref_audio, vc_lang],
                outputs=[vc_ref_text]
            )

            def _clone_fn(text, lang, ref_aud, ref_text, want_subs, ns, gs, dn, sp, du, pp, po):
                res = _gen_core(text, lang, ref_aud, None, ns, gs, dn, sp, du, pp, po, mode="clone", ref_text=ref_text)
                if res[0] is None:
                    return None, res[1], None, None, None, None
                
                audio_tuple, status = res
                sr, waveform = audio_tuple
                # tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                tmp_wav=tts_file_name(text, language=lang)
                wavfile.write(tmp_wav, sr, waveform)
                
                c_srt, w_srt, s_srt = generate_subtitles_if_needed(tmp_wav, lang, want_subs)
                
                return audio_tuple, status, tmp_wav, c_srt, w_srt, s_srt

            vc_btn.click(
                _clone_fn,
                inputs=[vc_text, vc_lang, vc_ref_audio, vc_ref_text, vc_want_subs, vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po],
                outputs=[vc_audio, vc_status, vc_out_wav, vc_out_custom_srt, vc_out_word_srt, vc_out_shorts_srt],
            )

        # ==============================================================
        # Voice Design Tab
        # ==============================================================
        with gr.TabItem("Voice Design"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Added elem_id for JS hook
                    vd_text = gr.Textbox(label="Text to Synthesize", lines=4, placeholder="Enter the text to synthesize...", elem_id="vd_textbox")
                    
                    # Tag Buttons for Voice Design
                    with gr.Row(elem_classes=["tag-container"]):
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes=["tag-btn"])
                            btn.click(
                                fn=None,
                                inputs=[btn, vd_text],
                                outputs=vd_text,
                                js=INSERT_TAG_JS_VD
                            )

                    with gr.Row():
                      vd_lang = _lang_dropdown(value='Auto')
                      vd_want_subs = gr.Checkbox(label="Want Subtitles ?", value=False)
                    vd_btn = gr.Button("Generate", variant="primary")
                    with gr.Accordion("Character Voice Design", open=False):
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            default_val = "Auto"
                            if _cat == "Gender":
                                default_val = "Female"
                            elif _cat == "Age":
                                default_val = "Young Adult"
                                
                            vd_groups.append(
                                gr.Dropdown(label=_cat, choices=["Auto"] + _choices, value=default_val, info=_ATTR_INFO.get(_cat))
                            )
                        
                    vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po = _gen_settings()
                
                with gr.Column(scale=1):
                    vd_audio = gr.Audio(label="Output Audio", type="numpy")
                    vd_status = gr.Textbox(label="Status", lines=1)
                    
                    with gr.Accordion("Download files", open=False):
                        vd_out_wav = gr.File(label="Generated Audio (WAV)")
                        vd_out_custom_srt = gr.File(label="Sentence Level SRT")
                        vd_out_word_srt = gr.File(label="Word Level SRT")
                        vd_out_shorts_srt = gr.File(label="Shorts SRT")

            def _build_instruct(groups):
                selected = [g for g in groups if g and g != "Auto"]
                if not selected: return None
                return ", ".join([DIALECT_MAP.get(v, v) for v in selected])

            def _design_fn(text, lang, want_subs, ns, gs, dn, sp, du, pp, po, *groups):
                instruct = _build_instruct(groups)
                res = _gen_core(text, lang, None, instruct, ns, gs, dn, sp, du, pp, po, mode="design")
                if res[0] is None:
                    return None, res[1], None, None, None, None
                
                audio_tuple, status = res
                sr, waveform = audio_tuple
                tmp_wav=tts_file_name(text, language=lang)
                # tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                wavfile.write(tmp_wav, sr, waveform)
                
                c_srt, w_srt, s_srt = generate_subtitles_if_needed(tmp_wav, lang, want_subs)
                
                return audio_tuple, status, tmp_wav, c_srt, w_srt, s_srt

            vd_btn.click(
                _design_fn,
                inputs=[vd_text, vd_lang, vd_want_subs, vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po] + vd_groups,
                outputs=[vd_audio, vd_status, vd_out_wav, vd_out_custom_srt, vd_out_word_srt, vd_out_shorts_srt],
            )

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)
```

### File: src/colab.txt
```
transformers==5.3
gradio==5.50.0
pysrt>=1.1.2
sentencex>=1.0.17
faster-whisper==1.1.1
ctranslate2==4.6.0

#ctranslate2==3.24.0
#ctranslate2==4.5.0
```

### File: src/hf_mirror.py
```
# %%writefile /content/Video-Dubbing/scripts/hf_mirror.py
import os
import time
import requests
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
  from huggingface_hub import snapshot_download
except Exception as e:
  print(e)

def download_file(url, path, redownload=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path) and not redownload:
        if os.path.getsize(path) > 0:
            return f"✔️ Skipped: {os.path.basename(path)}"

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))

            with open(path, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=os.path.basename(path),
                leave=False,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return f"⬇️ Downloaded: {os.path.basename(path)}"

    except Exception as e:
        return f"❌ Failed: {url} ({e})"


def download_model(
    repo_id,
    download_folder="./",
    redownload=False,
    workers=6,
    use_snapshot=True,
):
    start_time = time.time()

    # download_dir = os.path.abspath(
    #     f"{download_folder.rstrip('/')}/{repo_id.split('/')[-1]}"
    # )
    download_dir=download_folder
    os.makedirs(download_dir, exist_ok=True)
    download_dir = os.path.abspath(download_dir)
    print(f"📂 Download directory: {download_dir}")

    # ---------- SNAPSHOT DOWNLOAD ----------
    if use_snapshot:
        try:
            print("🚀 Trying snapshot_download...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=download_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )

            print("✅ Snapshot download successful")
            print(f"\n⏱ Total time: {time.time()-start_time:.2f} sec")
            return download_dir

        except Exception as e:
            print("⚠️ Snapshot failed → fallback to parallel download")
            print("Reason:", e)

    # ---------- FALLBACK PARALLEL DOWNLOAD ----------
    print("🚀 Starting parallel download...")

    api_url = f"https://huggingface.co/api/models/{repo_id}"
    response = requests.get(api_url)
    response.raise_for_status()

    files = [f["rfilename"] for f in response.json().get("siblings", [])]

    print(f"📦 {len(files)} files | Workers: {workers}\n")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []

        for file in files:
            url = f"https://huggingface.co/{repo_id}/resolve/main/{file}"
            path = os.path.join(download_dir, file)

            futures.append(
                executor.submit(download_file, url, path, redownload)
            )

        for future in tqdm(as_completed(futures), total=len(futures), desc="Overall"):
            print(future.result())

    print(f"\n⏱ Total time: {time.time()-start_time:.2f} sec")

    return download_dir
  
# Example usage
# pip install huggingface-hub
# from hf_mirror import download_model
# download_model(
#     "ACE-Step/Ace-Step1.5",
#     download_folder="./model",
#     redownload=True,
#     workers=6,
#     use_snapshot=True,  
# )
```

### File: src/requirements.txt
```
torch==2.8.0
torchaudio==2.8.0
torchvision==0.23.0
transformers==5.3
accelerate>=1.13.0
pydub>=0.25.1
soundfile>=0.13.1
numpy
scipy
gradio==5.50.0
#for subtitle and asr 
pysrt>=1.1.2
sentencex>=1.0.17
faster-whisper==1.1.1
ctranslate2==4.6.0
#ctranslate2==3.24.0
#ctranslate2==4.5.0
```

### File: src/subtitle.py
```


# ==============================================================================
# --- 1. IMPORTS
# ==============================================================================

import os
import re
import gc
import uuid
import math
import shutil
import string
import requests
import urllib.request
import urllib.error

import torch
import pysrt
from tqdm.auto import tqdm
from faster_whisper import WhisperModel


# ==============================================================================
# --- 2. CONSTANTS & CONFIGURATION
# ==============================================================================

# Folder paths for storing generated files and temporary audio
SUBTITLE_FOLDER = "./generated_subtitle"
TEMP_FOLDER = "./subtitle_audio"

# Mapping of language names to their ISO 639-1 codes
LANGUAGE_CODE = {
    'Akan': 'aka', 'Albanian': 'sq', 'Amharic': 'am', 'Arabic': 'ar', 'Armenian': 'hy',
    'Assamese': 'as', 'Azerbaijani': 'az', 'Basque': 'eu', 'Bashkir': 'ba', 'Bengali': 'bn',
    'Bosnian': 'bs', 'Bulgarian': 'bg', 'Burmese': 'my', 'Catalan': 'ca', 'Chinese': 'zh',
    'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da', 'Dutch': 'nl', 'English': 'en',
    'Estonian': 'et', 'Faroese': 'fo', 'Finnish': 'fi', 'French': 'fr', 'Galician': 'gl',
    'Georgian': 'ka', 'German': 'de', 'Greek': 'el', 'Gujarati': 'gu', 'Haitian Creole': 'ht',
    'Hausa': 'ha', 'Hebrew': 'he', 'Hindi': 'hi', 'Hungarian': 'hu', 'Icelandic': 'is',
    'Indonesian': 'id', 'Italian': 'it', 'Japanese': 'ja', 'Kannada': 'kn', 'Kazakh': 'kk',
    'Korean': 'ko', 'Kurdish': 'ckb', 'Kyrgyz': 'ky', 'Lao': 'lo', 'Lithuanian': 'lt',
    'Luxembourgish': 'lb', 'Macedonian': 'mk', 'Malay': 'ms', 'Malayalam': 'ml', 'Maltese': 'mt',
    'Maori': 'mi', 'Marathi': 'mr', 'Mongolian': 'mn', 'Nepali': 'ne', 'Norwegian': 'no',
    'Norwegian Nynorsk': 'nn', 'Pashto': 'ps', 'Persian': 'fa', 'Polish': 'pl', 'Portuguese': 'pt',
    'Punjabi': 'pa', 'Romanian': 'ro', 'Russian': 'ru', 'Serbian': 'sr', 'Sinhala': 'si',
    'Slovak': 'sk', 'Slovenian': 'sl', 'Somali': 'so', 'Spanish': 'es', 'Sundanese': 'su',
    'Swahili': 'sw', 'Swedish': 'sv', 'Tamil': 'ta', 'Telugu': 'te', 'Thai': 'th',
    'Turkish': 'tr', 'Ukrainian': 'uk', 'Urdu': 'ur', 'Uzbek': 'uz', 'Vietnamese': 'vi',
    'Welsh': 'cy', 'Yiddish': 'yi', 'Yoruba': 'yo', 'Zulu': 'zu'
}


# ==============================================================================
# --- 3. FILE & MODEL DOWNLOADING UTILITIES
# ==============================================================================

def download_file(url, download_file_path, redownload=False):
    """Download a single file with urllib and a tqdm progress bar."""
    base_path = os.path.dirname(download_file_path)
    os.makedirs(base_path, exist_ok=True)

    if os.path.exists(download_file_path):
        if redownload:
            os.remove(download_file_path)
            tqdm.write(f"♻️ Redownloading: {os.path.basename(download_file_path)}")
        elif os.path.getsize(download_file_path) > 0:
            tqdm.write(f"✔️ Skipped (already exists): {os.path.basename(download_file_path)}")
            return True

    try:
        request = urllib.request.urlopen(url)
        total = int(request.headers.get('Content-Length', 0))
    except urllib.error.URLError as e:
        print(f"❌ Error: Unable to open URL: {url}")
        print(f"Reason: {e.reason}")
        return False

    with tqdm(total=total, desc=os.path.basename(download_file_path), unit='B', unit_scale=True, unit_divisor=1024) as progress:
        try:
            urllib.request.urlretrieve(
                url,
                download_file_path,
                reporthook=lambda count, block_size, total_size: progress.update(block_size)
            )
        except urllib.error.URLError as e:
            print(f"❌ Error: Failed to download {url}")
            print(f"Reason: {e.reason}")
            return False

    tqdm.write(f"⬇️ Downloaded: {os.path.basename(download_file_path)}")
    return True


def download_model(repo_id, download_folder="./", redownload=False):
    """
    Downloads all files from a Hugging Face repository using the public API,
    avoiding the need for a Hugging Face token for public models.
    """
    if not download_folder.strip():
        download_folder = "."

    api_url = f"https://huggingface.co/api/models/{repo_id}"
    model_name = repo_id.split('/')[-1]
    download_dir = os.path.abspath(f"{download_folder.rstrip('/')}/{model_name}")
    os.makedirs(download_dir, exist_ok=True)

    print(f"📂 Download directory: {download_dir}")

    try:
        response = requests.get(api_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching repo info: {e}")
        return None

    data = response.json()
    files_to_download = [f["rfilename"] for f in data.get("siblings", [])]

    if not files_to_download:
        print(f"⚠️ No files found in repo '{repo_id}'.")
        return None

    print(f"📦 Found {len(files_to_download)} files in repo '{repo_id}'. Checking cache...")

    for file in tqdm(files_to_download, desc="Processing files", unit="file"):
        file_url = f"https://huggingface.co/{repo_id}/resolve/main/{file}"
        file_path = os.path.join(download_dir, file)
        download_file(file_url, file_path, redownload=redownload)

    return download_dir


# ==============================================================================
# --- 4. CORE TRANSCRIPTION & PROCESSING LOGIC
# ==============================================================================

def get_language_name(code):
    """Retrieves the full language name from its code."""
    for name, value in LANGUAGE_CODE.items():
        if value == code:
            return name
    return None

def clean_file_name(file_path):
    """Generates a clean, unique file name to avoid path issues."""
    dir_name = os.path.dirname(file_path)
    base_name, extension = os.path.splitext(os.path.basename(file_path))

    cleaned_base = re.sub(r'[^a-zA-Z\d]+', '_', base_name)
    cleaned_base = re.sub(r'_+', '_', cleaned_base).strip('_')
    random_uuid = uuid.uuid4().hex[:6]

    return os.path.join(dir_name, f"{cleaned_base}_{random_uuid}{extension}")

def format_segments(segments):
    """Formats the raw segments from Whisper into structured lists."""
    sentence_timestamp = []
    words_timestamp = []
    speech_to_text = ""

    for i in segments:
        text = i.text.strip()
        sentence_id = len(sentence_timestamp)
        sentence_timestamp.append({
            "id": sentence_id,
            "text": text,
            "start": i.start,
            "end": i.end,
            "words": []
        })
        speech_to_text += text + " "

        for word in i.words:
            word_data = {
                "word": word.word.strip(),
                "start": word.start,
                "end": word.end
            }
            sentence_timestamp[sentence_id]["words"].append(word_data)
            words_timestamp.append(word_data)

    return sentence_timestamp, words_timestamp, speech_to_text.strip()

# def get_audio_file(uploaded_file):
#     """Copies the uploaded media file to a temporary location for processing."""
#     temp_path = os.path.join(TEMP_FOLDER, os.path.basename(uploaded_file))
#     cleaned_path = clean_file_name(temp_path)
#     shutil.copy(uploaded_file, cleaned_path)
#     return cleaned_path

whisper_model=None

def load_whisper_model(model_name="deepdml/faster-whisper-large-v3-turbo-ct2"):
  global whisper_model
  if whisper_model is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"
    try:
      whisper_model = WhisperModel(
                      model_name,
                      device=device,
                      compute_type=compute_type,
                  )
    except Exception as e:
      model_dir = download_model(
                  "deepdml/faster-whisper-large-v3-turbo-ct2",
                  download_folder="./",
                  redownload=False)
      whisper_model = WhisperModel(
                  model_dir,
                  device=device,
                  compute_type=compute_type)
  return whisper_model




def whisper_subtitle(uploaded_file, source_language):
    """
    Main transcription function. Loads the model, transcribes the audio,
    and generates subtitle files.
    """

    model = load_whisper_model()

    # 2. Process audio file
    audio_file_path = uploaded_file

    # 3. Transcribe
    detected_language = source_language
    lang_code = LANGUAGE_CODE.get(source_language)

    #fallback to auto if language not found
    if source_language == "Auto" or lang_code is None:
        segments, info = model.transcribe(audio_file_path, word_timestamps=True)
        detected_lang_code = info.language
        detected_language = get_language_name(detected_lang_code)
    else:
        segments, _ = model.transcribe(
            audio_file_path,
            word_timestamps=True,
            language=lang_code
        )

    sentence_timestamps, word_timestamps, transcript_text = format_segments(segments)

    # 4. Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Prepare output file paths
    base_filename = os.path.splitext(os.path.basename(uploaded_file))[0][:30]
    srt_base = f"{SUBTITLE_FOLDER}/{base_filename}_{detected_language}.srt"
    clean_srt_path = clean_file_name(srt_base)
    txt_path = clean_srt_path.replace(".srt", ".txt")
    word_srt_path = clean_srt_path.replace(".srt", "_word_level.srt")
    custom_srt_path = clean_srt_path.replace(".srt", "_Multiline.srt")
    shorts_srt_path = clean_srt_path.replace(".srt", "_shorts.srt")

    # 6. Generate all subtitle files
    generate_srt_from_sentences(sentence_timestamps, srt_path=clean_srt_path)
    word_level_srt(word_timestamps, srt_path=word_srt_path)
    shorts_json = write_sentence_srt(
        word_timestamps, output_file=shorts_srt_path, max_lines=1,
        max_duration_s=2.0, max_chars_per_line=17
    )
    sentence_json = write_sentence_srt(
        word_timestamps, output_file=custom_srt_path, max_lines=2,
        max_duration_s=7.0, max_chars_per_line=38
    )

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(transcript_text)

    return (
        clean_srt_path, custom_srt_path, word_srt_path, shorts_srt_path,
        txt_path, transcript_text, sentence_json, shorts_json, detected_language
    )



# ==============================================================================
# --- 5. SUBTITLE GENERATION & FORMATTING
# ==============================================================================

def convert_time_to_srt_format(seconds):
    """Converts seconds to the standard SRT time format (HH:MM:SS,ms)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = round((seconds - int(seconds)) * 1000)

    if milliseconds == 1000:
        milliseconds = 0
        secs += 1
        if secs == 60:
            secs, minutes = 0, minutes + 1
            if minutes == 60:
                minutes, hours = 0, hours + 1

    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def split_line_by_char_limit(text, max_chars_per_line=38):
    """Splits a string into multiple lines based on a character limit."""
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if not current_line:
            current_line = word
        elif len(current_line + " " + word) <= max_chars_per_line:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

def merge_punctuation_glitches(subtitles):
    """Cleans up punctuation artifacts at the boundaries of subtitle entries."""
    if not subtitles:
        return []

    cleaned = [subtitles[0]]
    for i in range(1, len(subtitles)):
        prev = cleaned[-1]
        curr = subtitles[i]

        prev_text = prev["text"].rstrip()
        curr_text = curr["text"].lstrip()

        match = re.match(r'^([,.:;!?]+)(\s*)(.+)', curr_text)
        if match:
            punct, _, rest = match.groups()
            if not prev_text.endswith(tuple(punct)):
                prev["text"] = prev_text + punct
            curr_text = rest.strip()

        unwanted_chars = ['"', '“', '”', ';', ':']
        for ch in unwanted_chars:
            curr_text = curr_text.replace(ch, '')
        curr_text = curr_text.strip()

        if not curr_text or re.fullmatch(r'[.,!?]+', curr_text):
            prev["end"] = curr["end"]
            continue

        curr["text"] = curr_text
        prev["text"] = prev["text"].replace('"', '').replace('“', '').replace('”', '')
        cleaned.append(curr)

    return cleaned

import json
def write_sentence_srt(
    word_level_timestamps, output_file="subtitles_professional.srt", max_lines=2,
    max_duration_s=7.0, max_chars_per_line=38, hard_pause_threshold=0.5,
    merge_pause_threshold=0.4
):
    """Creates professional-grade SRT files and a corresponding timestamp.json file."""
    if not word_level_timestamps:
        return

    # Phase 1: Generate draft subtitles based on timing and length rules
    draft_subtitles = []
    i = 0
    while i < len(word_level_timestamps):
        start_time = word_level_timestamps[i]["start"]
        
        # We'll now store the full word objects, not just the text
        current_word_objects = []
        
        j = i
        while j < len(word_level_timestamps):
            entry = word_level_timestamps[j]
            
            # Create potential text from the word objects
            potential_words = [w["word"] for w in current_word_objects] + [entry["word"]]
            potential_text = " ".join(potential_words)

            if len(split_line_by_char_limit(potential_text, max_chars_per_line)) > max_lines: break
            if (entry["end"] - start_time) > max_duration_s and current_word_objects: break

            if j > i:
                prev_entry = word_level_timestamps[j-1]
                pause = entry["start"] - prev_entry["end"]
                if pause >= hard_pause_threshold: break
                if prev_entry["word"].endswith(('.','!','?')): break

            # Append the full word object
            current_word_objects.append(entry)
            j += 1

        if not current_word_objects:
            current_word_objects.append(word_level_timestamps[i])
            j = i + 1

        text = " ".join([w["word"] for w in current_word_objects])
        end_time = word_level_timestamps[j - 1]["end"]
        
        # Include the list of word objects in our draft subtitle
        draft_subtitles.append({
            "start": start_time,
            "end": end_time,
            "text": text,
            "words": current_word_objects
        })
        i = j

    # Phase 2: Post-process to merge single-word "orphan" subtitles
    if not draft_subtitles: return
    final_subtitles = [draft_subtitles[0]]
    for k in range(1, len(draft_subtitles)):
        prev_sub = final_subtitles[-1]
        current_sub = draft_subtitles[k]
        is_orphan = len(current_sub["text"].split()) == 1
        pause_from_prev = current_sub["start"] - prev_sub["end"]

        if is_orphan and pause_from_prev < merge_pause_threshold:
            merged_text = prev_sub["text"] + " " + current_sub["text"]
            if len(split_line_by_char_limit(merged_text, max_chars_per_line)) <= max_lines:
                prev_sub["text"] = merged_text
                prev_sub["end"] = current_sub["end"]
                
                # Merge the word-level data as well
                prev_sub["words"].extend(current_sub["words"])
                continue

        final_subtitles.append(current_sub)

    final_subtitles = merge_punctuation_glitches(final_subtitles)
    # print(final_subtitles)
    # ==============================================================================
    # NEW CODE BLOCK: Generate JSON data and write files
    # ==============================================================================
    
    # This dictionary will hold the data for our JSON file
    timestamps_data = {}
    
    # Phase 3: Write the final SRT file (and prepare JSON data)
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, sub in enumerate(final_subtitles, start=1):
            # --- SRT Writing (Unchanged) ---
            text = sub["text"].replace(" ,", ",").replace(" .", ".")
            formatted_lines = split_line_by_char_limit(text, max_chars_per_line)
            start_time_str = convert_time_to_srt_format(sub['start'])
            end_time_str = convert_time_to_srt_format(sub['end'])
            
            f.write(f"{idx}\n")
            f.write(f"{start_time_str} --> {end_time_str}\n")
            f.write("\n".join(formatted_lines) + "\n\n")
            
            # --- JSON Data Population (New) ---
            # Create the list of word dictionaries for the current subtitle
            word_data = []
            for word_obj in sub["words"]:
                word_data.append({
                    "word": word_obj["word"],
                    "start": convert_time_to_srt_format(word_obj["start"]),
                    "end": convert_time_to_srt_format(word_obj["end"])
                })
            
            # Add the complete entry to our main dictionary
            timestamps_data[str(idx)] = {
                "text": "\n".join(formatted_lines),
                "start": start_time_str,
                "end": end_time_str,
                "words": word_data
            }

    # Write the collected data to the JSON file
    json_output_file = output_file.replace(".srt",".json")
    with open(json_output_file, "w", encoding="utf-8") as f_json:
        json.dump(timestamps_data, f_json, indent=4, ensure_ascii=False)
        
    # print(f"Successfully generated SRT file: {output_file}")
    # print(f"Successfully generated JSON file: {json_output_file}")
    return json_output_file

def write_subtitles_to_file(subtitles, filename="subtitles.srt"):
    """Writes a dictionary of subtitles to a standard SRT file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for id, entry in subtitles.items():
            if entry['start'] is None or entry['end'] is None:
                print(f"Skipping subtitle ID {id} due to missing timestamps.")
                continue
            start_time = convert_time_to_srt_format(entry['start'])
            end_time = convert_time_to_srt_format(entry['end'])
            f.write(f"{id}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{entry['text']}\n\n")

def word_level_srt(words_timestamp, srt_path="word_level_subtitle.srt", shorts=False):
    """Generates an SRT file with one word per subtitle entry."""
    punctuation = re.compile(r'[.,!?;:"\–—_~^+*|]')
    with open(srt_path, 'w', encoding='utf-8') as srt_file:
        for i, word_info in enumerate(words_timestamp, start=1):
            start = convert_time_to_srt_format(word_info['start'])
            end = convert_time_to_srt_format(word_info['end'])
            word = re.sub(punctuation, '', word_info['word'])
            if word.strip().lower() == 'i': word = "I"
            if not shorts: word = word.replace("-", "")
            srt_file.write(f"{i}\n{start} --> {end}\n{word}\n\n")

def generate_srt_from_sentences(sentence_timestamp, srt_path="default_subtitle.srt"):
    """Generates a standard SRT file from sentence-level timestamps."""
    with open(srt_path, 'w', encoding='utf-8') as srt_file:
        for index, sentence in enumerate(sentence_timestamp, start=1):
            start = convert_time_to_srt_format(sentence['start'])
            end = convert_time_to_srt_format(sentence['end'])
            srt_file.write(f"{index}\n{start} --> {end}\n{sentence['text']}\n\n")




# ==============================================================================
# --- 7. MAIN ORCHESTRATOR FUNCTION
# ==============================================================================

def subtitle_maker(media_file, source_lang):
    """
    The main entry point to generate and optionally translate subtitles.

    Args:
        media_file (str): Path to the input media file.
        source_lang (str): The source language ('Automatic' for detection).
        target_lang (str): The target language for translation.

    Returns:
        A tuple containing paths to all generated files and the transcript text.
    """

    try:
        (
            default_srt, custom_srt, word_srt, shorts_srt,
            txt_path, transcript, sentence_json,word_json,detected_lang
        ) = whisper_subtitle(media_file, source_lang)
    except Exception as e:
        print(f"❌ An error occurred during transcription: {e}")
        return (None, None, None, None, None, None,None,None, f"Error: {e}")

    
    return (
        default_srt, custom_srt, word_srt,
        shorts_srt, txt_path,sentence_json,word_json, transcript,detected_lang
    )


# ==============================================================================
# --- 8. INITIALIZATION
# ==============================================================================
os.makedirs(SUBTITLE_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)


# from subtitle import subtitle_maker
# media_file = "/content/output.mp3"
# source_lang = "Auto" #"English"

# default_srt, custom_srt, word_srt,shorts_srt, txt_path,sentence_json,word_json, transcript,detected_lang= subtitle_maker(
#     media_file, source_lang
# )


# default_srt      -> Original subtitles generated directly by Whisper-Large-V3-Turbo-CT2
# custom_srt       -> Modified version of default subtitles with shorter segments 
#                      (better readability for horizontal videos, Maximum 38 characters per segment. )
# word_srt         -> Word-level timestamps (useful for creating YouTube Shorts/Reels)
# shorts_srt       -> Optimized subtitles for vertical videos (displays 3–4 words at a time , Maximum 17 characters per segment.)
# txt_path         -> Full transcript as plain text (useful for video summarization or for asking questions about the video or audio data with other LLM tools)
# sentence_json,word_json --> To Generate .ass file later
# transcript       -> Transcript text directly returned by the function, if you just need the transcript
# detected_lang    -> Detected Lang
# All functionality is contained in a single file, making it portable 
# and reusable across multiple projects for different purposes.
```

