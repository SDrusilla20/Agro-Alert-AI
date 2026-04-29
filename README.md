# 🌾 AgroAlert AI

A premium, dashboard-style Streamlit app that turns raw IMD agromet advisories into:

- **Extracted weather entities** — `TEMP_MAX`, `TEMP_MIN`, `RAINFALL_LEVEL`, `HUMIDITY`, `WIND_SPEED` (fine-tuned NER)
- **Severity classification** — `Green / Yellow / Orange / Red` (fine-tuned DistilBERT/RoBERTa)
- **Crop-action recommendations** — rule-based engine over the extracted values
- **Indian-language briefings** — Tamil / Hindi / Telugu / Kannada via IndicTrans2
- **Downloadable PDF report**

Input methods: paste text, **upload .txt or .pdf**, or load the bundled IMD sample with one click.

A heuristic fallback ships with the app so it runs end-to-end **even before you upload your trained models**.

---

## 📂 Project structure

```
AgroAlertAI/
├── app.py                      # Streamlit entry point
├── requirements.txt
├── README.md
├── assets/                     # Optional logo, etc.
├── models/
│   ├── ner_model/              # Drop your fine-tuned HF NER weights here
│   └── classifier_model/       # Drop your fine-tuned classifier weights here
├── sample_data/
│   └── sample_advisory.txt
└── utils/
    ├── __init__.py
    ├── ner.py                  # NER (HF model + regex fallback)
    ├── classifier.py           # Severity classifier (HF model + heuristic)
    ├── rules.py                # Rule-based crop action engine
    ├── translator.py           # IndicTrans2 wrapper
    └── pdf_export.py           # ReportLab PDF generator
```

---

## ▶️ Run locally

```bash
cd AgroAlertAI
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

> First run will download IndicTrans2 weights (~1 GB) on the first translation.
> Add `IndicTransToolkit` if you want script-aware pre/post-processing:
> `pip install IndicTransToolkit`

---

## 📦 Plug in your Colab-trained models

After fine-tuning in Colab, save and download the model directories, then place them as:

### NER model (HuggingFace token classification)

```
models/ner_model/
├── config.json
├── pytorch_model.bin            # or model.safetensors
├── tokenizer.json / vocab.txt
└── special_tokens_map.json ...
```

In Colab:

```python
trainer.save_model("ner_model")
tokenizer.save_pretrained("ner_model")
!zip -r ner_model.zip ner_model
```

Download `ner_model.zip`, unzip it into `AgroAlertAI/models/ner_model/`.

### Classifier model (DistilBERT / RoBERTa sequence classification)

Same flow, into `AgroAlertAI/models/classifier_model/`.

> Make sure the `id2label` / `label2id` in `config.json` contain the strings
> `GREEN`, `YELLOW`, `ORANGE`, `RED` (case-insensitive substring match is
> performed in `utils/classifier.py`).

The sidebar will switch from `Heuristic 🔄` to `Fine-tuned ✅` automatically
once the directories are present and load successfully.

---

## ☁️ Deploy on Streamlit Cloud

1. Push the `AgroAlertAI/` folder to a **GitHub repo**.
2. Go to <https://share.streamlit.io> → **New app**.
3. Pick the repo, branch, and `app.py`.
4. Streamlit Cloud will install `requirements.txt` automatically.
5. For large model weights, prefer **Git LFS** or load them from a remote URL on
   first start (see "Handling large models" below).

---

## 🤗 Deploy on HuggingFace Spaces

1. Create a new Space → SDK = **Streamlit**.
2. Upload everything in `AgroAlertAI/` (or push via git).
3. Spaces will read `requirements.txt` and run `app.py`.
4. Heavy weights can either be:
   - Committed via Git LFS, **or**
   - Loaded from another HF model repo at runtime
     (`AutoModel.from_pretrained("your-username/your-model")`).

A minimal `README.md` header for Spaces:

```yaml
---
title: AgroAlert AI
emoji: 🌾
colorFrom: green
colorTo: indigo
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---
```

---

## 🧠 Handling large models

- All loaders use `@st.cache_resource`, so models load **once per process**.
- Translation uses lazy initialization — IndicTrans2 only downloads on the first
  visit to the **Multilingual Translation** page.
- For Spaces / Streamlit Cloud, prefer pulling weights from an HF Model repo
  (much more reliable than zipping into the repo):

  ```python
  from huggingface_hub import snapshot_download
  snapshot_download("your-username/agroalert-ner", local_dir="models/ner_model")
  ```

  Run this once at app startup, gated behind `if not os.path.isdir(...)`.

---

## 🎨 UI features

- Dark, gradient, glassmorphism dashboard — bespoke CSS in `st.markdown`
- Sidebar nav: Dashboard · Weather Extraction · Alert & Recommendations · Translation
- Live metric cards for every weather signal
- Animated alert badge (Green / Yellow / Orange / Red)
- NER-style highlighter inside the advisory text
- Class probability chart (Plotly)
- Checklist-style recommendation cards
- Explainability panel (triggers + top severity keywords)
- Copy-to-clipboard + Download PDF on the translation page

---

## 🔁 Fallback behaviour

| Component       | When weights present              | When weights missing                     |
| --------------- | --------------------------------- | ---------------------------------------- |
| NER             | HF token-classification pipeline  | Robust regex extractor over IMD wording  |
| Classifier      | HF sequence classification model  | Keyword + numeric-rule severity scorer   |
| Translator      | IndicTrans2 (`ai4bharat/...`)     | Google Translate via `deep-translator`, then English text |

This means the app **never crashes** when models are not yet uploaded — it just
shows the heuristic baseline so you can demo the full UX immediately.


---

## 🐍 Python version

- **Recommended:** Python **3.10 – 3.12** (PyTorch wheels are stable, IndicTrans2 loads cleanly).
- **Python 3.13+ / 3.14:** PyTorch may not yet have wheels for these versions, so IndicTrans2 silently fails to load. The app automatically falls back to the lightweight `deep-translator` engine (free Google public endpoint, no API key) so you still get real Tamil/Hindi/Telugu/Kannada output. The engine in use is shown as a small badge on the Translation page (`VIA INDICTRANS2` / `VIA GOOGLE TRANSLATE`).
- If you see `VIA TRANSLATION UNAVAILABLE`, expand the **Diagnostic details** panel to see what failed — most often it's a missing internet connection or a missing dependency (`pip install deep-translator`).
