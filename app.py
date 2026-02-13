import os
import streamlit as st

# IMPORTANT: smt.py must be in the same folder (src/) and must expose:
# build_smt_system(), translate_text(), save_model(), load_model(), top_translations()
from smt import build_smt_system, translate_text, save_model, load_model, top_translations

st.set_page_config(page_title="EN→HI SMT Demo", layout="wide")

st.title("EN→HI Statistical Machine Translation (SMT)")
st.caption("IBM Model 1 word alignment → phrase extraction → phrase table → bigram Hindi LM → noisy-channel decoding")

MODELS_DIR = "models"
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "smt_iitb_500k.pkl")


def file_exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False


def safe_load_model(path: str):
    try:
        return load_model(path)
    except Exception as e:
        st.error("Failed to load model.")
        st.exception(e)
        return None


def safe_save_model(model, path: str):
    try:
        save_model(model, path)
        st.success(f"Saved model to: {path}")
    except Exception as e:
        st.error("Failed to save model.")
        st.exception(e)


@st.cache_resource(show_spinner=False)
def cached_train(train_src, train_tgt, max_train, ibm_iters, phrase_pairs, max_phrase_len, seed):
    # verbose=False to avoid huge logs in Streamlit; Streamlit can still show outputs.
    return build_smt_system(
        train_src=train_src,
        train_tgt=train_tgt,
        max_train=max_train,
        seed=seed,
        ibm_iters=ibm_iters,
        phrase_build_pairs=phrase_pairs,
        max_phrase_len=max_phrase_len,
        verbose=False,
    )


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Corpus Paths (relative to src/)")
    train_src = st.text_input("train.en", "data/train.en")
    train_tgt = st.text_input("train.hi", "data/train.hi")

    st.header("Training")
    max_train = st.selectbox("Max training pairs", [50000, 100000, 200000, 500000], index=3)
    ibm_iters = st.slider("IBM1 EM iterations", 5, 15, 10)
    phrase_pairs = st.selectbox("Phrase extraction pairs", [30000, 100000, 200000, 300000], index=3)
    max_phrase_len = st.slider("Max phrase length", 2, 7, 5)
    seed = st.number_input("Random seed", value=42, step=1)

    st.header("Decoding")
    beam = st.slider("Beam size", 5, 50, 20)
    cand = st.slider("Candidates/phrase", 1, 20, 10)
    lam = st.slider("LM weight (λ)", 0.0, 2.0, 1.2, step=0.1)

    st.header("Model Cache")
    model_path = st.text_input("Model path", DEFAULT_MODEL_PATH)

st.write("### Step 1 — Load a saved model (fast) OR Train a new model")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("Load Model", use_container_width=True):
        if not file_exists(model_path):
            st.error(f"Model not found: {model_path}")
        else:
            model = safe_load_model(model_path)
            if model:
                st.session_state["model"] = model
                st.success("Model loaded.")

with col2:
    if st.button("Train Model", use_container_width=True):
        # Basic file checks to avoid silent crashes
        if not file_exists(train_src):
            st.error(f"Missing file: {train_src} (Run from src/, and ensure data/ exists)")
        elif not file_exists(train_tgt):
            st.error(f"Missing file: {train_tgt} (Run from src/, and ensure data/ exists)")
        else:
            with st.spinner("Training SMT system (this can take time for 500k)..."):
                try:
                    model = cached_train(train_src, train_tgt, max_train, ibm_iters, phrase_pairs, max_phrase_len, seed)
                    st.session_state["model"] = model
                    st.success("Training complete.")
                except Exception as e:
                    st.error("Training crashed. See error below.")
                    st.exception(e)

with col3:
    if st.button("Save Model", use_container_width=True):
        if "model" not in st.session_state:
            st.error("No model in memory. Load or train first.")
        else:
            safe_save_model(st.session_state["model"], model_path)

st.divider()

# ---------------- Main Panel ----------------
if "model" not in st.session_state:
    st.info("No model loaded yet. Click **Load Model** or **Train Model**.")
    st.stop()

model = st.session_state["model"]
stats = model.get("stats", {})

st.write("### System Requirements Output")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sentence pairs", stats.get("num_pairs", "—"))
c2.metric("EN vocab", stats.get("vocab_en", "—"))
c3.metric("HI vocab", stats.get("vocab_hi", "—"))
c4.metric("HI LM vocab", stats.get("lm_vocab", "—"))

st.caption(
    f"IBM iters: {stats.get('ibm_iters','—')} | "
    f"Phrase pairs: {stats.get('phrase_build_pairs','—')} | "
    f"Max phrase len: {stats.get('max_phrase_len','—')} | "
    f"Build sec: {stats.get('build_seconds','—')}"
)

st.write("### Top learned lexical translation probabilities: t(hi|en)")
t_ibm1 = model["t_ibm1"]
sample_words = ["the", "and", "of", "to", "is", "in", "a", "you"]

for w in sample_words:
    tops = top_translations(t_ibm1, w, k=10)
    if tops:
        st.write(f'**{w}** → ' + ", ".join([f"{f} ({p:.3f})" for f, p in tops[:8]]))

st.divider()

st.write("### Translate (Noisy Channel Decoding)")
en_in = st.text_area("English input", "A black box in your car?")
if st.button("Translate", type="primary"):
    try:
        out = translate_text(model, en_in, beam=beam, cand_per_phrase=cand, lambda_lm=lam)
        st.success("Translation completed.")
        st.write("**HYP:**", out["hyp"])
        st.caption(f"TM log: {out['tm_log']:.3f} | LM log: {out['lm_log']:.3f} | Total: {out['total']:.3f}")
    except Exception as e:
        st.error("Translation crashed. See error below.")
        st.exception(e)
