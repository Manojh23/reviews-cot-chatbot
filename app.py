# app.py
import streamlit as st
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftConfig, PeftModel

# ————— 1) Config & secrets —————  
HF_TOKEN    = st.secrets["HF_TOKEN"]
REPO_ID     = "MANOJHMANOJ/sft-product-reviews"
JSON_PATH   = "test_renamed.json"   # <-- make sure this file is in your repo root

# ————— 2) Load model & tokenizer —————  
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    # LoRA metadata → base_model
    peft_cfg = PeftConfig.from_pretrained(REPO_ID, use_auth_token=HF_TOKEN)
    # same 4‑bit bitsandbytes config you used at train time
    bnb_cfg  = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    # load base in 4‑bit
    base = AutoModelForCausalLM.from_pretrained(
        peft_cfg.base_model_name_or_path,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        use_auth_token=HF_TOKEN,
    )
    base.config.use_cache = True
    # wrap with your LoRA adapter
    model = PeftModel.from_pretrained(base, REPO_ID, torch_dtype=torch.float16, use_auth_token=HF_TOKEN)
    model.eval()
    # tokenizer
    tok = AutoTokenizer.from_pretrained(REPO_ID, trust_remote_code=True, use_auth_token=HF_TOKEN)
    tok.pad_token     = tok.eos_token
    tok.padding_side  = "right"
    return model, tok

model, tokenizer = load_model_and_tokenizer()

# ————— 3) Load your JSON test set —————  
@st.cache_data(show_spinner=False)
def load_test_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

raw_test = load_test_json(JSON_PATH)

# ————— 4) UI —————  
st.title("🗣️ Reviews & Summaries QA Bot")

# 4a) pick which example
item_ids = [ex["item_id"] for ex in raw_test]
selected = st.selectbox("Choose a product instance:", item_ids)

# 4b) look up that example
ex = next(x for x in raw_test if x["item_id"] == selected)

# 4c) display its summaries & reviews
st.markdown("**Summaries:**")
for s in ex.get("summaries", []):
    st.write(f"- {s}")

st.markdown("**Reviews:**")
for r in ex.get("reviews", []):
    st.write(f"- {r}")

# 4d) let user type a question
question = st.text_input("Your question about this product:", "")

# 4e) when they click “Ask”, generate CoT + answer
if st.button("Ask"):
    if not question:
        st.warning("Please enter a question.")
    else:
        # rebuild exactly as in training
        summ_str = "Summaries:\n" + "\n".join(f"- {s}" for s in ex["summaries"]) + "\n"
        rev_str  = "Reviews:\n"   + "\n".join(f"- {r}" for r in ex["reviews"])   + "\n"
        prompt   = (
            summ_str
            + rev_str
            + "Question:\n" + question
            + "\nAnswer reasoning based on the reviews and summaries:\n"
        )

        # tokenize & generate
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
        out    = model.generate(**inputs, max_length=512, num_beams=5, early_stopping=True)
        text   = tokenizer.decode(out[0], skip_special_tokens=True)
        cot_and_ans = text[len(prompt):].strip()

        st.subheader("🤖 Chain‑of‑Thought + Answer")
        st.write(cot_and_ans)
