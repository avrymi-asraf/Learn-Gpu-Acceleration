# %%
from transformers import AutoTokenizer, Gemma2Model, Gemma2ForCausalLM, Gemma2Config
import torch
#%%
config = Gemma2Config.from_pretrained("google/gemma-2-2b")
config.use_cache = False

# %%
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model = Gemma2Model.from_pretrained(
    "google/gemma-2-2b",
    config=config,
    # device_map="auto",
)
# modelForCausalLM = Gemma2ForCausalLM.from_pretrained("google/gemma-2-9b")


# %%
input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt")
# %%
outputs = model(**input_ids)
print(outputs["last_hidden_state"].shape)


# %%
