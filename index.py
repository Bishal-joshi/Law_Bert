# -------------------------------
# TinyBERT Fill-in-the-Blank with styled output
# -------------------------------

import torch
import re
import string
import gradio as gr
from src.model import TinyBERT  # replace with your model file

# -------------------------------
# Load saved model checkpoint
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("tinybert_model.pth", map_location=device)

vocab = checkpoint["vocab"]
word2idx = checkpoint["word2idx"]
idx2word = checkpoint["idx2word"]
max_seq_len = checkpoint["max_seq_len"]
vocab_size = len(vocab)

model = TinyBERT(vocab_size, max_seq_len).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# -------------------------------
# Text preprocessing
# -------------------------------


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"'s\b", "", text)
    text = text.replace("\x0c", " ")
    text = re.sub(r"\(\s*\)", " ", text)
    text = re.sub(r"Part-\s*\w+", " ", text)
    text = re.sub(r"\b\d+\.\b", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# -------------------------------
# Prediction function
# -------------------------------


def predict_masked(sentence, mask_word):
    sentence = preprocess_text(sentence)
    tokens = ["[CLS]"] + ["[MASK]" if t ==
                          mask_word else t for t in sentence.split()] + ["[SEP]"]

    input_ids = [word2idx.get(t, word2idx["[UNK]"]) for t in tokens]
    pad_len = max_seq_len - len(input_ids)
    input_ids += [word2idx["[PAD]"]] * pad_len
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_ids)

    mask_index = (input_ids[0] == word2idx["[MASK]"]
                  ).nonzero(as_tuple=True)[0][0]

    probs = torch.softmax(logits[0, mask_index], dim=-1)
    top_ids = torch.topk(probs, 5).indices
    predicted_words = [idx2word[i.item()] for i in top_ids]

    # Generate 5 sentences with masked word replaced
    sentences = []
    for word in predicted_words:
        new_tokens = [word if t == "[MASK]" else t for t in tokens]
        sentences.append(" ".join(new_tokens))

    return "\n".join(sentences)


# -------------------------------
# Gradio interface with nicer design
# -------------------------------
with gr.Blocks(css="""
    .output-box textarea {
        font-size: 16px;
        line-height: 1.5;
        min-height: 200px;
        width: 100%;
        resize: vertical;
    }
""") as demo:
    gr.Markdown(
        "<h2 style='text-align:center;color:#4B0082;'>TinyBERT Fill-in-the-Blank</h2>")
    gr.Markdown(
        "Type a sentence and select a word to mask. TinyBERT will predict 5 sentences with the masked word filled.")

    with gr.Row():
        sentence_input = gr.Textbox(
            label="Input Sentence", placeholder="Type your sentence here...", lines=2)
        mask_input = gr.Textbox(
            label="Word to Mask", placeholder="Type the word you want to mask...", lines=1)

    output_box = gr.Textbox(label="Predicted Sentences",
                            lines=10, elem_classes="output-box")

    predict_btn = gr.Button("Predict")
    predict_btn.click(fn=predict_masked, inputs=[
                      sentence_input, mask_input], outputs=output_box)

demo.launch()
