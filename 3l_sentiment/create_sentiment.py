#!/usr/bin/env python3

import argparse
import json
import random
from datasets import load_dataset
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

NB_PROMPTS = [
    "Her følger dokumenter og deres sentiment, som kan være {labels_str}.",
    "Du får en tekst. Hva er følelsen i teksten? Mulige svar: {labels_str}.",
    "Klassifiser teksten under som en av følgende: {labels_str}.",
    "Vurder følgende dokument og angi sentiment ({labels_str}):",
    "Finn følelsen (kun {labels_str}) i denne teksten:",
    "Teksten under skal klassifiseres. Mulige verdier: {labels_str}.",
    "Hva er sentimentet i teksten? Svar med {labels_str}.",
    "Gi kun ett av følgende som svar: {labels_str}.",
    "Klassifiser følelsen, velg blant: {labels_str}.",
    "Dokument: {{text}}\nHva er sentimentet? ({labels_str})",
    "Følelsesanalyse: Hva er riktig kategori ({labels_str}) for teksten?",
    "Svar med én av disse: {labels_str}.",
    "Vurder tekstens sentiment. Bare {labels_str} er gyldig.",
    "Sentimentklassifisering: kun {labels_str} tillatt.",
    "Angi tekstens sentiment. Mulig: {labels_str}.",
    "Tekstanalyse: velg {labels_str} som svar.",
    "Klassifiser denne teksten, og bruk {labels_str}.",
    "Oppgi kun én kategori: {labels_str}.",
    "Hvilken av {labels_str} passer best for teksten?",
    "Vurder teksten og svar med {labels_str}.",
]

EN_PROMPTS = [
    "The following are documents and their sentiment, which can be {labels_str}.",
    "Classify the sentiment of the following document. Choices: {labels_str}.",
    "Read the text below and label as {labels_str}.",
    "Assign a sentiment to this document (choose: {labels_str}):",
    "For the text below, give the sentiment ({labels_str} only):",
    "Document sentiment. Valid answers: {labels_str}.",
    "What is the sentiment? Answer with {labels_str}.",
    "Provide only one of the following: {labels_str}.",
    "Select the sentiment, use one of: {labels_str}.",
    "Document: {{text}}\nWhat is the sentiment? ({labels_str})",
    "Sentiment analysis: choose from {labels_str}.",
    "Reply using exactly one of these: {labels_str}.",
    "Label the document's sentiment. Only {labels_str} allowed.",
    "Sentiment classification: pick {labels_str}.",
    "Assign the document to one of: {labels_str}.",
    "Text analysis: select {labels_str}.",
    "Classify this text and use {labels_str} as your answer.",
    "Return only one class: {labels_str}.",
    "Which of {labels_str} fits this text?",
    "Evaluate the sentiment, answer with {labels_str}.",
]

LABEL_MAPS = {
    "en": {1: "positive", 0: "negative"},
    "no": {1: "positiv", 0: "negativ"},
}
LABELS_STR = {
    "en": "positive, negative",
    "no": "positiv, negativ"
}
PROMPT_TEMPLATE = {
    "en": "Document: {text}\nSentiment: {label}",
    "no": "Dokument: {text}\nSentiment: {label}",
}
INSTRUCTION_TEMPLATE = {
    "en": "Document: {text}\n\nClassify the sentiment in the document. Answer with {labels_str}, and nothing else.",
    "no": "Dokument: {text}\n\nKlassifiser følelsen i teksten. Svar med {labels_str}, og ikke noe annet.",
}

def process_dataset(output_file, use_chat_template, chat_template_model):
    dataset = load_dataset('EleutherAI/twitter-sentiment', split='train')

    tokenizer = None
    if use_chat_template:
        if AutoTokenizer is None:
            raise ImportError("transformers not installed. Required for --chat_template.")
        tokenizer = AutoTokenizer.from_pretrained(chat_template_model, trust_remote_code=True)
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(f"Model '{chat_template_model}' does not support apply_chat_template.")

    with open(output_file, 'w', encoding='utf-8') as f:
        pbar = tqdm(total=len(dataset), desc="Processing dataset")
        for idx in range(len(dataset)):
            language = "no" if random.random() < 0.5 else "en"
            prompt_list = NB_PROMPTS if language == "no" else EN_PROMPTS
            labels_str = LABELS_STR[language]
            prompt_prefix = random.choice(prompt_list).format(labels_str=labels_str)
            prompt_template = PROMPT_TEMPLATE[language]
            text = dataset[idx]['text']
            label = LABEL_MAPS[language][dataset[idx]['label']]

            if use_chat_template:
                chat_message = [
                    {"role": "system", "content": prompt_prefix},
                    {"role": "user", "content": INSTRUCTION_TEMPLATE[language].format(text=text, labels_str=labels_str)},
                    {"role": "assistant", "content": label}
                ]
                formatted = tokenizer.apply_chat_template(
                    [chat_message], tokenize=False, add_generation_prompt=False
                )[0]
                record = {
                    "source": "EleutherAI/twitter-sentiment",
                    "language": language,
                    "text": formatted,
                    "prompt": prompt_prefix
                }
            else:
                record = {
                    "source": "EleutherAI/twitter-sentiment",
                    "language": language,
                    "text": prompt_prefix + "\n\n" + prompt_template.format(text=text, label=label),
                    "prompt": prompt_prefix
                }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            pbar.update(1)
        pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process EleutherAI Twitter Sentiment dataset (bilingual, prompt augmentation, one sample per record).')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSONLines file')
    parser.add_argument('--chat_template', type=str, default=None, help='HF model for chat template formatting (optional).')
    args = parser.parse_args()
    process_dataset(
        args.output_file,
        use_chat_template=bool(args.chat_template),
        chat_template_model=args.chat_template
    )
