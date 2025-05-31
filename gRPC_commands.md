# gRPC Command Overview for Data Processing

This document contains all gRPC commands used to process, evaluate, and augment corpora across various tasks and languages.

---

## 🇳🇴 Norwegian Evaluation & Augmentation (Parts 1–8)

Each part includes evaluation of:
- Rendered SPM (eval_spm)
- Clean text (eval_text)
- 10-alternative format (spm10)

```bash
# Part 1
python grpc_processor.py --input_file gs://synthncc/spmrendered/edu2_ling1_no_clean_spm_part1_rendered.jsonl --task eval_spm --output_file edu2_ling1_no_clean_spm_rendered_part1.jsonl --output_bucket_dir gs://synthncc/eval/ --template_file sporsmal_evaluation_template.txt && \
python grpc_processor.py --input_file gs://synthncc/text_clean/edu2_ling1_no_clean_part1.jsonl --task eval_text --output_file edu2_ling1_no_text_clean_part1.jsonl --output_bucket_dir gs://synthncc/eval/ --template_file text_evaluation_template.txt && \
python grpc_processor.py --input_file gs://synthncc/reduced/edu2_ling1_no_part1.jsonl --task spm10 --output_file edu2_ling1_no_clean_spm10_part1.jsonl --output_bucket_dir gs://synthncc/sporsmal/ --template_file sporsmal10_template.txt

# Part 2
python grpc_processor.py --input_file gs://synthncc/spmrendered/edu2_ling1_no_clean_spm_part2_rendered.jsonl --task eval_spm --output_file edu2_ling1_no_clean_spm_rendered_part2.jsonl --output_bucket_dir gs://synthncc/eval/ --template_file sporsmal_evaluation_template.txt && \
python grpc_processor.py --input_file gs://synthncc/text_clean/edu2_ling1_no_clean_part2.jsonl --task eval_text --output_file edu2_ling1_no_text_clean_part2.jsonl --output_bucket_dir gs://synthncc/eval/ --template_file text_evaluation_template.txt && \
python grpc_processor.py --input_file gs://synthncc/reduced/edu2_ling1_no_part2.jsonl --task spm10 --output_file edu2_ling1_no_clean_spm10_part2.jsonl --output_bucket_dir gs://synthncc/sporsmal/ --template_file sporsmal10_template.txt

# Part 3
python grpc_processor.py --input_file gs://synthncc/spmrendered/edu2_ling1_no_clean_spm_part3_rendered.jsonl --task eval_spm --output_file edu2_ling1_no_clean_spm_rendered_part3.jsonl --output_bucket_dir gs://synthncc/eval/ --template_file sporsmal_evaluation_template.txt && \
python grpc_processor.py --input_file gs://synthncc/text_clean/edu2_ling1_no_clean_part3.jsonl --task eval_text --output_file edu2_ling1_no_text_clean_part3.jsonl --output_bucket_dir gs://synthncc/eval/ --template_file text_evaluation_template.txt && \
python grpc_processor.py --input_file gs://synthncc/reduced/edu2_ling1_no_part3.jsonl --task spm10 --output_file edu2_ling1_no_clean_spm10_part3.jsonl --output_bucket_dir gs://synthncc/sporsmal/ --template_file sporsmal10_template.txt

# Part 4
python grpc_processor.py --input_file gs://synthncc/spmrendered/edu2_ling1_no_clean_spm_part4_rendered.jsonl --task eval_spm --output_file edu2_ling1_no_clean_spm_rendered_part4.jsonl --output_bucket_dir gs://synthncc/eval/ --template_file sporsmal_evaluation_template.txt && \
python grpc_processor.py --input_file gs://synthncc/text_clean/edu2_ling1_no_clean_part4.jsonl --task eval_text --output_file edu2_ling1_no_text_clean_part4.jsonl --output_bucket_dir gs://synthncc/eval/ --template_file text_evaluation_template.txt && \
python grpc_processor.py --input_file gs://synthncc/reduced/edu2_ling1_no_part4.jsonl --task spm10 --output_file edu2_ling1_no_clean_spm10_part4.jsonl --output_bucket_dir gs://synthncc/sporsmal/ --template_file sporsmal10_template.txt

# Part 5
python grpc_processor.py --input_file gs://synthncc/spmrendered/edu2_ling1_no_clean_spm_part5_rendered.jsonl --task eval_spm --output_file edu2_ling1_no_clean_spm_rendered_part5.jsonl --output_bucket_dir gs://synthncc/eval/ --template_file sporsmal_evaluation_template.txt && \
python grpc_processor.py --input_file gs://synthncc/text_clean/edu2_ling1_no_clean_part5.jsonl --task eval_text --output_file edu2_ling1_no_text_clean_part5.jsonl --output_bucket_dir gs://synthncc/eval/ --template_file text_evaluation_template.txt && \
python grpc_processor.py --input_file gs://synthncc/reduced/edu2_ling1_no_part5.jsonl --task spm10 --output_file edu2_ling1_no_clean_spm10_part5.jsonl --output_bucket_dir gs://synthncc/sporsmal/ --template_file sporsmal10_template.txt

# Part 6
python grpc_processor.py --input_file gs://synthncc/spmrendered/edu2_ling1_no_clean_spm_part6_rendered.jsonl --task eval_spm --output_file edu2_ling1_no_clean_spm_rendered_part6.jsonl --output_bucket_dir gs://synthncc/eval/ --template_file sporsmal_evaluation_template.txt && \
python grpc_processor.py --input_file gs://synthncc/text_clean/edu2_ling1_no_clean_part6.jsonl --task eval_text --output_file edu2_ling1_no_text_clean_part6.jsonl --output_bucket_dir gs://synthncc/eval/ --template_file text_evaluation_template.txt && \
python grpc_processor.py --input_file gs://synthncc/reduced/edu2_ling1_no_part6.jsonl --task spm10 --output_file edu2_ling1_no_clean_spm10_part6.jsonl --output_bucket_dir gs://synthncc/sporsmal/ --template_file sporsmal10_template.txt

# Part 7
python grpc_processor.py --input_file gs://synthncc/spmrendered/edu2_ling1_no_clean_spm_part7_rendered.jsonl --task eval_spm --output_file edu2_ling1_no_clean_spm_rendered_part7.jsonl --output_bucket_dir gs://synthncc/eval/ --template_file sporsmal_evaluation_template.txt && \
python grpc_processor.py --input_file gs://synthncc/text_clean/edu2_ling1_no_clean_part7.jsonl --task eval_text --output_file edu2_ling1_no_text_clean_part7.jsonl --output_bucket_dir gs://synthncc/eval/ --template_file text_evaluation_template.txt && \
python grpc_processor.py --input_file gs://synthncc/reduced/edu2_ling1_no_part7.jsonl --task spm10 --output_file edu2_ling1_no_clean_spm10_part7.jsonl --output_bucket_dir gs://synthncc/sporsmal/ --template_file sporsmal10_template.txt

# Part 8
python grpc_processor.py --input_file gs://synthncc/spmrendered/edu2_ling1_no_clean_spm_part8_rendered.jsonl --task eval_spm --output_file edu2_ling1_no_clean_spm_rendered_part8.jsonl --output_bucket_dir gs://synthncc/eval/ --template_file sporsmal_evaluation_template.txt && \
python grpc_processor.py --input_file gs://synthncc/text_clean/edu2_ling1_no_clean_part8.jsonl --task eval_text --output_file edu2_ling1_no_text_clean_part8.jsonl --output_bucket_dir gs://synthncc/eval/ --template_file text_evaluation_template.txt && \
python grpc_processor.py --input_file gs://synthncc/reduced/edu2_ling1_no_part8.jsonl --task spm10 --output_file edu2_ling1_no_clean_spm10_part8.jsonl --output_bucket_dir gs://synthncc/sporsmal/ --template_file sporsmal10_template.txt
```

---

## 🌍 Multilingual Question Generation

```bash
python grpc_processor.py --input_file gs://synthncc/reduced/fineweb_en_part01.jsonl --task sporsmal_en --output_file fineweb_en_question_part01.jsonl --output_bucket_dir gs://synthncc/sporsmal/ --template_file sporsmal_template_en.txt && \
python grpc_processor.py --input_file gs://synthncc/reduced/edu15_sv_part01.jsonl --task sporsmal_sv --output_file edu15_sv_question_part01.jsonl --output_bucket_dir gs://synthncc/sporsmal/ --template_file sporsmal_template_sv.txt && \
python grpc_processor.py --input_file gs://synthncc/reduced/edu15_da_part01.jsonl --task sporsmal_da --output_file edu15_da_question_part01.jsonl --output_bucket_dir gs://synthncc/sporsmal/ --template_file sporsmal_template_da.txt && \
python grpc_processor.py --input_file gs://synthncc/reduced/nn_part01.jsonl --task sporsmal_nn --output_file nn_question_part01.jsonl --output_bucket_dir gs://synthncc/sporsmal/ --template_file sporsmal_template_nn.txt
```

---

## 🌍 Multilingual Question Evaluation

```bash
python grpc_processor.py --input_file gs://synthncc/spmrendered/fineweb_en_question_part01_rendered.jsonl --task eval_spm --output_file fineweb_en_question_part01_rendered_eval.jsonl --output_bucket_dir gs://synthncc/eval/ --template_file sporsmal_evaluation_template_multiling.txt && \
python grpc_processor.py --input_file gs://synthncc/spmrendered/edu15_da_question_part01_rendered.jsonl --task eval_spm --output_file edu15_da_question_part01_rendered_eval.jsonl --output_bucket_dir gs://synthncc/eval/ --template_file sporsmal_evaluation_template_multiling.txt && \
python grpc_processor.py --input_file gs://synthncc/spmrendered/edu15_sv_question_part01_rendered.jsonl --task eval_spm --output_file edu15_sv_question_part01_rendered_eval.jsonl --output_bucket_dir gs://synthncc/eval/ --template_file sporsmal_evaluation_template_multiling.txt && \
python grpc_processor.py --input_file gs://synthncc/spmrendered/nn_question_part01_rendered.jsonl --task eval_spm --output_file nn_question_part01_rendered_eval.jsonl --output_bucket_dir gs://synthncc/eval/ --template_file sporsmal_evaluation_template_multiling.txt && \
python grpc_processor.py --input_file gs://synthncc/spmrendered/edu2_ling1_no_clean_spm10_part1_rendered.jsonl --task eval_spm --output_file edu2_ling1_no_clean_spm10_part1_rendered_eval.jsonl --output_bucket_dir gs://synthncc/eval/ --template_file sporsmal_evaluation_template_multiling.txt
```

---

## 🧠 Flashcard QA Evaluation

```bash
cd /nfsmounts/datastore0/perk/enhancedNCC/3k_flashcards/
gsutil -m cp gs://synthncc/all_text_flash/*.* .
for i in $(seq -w 1 8); do \
  python extract_qa.py --input_file all_text_flash_part0${i}.jsonl --output_file all_text_flash_part0${i}_qa.jsonl; \
done
gsutil -m cp *_qa.jsonl gs://synthncc/flasheval/
```

```bash
python grpc_processor.py --input_file gs://synthncc/flasheval/all_text_flash_part02_qa.jsonl --task flasheval --output_file all_text_flash_part02_qa_eval.jsonl --output_bucket_dir gs://synthncc/flasheval/ --template_file flashcard_evaluation_template.txt
```

---

## 🔁 Translation Evaluation

```bash
python grpc_processor.py --input_file gs://synthncc/transeval/translation_english_norwegian_part1.jsonl --task transeval --output_file translation_english_norwegian_part1.jsonl --output_bucket_dir gs://synthncc/transeval/ --template_file translation_evaluation_template.txt


#gsutil -m cp translation_english_nynorsk_part* gs://synthncc/transeval/
python grpc_processor.py --input_file gs://synthncc/transeval/translation_english_nynorsk_part01.jsonl --task transeval --output_file translation_english_nynorsk_part01_eval.jsonl --output_bucket_dir gs://synthncc/transeval/ --template_file translation_evaluation_template_nbnn.txt
python grpc_processor.py --input_file gs://synthncc/transeval/translation_english_nynorsk_part02.jsonl --task transeval --output_file translation_english_nynorsk_part02_eval.jsonl --output_bucket_dir gs://synthncc/transeval/ --template_file translation_evaluation_template_nbnn.txt
python grpc_processor.py --input_file gs://synthncc/transeval/translation_english_nynorsk_part03.jsonl --task transeval --output_file translation_english_nynorsk_part03_eval.jsonl --output_bucket_dir gs://synthncc/transeval/ --template_file translation_evaluation_template_nbnn.txt
python grpc_processor.py --input_file gs://synthncc/transeval/translation_english_nynorsk_part04.jsonl --task transeval --output_file translation_english_nynorsk_part04_eval.jsonl --output_bucket_dir gs://synthncc/transeval/ --template_file translation_evaluation_template_nbnn.txt

#gsutil -m cp bokmal_nynorsk_nynorskpressekontor_part* gs://synthncc/transeval/
python grpc_processor.py --input_file gs://synthncc/transeval/translation_english_nynorsk_part01.jsonl --task transeval --output_file translation_english_nynorsk_part01_eval.jsonl --output_bucket_dir gs://synthncc/transeval/ --template_file translation_evaluation_template_ennn.txt
python grpc_processor.py --input_file gs://synthncc/transeval/translation_english_nynorsk_part02.jsonl --task transeval --output_file translation_english_nynorsk_part02_eval.jsonl --output_bucket_dir gs://synthncc/transeval/ --template_file translation_evaluation_template_ennn.txt

```
