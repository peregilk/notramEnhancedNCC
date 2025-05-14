---

### Flashcards

#### Eval and instruct generation:
```bash
python process_eval_flashcards.py \
  --input_file all_text_flash_complete_qa_eval.jsonl \
  --output_file all_text_flash_complete_qa_eval_best_llama3.jsonl \
  --chat_template meta-llama/Llama-3.1-8B-Instruct
```

#### Optionally (Note that I will only use part of this):
```bash
python process_eval_flashcards.py \
  --input_file all_text_flash_complete_qa_eval.jsonl \
  --output_file all_text_flash_complete_qa_eval_best.jsonl
```

#### Limit to 1M:
```bash
head -n 1000000 ../3k_flashcards/all_text_flash_complete_qa_eval_best.jsonl > train_all_text_flash_complete_qa_eval_best_1M.jsonl
```

#### Split rest:
```bash
tail -n +1000001 all_text_flash_complete_qa_eval_best_llama3.jsonl > ../3k_flashcards/train_all_text_flash_complete_qa_eval_best_rest_llama3.jsonl
```

---

### Translation

#### Download and preprocess:
```bash
python download_translation_datasets.py --output_file translation_english_norwegian.py
python add_textfield.py --input_file translation_english_norwegian.jsonl --output_file translation_english_norwegian_text.jsonl
```

#### Shuffle and split into 8 parts:
```bash
shuf translation_english_norwegian_text.jsonl -o shuffled.tmp && \
split -n l/8 -d --additional-suffix=.jsonl shuffled.tmp translation_english_norwegian_part && \
rm shuffled.tmp && \
for i in {0..7}; do mv translation_english_norwegian_part0${i}.jsonl translation_english_norwegian_part$((i+1)).jsonl; done
```

---

### Magpie / Magpie NO
Description TBD.

---

#### Source files:
```
/nfsmounts/ficino/lv_ai_2_ficino/perk/NCC2/instruction_sets_0/scripts/process_magpie_instructions_norwegian.py
/nfsmounts/ficino/lv_ai_2_ficino/perk/NCC2/instruction_sets_0/download/magpie_300k_norwegian_filtered.jsonl
```

---

### Multiturn

#### English multiturn (no NO data):
```bash
python download_multi_magpie.py --output_file multi_magpie_english.jsonl
cp multi_magpie_english.jsonl ../4a_evalueted_noglotlid/
```

---

#### Add GlotLID:
```bash
cd /nfsmounts/datastore0/perk/enhancedNCC/5a_cleaned_noglotlid
python annotate_multi_glotlid.py --input_dir ../5a_cleaned_noglotlid/ --output_dir ../5b_cleaned_glotlid/
```

#### Run semantic deduplication:
```bash
cd ../5b_cleaned_glotlid
for f in *.jsonl; do \
  python run_sem_dedup.py --input_file "$f" --output_file "../6c_cleaned_glotlid_semdedup/$f"; \
done
```

---

# How to regenerate

Run this in the root directory:

```bash
./create_tree.bash
```

# File Tree

- 1_start/external_culturax_da.jsonl
- 1_start/external_culturax_nn.jsonl
- 1_start/external_culturax_sv.jsonl
- 1_start/external_finewebedu_50b_en.jsonl
- 1_start/external_hplt_nn.jsonl
- 1_start/external_hplt_sv.jsonl
- 1_start/legal_edu2_ling1_glotlid_no.jsonl
- 1_start/open_wikipedia_da.jsonl
- 1_start/open_wikipedia_sv.jsonl
- 2_reduced/edu15_da.jsonl
- 2_reduced/edu15_da_part01.jsonl
- 2_reduced/edu15_da_part02.jsonl
- 2_reduced/edu15_da_part03.jsonl
- 2_reduced/edu15_da_part04.jsonl
- 2_reduced/edu15_da_part05.jsonl
- 2_reduced/edu15_da_part06.jsonl
- 2_reduced/edu15_da_part07.jsonl
- 2_reduced/edu15_da_part08.jsonl
- 2_reduced/edu15_sv.jsonl
- 2_reduced/edu15_sv_part01.jsonl
- 2_reduced/edu15_sv_part02.jsonl
- 2_reduced/edu15_sv_part03.jsonl
- 2_reduced/edu15_sv_part04.jsonl
- 2_reduced/edu15_sv_part05.jsonl
- 2_reduced/edu15_sv_part06.jsonl
- 2_reduced/edu15_sv_part07.jsonl
- 2_reduced/edu15_sv_part08.jsonl
- 2_reduced/edu2_ling1_no.jsonl
- 2_reduced/edu2_ling1_no_part1.jsonl
- 2_reduced/edu2_ling1_no_part2.jsonl
- 2_reduced/edu2_ling1_no_part3.jsonl
- 2_reduced/edu2_ling1_no_part4.jsonl
- 2_reduced/edu2_ling1_no_part5.jsonl
- 2_reduced/edu2_ling1_no_part5_subpart1.jsonl
- 2_reduced/edu2_ling1_no_part5_subpart2.jsonl
- 2_reduced/edu2_ling1_no_part5_subpart3.jsonl
- 2_reduced/edu2_ling1_no_part5_subpart4.jsonl
- 2_reduced/edu2_ling1_no_part5_subpart5.jsonl
- 2_reduced/edu2_ling1_no_part6.jsonl
- 2_reduced/edu2_ling1_no_part7.jsonl
- 2_reduced/edu2_ling1_no_part8.jsonl
- 2_reduced/filter.py
- 2_reduced/fineweb_en.jsonl
- 2_reduced/fineweb_en_part01.jsonl
- 2_reduced/fineweb_en_part02.jsonl
- 2_reduced/fineweb_en_part03.jsonl
- 2_reduced/fineweb_en_part04.jsonl
- 2_reduced/fineweb_en_part05.jsonl
- 2_reduced/fineweb_en_part06.jsonl
- 2_reduced/fineweb_en_part07.jsonl
- 2_reduced/fineweb_en_part08.jsonl
- 2_reduced/nn.jsonl
- 2_reduced/nn_part01.jsonl
- 2_reduced/nn_part02.jsonl
- 2_reduced/nn_part03.jsonl
- 2_reduced/nn_part04.jsonl
- 2_reduced/nn_part05.jsonl
- 2_reduced/nn_part06.jsonl
- 2_reduced/nn_part07.jsonl
- 2_reduced/nn_part08.jsonl
- 3a_clean/edu2_ling1_no_clean_all.jsonl
- 3a_clean/edu2_ling1_no_clean_part1.jsonl
- 3a_clean/edu2_ling1_no_clean_part2.jsonl
- 3a_clean/edu2_ling1_no_clean_part3.jsonl
- 3a_clean/edu2_ling1_no_clean_part4.jsonl
- 3a_clean/edu2_ling1_no_clean_part5.jsonl
- 3a_clean/edu2_ling1_no_clean_part5_subpart1.jsonl
- 3a_clean/edu2_ling1_no_clean_part5_subpart2.jsonl
- 3a_clean/edu2_ling1_no_clean_part5_subpart3.jsonl
- 3a_clean/edu2_ling1_no_clean_part5_subpart4.jsonl
- 3a_clean/edu2_ling1_no_clean_part5_subpart5.jsonl
- 3a_clean/edu2_ling1_no_clean_part6.jsonl
- 3a_clean/edu2_ling1_no_clean_part7.jsonl
- 3a_clean/edu2_ling1_no_clean_part8.jsonl
- 3a_clean/error_report.txt
- 3a_clean/filter.py
- 3a_clean/train_edu2_ling1_no_clean_all.jsonl
- 3b_spm/backup_old_process.py
- 3b_spm/edu15_da_question_part01.jsonl
- 3b_spm/edu15_da_question_part01_rendered.jsonl
- 3b_spm/edu15_da_question_part02.jsonl
- 3b_spm/edu15_da_question_part02_rendered.jsonl
- 3b_spm/edu15_da_question_part03_rendered.jsonl
- 3b_spm/edu15_da_question_part04.jsonl
- 3b_spm/edu15_da_question_part04_rendered.jsonl
- 3b_spm/edu15_da_question_part05.jsonl
- 3b_spm/edu15_da_question_part05_rendered.jsonl
- 3b_spm/edu15_da_question_part06_rendered.jsonl
- 3b_spm/edu15_da_question_part07.jsonl
- 3b_spm/edu15_da_question_part07_rendered.jsonl
- 3b_spm/edu15_da_question_part08.jsonl
- 3b_spm/edu15_da_question_part08_rendered.jsonl
- 3b_spm/edu15_sv_question_part01.jsonl
- 3b_spm/edu15_sv_question_part01_rendered.jsonl
- 3b_spm/edu15_sv_question_part02.jsonl
- 3b_spm/edu15_sv_question_part02_rendered.jsonl
- 3b_spm/edu15_sv_question_part03_rendered.jsonl
- 3b_spm/edu15_sv_question_part04.jsonl
- 3b_spm/edu15_sv_question_part04_rendered.jsonl
- 3b_spm/edu15_sv_question_part05.jsonl
- 3b_spm/edu15_sv_question_part05_rendered.jsonl
- 3b_spm/edu15_sv_question_part06_rendered.jsonl
- 3b_spm/edu15_sv_question_part07.jsonl
- 3b_spm/edu15_sv_question_part07_rendered.jsonl
- 3b_spm/edu15_sv_question_part08.jsonl
- 3b_spm/edu15_sv_question_part08_rendered.jsonl
- 3b_spm/edu2_ling1_no_clean_answer_part1.jsonl
- 3b_spm/edu2_ling1_no_clean_answer_part2.jsonl
- 3b_spm/edu2_ling1_no_clean_answer_part3.jsonl
- 3b_spm/edu2_ling1_no_clean_answer_part4.jsonl
- 3b_spm/edu2_ling1_no_clean_answer_part5.jsonl
- 3b_spm/edu2_ling1_no_clean_answer_part6.jsonl
- 3b_spm/edu2_ling1_no_clean_answer_part7.jsonl
- 3b_spm/edu2_ling1_no_clean_answer_part8.jsonl
- 3b_spm/edu2_ling1_no_clean_part1.jsonl
- 3b_spm/edu2_ling1_no_clean_part3.jsonl
- 3b_spm/edu2_ling1_no_clean_part4.jsonl
- 3b_spm/edu2_ling1_no_clean_spm10_part1.jsonl
- 3b_spm/edu2_ling1_no_clean_spm10_part1_rendered.jsonl
- 3b_spm/edu2_ling1_no_clean_spm10_part2.jsonl
- 3b_spm/edu2_ling1_no_clean_spm10_part2_rendered.jsonl
- 3b_spm/edu2_ling1_no_clean_spm10_part3.jsonl
- 3b_spm/edu2_ling1_no_clean_spm10_part3_rendered.jsonl
- 3b_spm/edu2_ling1_no_clean_spm10_part4.jsonl
- 3b_spm/edu2_ling1_no_clean_spm10_part4_rendered.jsonl
- 3b_spm/edu2_ling1_no_clean_spm10_part5.jsonl
- 3b_spm/edu2_ling1_no_clean_spm10_part5_rendered.jsonl
- 3b_spm/edu2_ling1_no_clean_spm10_part6.jsonl
- 3b_spm/edu2_ling1_no_clean_spm10_part6_rendered.jsonl
- 3b_spm/edu2_ling1_no_clean_spm10_part7.jsonl
- 3b_spm/edu2_ling1_no_clean_spm10_part7_rendered.jsonl
- 3b_spm/edu2_ling1_no_clean_spm10_part8.jsonl
- 3b_spm/edu2_ling1_no_clean_spm10_part8_rendered.jsonl
- 3b_spm/edu2_ling1_no_clean_spm_all.jsonl
- 3b_spm/edu2_ling1_no_clean_spm_part1.jsonl
- 3b_spm/edu2_ling1_no_clean_spm_part1_rendered.jsonl
- 3b_spm/edu2_ling1_no_clean_spm_part2.jsonl
- 3b_spm/edu2_ling1_no_clean_spm_part2_rendered.jsonl
- 3b_spm/edu2_ling1_no_clean_spm_part3.jsonl
- 3b_spm/edu2_ling1_no_clean_spm_part3_rendered.jsonl
- 3b_spm/edu2_ling1_no_clean_spm_part4.jsonl
- 3b_spm/edu2_ling1_no_clean_spm_part4_rendered.jsonl
- 3b_spm/edu2_ling1_no_clean_spm_part5.jsonl
- 3b_spm/edu2_ling1_no_clean_spm_part5_rendered.jsonl
- 3b_spm/edu2_ling1_no_clean_spm_part6.jsonl
- 3b_spm/edu2_ling1_no_clean_spm_part6_rendered.jsonl
- 3b_spm/edu2_ling1_no_clean_spm_part7.jsonl
- 3b_spm/edu2_ling1_no_clean_spm_part7_rendered.jsonl
- 3b_spm/edu2_ling1_no_clean_spm_part8.jsonl
- 3b_spm/edu2_ling1_no_clean_spm_part8_rendered.jsonl
- 3b_spm/edu2_ling1_no_clean_spm_rendered_all.jsonl
- 3b_spm/fineweb_en_question_part01.jsonl
- 3b_spm/fineweb_en_question_part01_rendered.jsonl
- 3b_spm/fineweb_en_question_part02.jsonl
- 3b_spm/fineweb_en_question_part02_rendered.jsonl
- 3b_spm/fineweb_en_question_part03.jsonl
- 3b_spm/fineweb_en_question_part03_rendered.jsonl
- 3b_spm/fineweb_en_question_part04.jsonl
- 3b_spm/fineweb_en_question_part04_rendered.jsonl
- 3b_spm/fineweb_en_question_part05.jsonl
- 3b_spm/fineweb_en_question_part05_rendered.jsonl
- 3b_spm/fineweb_en_question_part06.jsonl
- 3b_spm/fineweb_en_question_part06_rendered.jsonl
- 3b_spm/fineweb_en_question_part07.jsonl
- 3b_spm/fineweb_en_question_part07_rendered.jsonl
- 3b_spm/fineweb_en_question_part08.jsonl
- 3b_spm/fineweb_en_question_part08_rendered.jsonl
- 3b_spm/nn_question_part01.jsonl
- 3b_spm/nn_question_part01_rendered.jsonl
- 3b_spm/nn_question_part02.jsonl
- 3b_spm/nn_question_part02_rendered.jsonl
- 3b_spm/nn_question_part03_rendered.jsonl
- 3b_spm/nn_question_part04.jsonl
- 3b_spm/nn_question_part04_rendered.jsonl
- 3b_spm/nn_question_part05.jsonl
- 3b_spm/nn_question_part05_rendered.jsonl
- 3b_spm/nn_question_part06_rendered.jsonl
- 3b_spm/nn_question_part07.jsonl
- 3b_spm/nn_question_part07_rendered.jsonl
- 3b_spm/nn_question_part08.jsonl
- 3b_spm/nn_question_part08_rendered.jsonl
- 3b_spm/process_question.py
- 3b_spm/train_edu2_ling1_no_clean_spm_all_aug3.jsonl
- 3b_spm/train_edu2_ling1_no_clean_spm_all_shots5.jsonl
- 3c_spm_eval/edu15_da_question_part01_rendered_eval.jsonl
- 3c_spm_eval/edu15_da_question_part02_rendered_eval.jsonl
- 3c_spm_eval/edu15_da_question_part03_rendered_eval.jsonl
- 3c_spm_eval/edu15_da_question_part04_rendered_eval.jsonl
- 3c_spm_eval/edu15_da_question_part05_rendered_eval.jsonl
- 3c_spm_eval/edu15_da_question_part06_rendered_eval.jsonl
- 3c_spm_eval/edu15_da_question_part07_rendered_eval.jsonl
- 3c_spm_eval/edu15_da_question_part08_rendered_eval.jsonl
- 3c_spm_eval/edu15_da_question_rendered_eval_all.jsonl
- 3c_spm_eval/edu15_da_question_rendered_eval_aug1_llama3.jsonl
- 3c_spm_eval/edu15_da_question_rendered_eval_best_aug1.jsonl
- 3c_spm_eval/edu15_da_question_rendered_eval_best.jsonl
- 3c_spm_eval/edu15_sv_question_part01_rendered_eval.jsonl
- 3c_spm_eval/edu15_sv_question_part02_rendered_eval.jsonl
- 3c_spm_eval/edu15_sv_question_part03_rendered_eval.jsonl
- 3c_spm_eval/edu15_sv_question_part04_rendered_eval.jsonl
- 3c_spm_eval/edu15_sv_question_part05_rendered_eval.jsonl
- 3c_spm_eval/edu15_sv_question_part06_rendered_eval.jsonl
- 3c_spm_eval/edu15_sv_question_part07_rendered_eval.jsonl
- 3c_spm_eval/edu15_sv_question_part08_rendered_eval.jsonl
- 3c_spm_eval/edu15_sv_question_rendered_eval_all_aug1.jsonl
- 3c_spm_eval/edu15_sv_question_rendered_eval_all.jsonl
- 3c_spm_eval/edu15_sv_question_rendered_eval_aug1_llama3.jsonl
- 3c_spm_eval/edu15_sv_question_rendered_eval_best.jsonl
- 3c_spm_eval/edu2_ling1_no_clean_spm10_part1_rendered_eval.jsonl
- 3c_spm_eval/edu2_ling1_no_clean_spm10_part2_rendered_eval.jsonl
- 3c_spm_eval/edu2_ling1_no_clean_spm10_part3_rendered_eval.jsonl
- 3c_spm_eval/edu2_ling1_no_clean_spm10_part4_rendered_eval.jsonl
- 3c_spm_eval/edu2_ling1_no_clean_spm10_part5_rendered_eval.jsonl
- 3c_spm_eval/edu2_ling1_no_clean_spm10_part6_rendered_eval.jsonl
- 3c_spm_eval/edu2_ling1_no_clean_spm10_part7_rendered_eval.jsonl
- 3c_spm_eval/edu2_ling1_no_clean_spm10_part8_rendered_eval.jsonl
- 3c_spm_eval/edu2_ling1_no_clean_spm10_rendered_eval_all.jsonl
- 3c_spm_eval/edu2_ling1_no_clean_spm10_rendered_eval_best_aug2.jsonl
- 3c_spm_eval/edu2_ling1_no_clean_spm10_rendered_eval_best_llama3.jsonl
- 3c_spm_eval/edu2_ling1_no_clean_spm_rendered_all.jsonl
- 3c_spm_eval/edu2_ling1_no_clean_spm_rendered_merged_all.jsonl
- 3c_spm_eval/edu2_ling1_no_clean_spm_rendered_merged_best.jsonl
- 3c_spm_eval/edu2_ling1_no_clean_spm_rendered_merged_part1.jsonl
- 3c_spm_eval/edu2_ling1_no_clean_spm_rendered_part1.jsonl
- 3c_spm_eval/edu2_ling1_no_clean_spm_rendered_part2.jsonl
- 3c_spm_eval/edu2_ling1_no_clean_spm_rendered_part3.jsonl
- 3c_spm_eval/edu2_ling1_no_clean_spm_rendered_part4.jsonl
- 3c_spm_eval/edu2_ling1_no_clean_spm_rendered_part5.jsonl
- 3c_spm_eval/edu2_ling1_no_clean_spm_rendered_part6.jsonl
- 3c_spm_eval/edu2_ling1_no_clean_spm_rendered_part7.jsonl
- 3c_spm_eval/edu2_ling1_no_clean_spm_rendered_part8.jsonl
- 3c_spm_eval/edu2_ling1_no_text_clean_part1.jsonl
- 3c_spm_eval/edu2_ling1_no_text_clean_part2.jsonl
- 3c_spm_eval/edu2_ling1_no_text_clean_part3.jsonl
- 3c_spm_eval/edu2_ling1_no_text_clean_part4.jsonl
- 3c_spm_eval/edu2_ling1_no_text_clean_part5.jsonl
- 3c_spm_eval/edu2_ling1_no_text_clean_part6.jsonl
- 3c_spm_eval/edu2_ling1_no_text_clean_part7.jsonl
- 3c_spm_eval/edu2_ling1_no_text_clean_part8.jsonl
- 3c_spm_eval/filter_best.py
- 3c_spm_eval/fineweb_en_question_part01_rendered_eval.jsonl
- 3c_spm_eval/fineweb_en_question_part02_rendered_eval.jsonl
- 3c_spm_eval/fineweb_en_question_part03_rendered_eval.jsonl
- 3c_spm_eval/fineweb_en_question_part04_rendered_eval.jsonl
- 3c_spm_eval/fineweb_en_question_part05_rendered_eval.jsonl
- 3c_spm_eval/fineweb_en_question_part06_rendered_eval.jsonl
- 3c_spm_eval/fineweb_en_question_part07_rendered_eval.jsonl
- 3c_spm_eval/fineweb_en_question_part08_rendered_eval.jsonl
- 3c_spm_eval/fineweb_en_question_rendered_eval_all_aug1.jsonl
- 3c_spm_eval/fineweb_en_question_rendered_eval_all.jsonl
- 3c_spm_eval/fineweb_en_question_rendered_eval_aug1_llama3.jsonl
- 3c_spm_eval/fineweb_en_question_rendered_eval_best.jsonl
- 3c_spm_eval/merge_result_fields.py
- 3c_spm_eval/nn_question_part01_rendered_eval.jsonl
- 3c_spm_eval/nn_question_part02_rendered_eval.jsonl
- 3c_spm_eval/nn_question_part03_rendered_eval.jsonl
- 3c_spm_eval/nn_question_part04_rendered_eval.jsonl
- 3c_spm_eval/nn_question_part05_rendered_eval.jsonl
- 3c_spm_eval/nn_question_part06_rendered_eval.jsonl
- 3c_spm_eval/nn_question_part07_rendered_eval.jsonl
- 3c_spm_eval/nn_question_part08_rendered_eval.jsonl
- 3c_spm_eval/nn_question_rendered_eval_all_aug1.jsonl
- 3c_spm_eval/nn_question_rendered_eval_all.jsonl
- 3c_spm_eval/nn_question_rendered_eval_aug1_llama3.jsonl
- 3c_spm_eval/nn_question_rendered_eval_best.jsonl
- 3c_spm_eval/process_question.py
- 3c_spm_eval/prompts.txt
- 3c_spm_eval/result_stats.py
- 3c_spm_eval/testdelete.jsonl
- 3c_spm_eval/train_edu2_ling1_no_clean_spm_rendered_merged_best_aug2.jsonl
- 3c_spm_eval/train_edu2_ling1_no_clean_spm_rendered_merged_best_aug2_llama3.jsonl
- 3c_spm_eval/train_edu2_ling1_no_clean_spm_rendered_merged_best_shots5_100k.jsonl
- 3c_spm_eval/train_edu2_ling1_no_clean_spm_rendered_merged_best_shots5.jsonl
- 3d_text_eval/edu2_ling1_no_text_clean_all.jsonl
- 3d_text_eval/edu2_ling1_no_text_clean_part1.jsonl
- 3d_text_eval/edu2_ling1_no_text_clean_part2.jsonl
- 3d_text_eval/edu2_ling1_no_text_clean_part3.jsonl
- 3d_text_eval/edu2_ling1_no_text_clean_part4.jsonl
- 3d_text_eval/edu2_ling1_no_text_clean_part5.jsonl
- 3d_text_eval/edu2_ling1_no_text_clean_part6.jsonl
- 3d_text_eval/edu2_ling1_no_text_clean_part7.jsonl
- 3d_text_eval/edu2_ling1_no_text_clean_part8.jsonl
- 3d_text_eval/filter_best.py
- 3d_text_eval/result_stats.py
- 3d_text_eval/train_edu2_ling1_no_text_clean_best.jsonl
- 3e_10spm/edu2_ling1_no_clean_spm10_part1.jsonl_.gstmp
- 3e_10spm/edu2_ling1_no_clean_spm10_part2.jsonl_.gstmp
- 3e_10spm/edu2_ling1_no_clean_spm10_part3.jsonl_.gstmp
- 3e_10spm/edu2_ling1_no_clean_spm10_part4.jsonl_.gstmp
- 3e_10spm/edu2_ling1_no_clean_spm10_part5.jsonl_.gstmp
- 3e_10spm/edu2_ling1_no_clean_spm10_part6.jsonl_.gstmp
- 3e_10spm/edu2_ling1_no_clean_spm10_part7.jsonl_.gstmp
- 3e_10spm/edu2_ling1_no_clean_spm10_part8.jsonl_.gstmp
- 3f_all_text/all_text.jsonl
- 3f_all_text/all_text_part00.jsonl
- 3f_all_text/all_text_part01.jsonl
- 3f_all_text/all_text_part02.jsonl
- 3f_all_text/all_text_part03.jsonl
- 3f_all_text/all_text_part04.jsonl
- 3f_all_text/all_text_part05.jsonl
- 3f_all_text/all_text_part06.jsonl
- 3f_all_text/all_text_part07.jsonl
- 3g_magpie/analyse.py
- 3g_magpie/download_multi_magpie.py
- 3g_magpie/magpie_300k_english.jsonl
- 3g_magpie/magpie_300k_norwegian_filtered.jsonl
- 3g_magpie/magpie_en_llama3.jsonl
- 3g_magpie/magpie_no_llama3.jsonl
- 3g_magpie/multi_magpie_english.jsonl
- 3g_magpie/old_process_magpie_instructions_norwegian.py
- 3g_magpie/process_magpie_chat.py
- 3g_magpie/test.jsonl
- 3h_playwithwords/external_playwords_part1.jsonl
- 3h_playwithwords/process_play.py
- 3h_playwithwords/train_play.jsonl
- 3i_tinycode/process_tinycode.py
- 3i_tinycode/tiny_code_singleshot.jsonl
- 3i_tinycode/train_tinycode_llama3.py
- 3j_translation/add_textfield.py
- 3j_translation/download_translation_datasets.py
- 3j_translation/translation_bokmaal_nynorsk.py
- 3j_translation/translation_english_norwegian.jsonl
- 3j_translation/translation_english_norwegian_part1.jsonl
- 3j_translation/translation_english_norwegian_part2.jsonl
- 3j_translation/translation_english_norwegian_part3.jsonl
- 3j_translation/translation_english_norwegian_part4.jsonl
- 3j_translation/translation_english_norwegian_part5.jsonl
- 3j_translation/translation_english_norwegian_part6.jsonl
- 3j_translation/translation_english_norwegian_part7.jsonl
- 3j_translation/translation_english_norwegian_part8.jsonl
- 3j_translation/translation_english_norwegian_text.jsonl
- 3j_translation/translation_english_nynorsk.py
- 3k_flashcards/all_text_flash_complete_qa_eval_best.jsonl
- 3k_flashcards/all_text_flash_complete_qa_eval_best_llama3.jsonl
- 3k_flashcards/all_text_flash_complete_qa_eval.jsonl
- 3k_flashcards/all_text_flash_part00.jsonl
- 3k_flashcards/all_text_flash_part00_qa_eval.jsonl
- 3k_flashcards/all_text_flash_part00_qa.jsonl
- 3k_flashcards/all_text_flash_part01.jsonl
- 3k_flashcards/all_text_flash_part01_qa_eval.jsonl
- 3k_flashcards/all_text_flash_part01_qa.jsonl
- 3k_flashcards/all_text_flash_part02.jsonl
- 3k_flashcards/all_text_flash_part02_qa_eval.jsonl
- 3k_flashcards/all_text_flash_part02_qa.jsonl
- 3k_flashcards/all_text_flash_part03.jsonl
- 3k_flashcards/all_text_flash_part03_qa_eval.jsonl
- 3k_flashcards/all_text_flash_part03_qa.jsonl
- 3k_flashcards/all_text_flash_part04.jsonl
- 3k_flashcards/all_text_flash_part04_qa_eval.jsonl
- 3k_flashcards/all_text_flash_part04_qa.jsonl
- 3k_flashcards/all_text_flash_part05.jsonl
- 3k_flashcards/all_text_flash_part05_qa_eval.jsonl
- 3k_flashcards/all_text_flash_part05_qa.jsonl
- 3k_flashcards/all_text_flash_part06.jsonl
- 3k_flashcards/all_text_flash_part06_qa_eval.jsonl
- 3k_flashcards/all_text_flash_part06_qa.jsonl
- 3k_flashcards/all_text_flash_part07.jsonl
- 3k_flashcards/all_text_flash_part07_qa_eval.jsonl
- 3k_flashcards/all_text_flash_part07_qa.jsonl
- 3k_flashcards/extract_qa.py
- 3k_flashcards/process_eval_flashcards.py
- 3k_flashcards/testdelete.jsonl
- 3x_testcorpus/corpus/train_aa.jsonl
- 3x_testcorpus/corpus/train_ab.jsonl
- 3x_testcorpus/corpus/train_ac.jsonl
- 3x_testcorpus/corpus/train_ad.jsonl
- 3x_testcorpus/corpus/train_ae.jsonl
- 3x_testcorpus/corpus/train_af.jsonl
- 3x_testcorpus/corpus/train_ag.jsonl
- 3x_testcorpus/corpus/train_ah.jsonl
- 3x_testcorpus/corpus/train_ai.jsonl
- 3x_testcorpus/corpus/train_aj.jsonl
- 3x_testcorpus/corpus/train_ak.jsonl
- 3x_testcorpus/corpus/train_al.jsonl
- 3x_testcorpus/corpus/train_am.jsonl
- 3x_testcorpus/corpus/train_an.jsonl
- 3x_testcorpus/corpus/train_ao.jsonl
- 3x_testcorpus/corpus/train_ap.jsonl
- 3x_testcorpus/corpus/train_aq.jsonl
- 3x_testcorpus/corpus/train_ar.jsonl
- 3x_testcorpus/corpus/train_as.jsonl
- 3x_testcorpus/corpus/train_at.jsonl
- 3x_testcorpus/corpus/train_au.jsonl
- 3x_testcorpus/corpus/train_av.jsonl
- 3x_testcorpus/corpus/train_aw.jsonl
- 3x_testcorpus/corpus/train_ax.jsonl
- 3x_testcorpus/corpus/train_ay.jsonl
- 3x_testcorpus/corpus/train_az.jsonl
- 3x_testcorpus/corpus/train_ba.jsonl
- 3x_testcorpus/corpus/train_bb.jsonl
- 3x_testcorpus/corpus/train_bc.jsonl
- 3x_testcorpus/corpus/train_bd.jsonl
- 3x_testcorpus/corpus/train_be.jsonl
- 3x_testcorpus/corpus/train_bf.jsonl
- 3x_testcorpus/corpus/train_bg.jsonl
- 3x_testcorpus/corpus/train_bh.jsonl
- 3x_testcorpus/corpus/train_bi.jsonl
- 3x_testcorpus/corpus/train_bj.jsonl
- 3x_testcorpus/corpus/train_bk.jsonl
- 3x_testcorpus/corpus/train_bl.jsonl
- 3x_testcorpus/corpus/train_bm.jsonl
- 3x_testcorpus/corpus/train_bn.jsonl
- 3x_testcorpus/corpus/train_bo.jsonl
- 3x_testcorpus/corpus/train_bp.jsonl
- 3x_testcorpus/corpus/train_bq.jsonl
- 3x_testcorpus/corpus/train_br.jsonl
- 3x_testcorpus/corpus/train_bs.jsonl
- 3x_testcorpus/corpus/train_bt.jsonl
- 3x_testcorpus/corpus/train_bu.jsonl
- 3x_testcorpus/corpus/train_bv.jsonl
- 3x_testcorpus/corpus/train_bw.jsonl
- 3x_testcorpus/corpus/train_bx.jsonl
- 3x_testcorpus/corpus/train_by.jsonl
- 3x_testcorpus/corpus/train_bz.jsonl
- 3x_testcorpus/corpus/train_ca.jsonl
- 3x_testcorpus/corpus/train_cb.jsonl
- 3x_testcorpus/corpus/train_cc.jsonl
- 3x_testcorpus/corpus/train_cd.jsonl
- 3x_testcorpus/corpus/train_ce.jsonl
- 3x_testcorpus/corpus/train_cf.jsonl
- 3x_testcorpus/corpus/train_cg.jsonl
- 3x_testcorpus/corpus/train_ch.jsonl
- 3x_testcorpus/corpus/train_ci.jsonl
- 3x_testcorpus/corpus/train_cj.jsonl
- 3x_testcorpus/corpus/train_ck.jsonl
- 3x_testcorpus/corpus/train_cl.jsonl
- 3x_testcorpus/corpus/train_cm.jsonl
- 3x_testcorpus/corpus/train_cn.jsonl
- 3x_testcorpus/corpus/train_co.jsonl
- 3x_testcorpus/corpus/train_cp.jsonl
- 3x_testcorpus/corpus/train_cq.jsonl
- 3x_testcorpus/corpus/train_cr.jsonl
- 3x_testcorpus/corpus/train_cs.jsonl
- 3x_testcorpus/corpus/train_ct.jsonl
- 3x_testcorpus/corpus/train_cu.jsonl
- 3x_testcorpus/corpus/train_cv.jsonl
- 3x_testcorpus/corpus/train_cw.jsonl
- 3x_testcorpus/corpus/train_cx.jsonl
- 3x_testcorpus/corpus/train_cy.jsonl
- 3x_testcorpus/corpus/train_cz.jsonl
- 3x_testcorpus/corpus/train_da.jsonl
- 3x_testcorpus/corpus/train_db.jsonl
- 3x_testcorpus/corpus/train_dc.jsonl
- 3x_testcorpus/corpus/train_dd.jsonl
- 3x_testcorpus/corpus/train_de.jsonl
- 3x_testcorpus/corpus/train_df.jsonl
- 3x_testcorpus/corpus/train_dg.jsonl
- 3x_testcorpus/corpus/train_dh.jsonl
- 3x_testcorpus/corpus/train_di.jsonl
- 3x_testcorpus/corpus/train_dj.jsonl
- 3x_testcorpus/corpus/train_dk.jsonl
- 3x_testcorpus/corpus/train_dl.jsonl
- 3x_testcorpus/corpus/train_dm.jsonl
- 3x_testcorpus/corpus/train_dn.jsonl
- 3x_testcorpus/corpus/train_do.jsonl
- 3x_testcorpus/corpus/train_dp.jsonl
- 3x_testcorpus/corpus/train_dq.jsonl
- 3x_testcorpus/corpus/train_dr.jsonl
- 3x_testcorpus/corpus/train_ds.jsonl
- 3x_testcorpus/corpus/train_dt.jsonl
- 3x_testcorpus/corpus/train_du.jsonl
- 3x_testcorpus/corpus/train_dv.jsonl
- 3x_testcorpus/corpus/train_dw.jsonl
- 3x_testcorpus/corpus/train_dx.jsonl
- 3x_testcorpus/corpus/train_dy.jsonl
- 3x_testcorpus/corpus/train_dz.jsonl
- 3x_testcorpus/corpus/train_ea.jsonl
- 3x_testcorpus/corpus/train_eb.jsonl
- 3x_testcorpus/corpus/train_ec.jsonl
- 3x_testcorpus/corpus/train_ed.jsonl
- 3x_testcorpus/corpus/train_ee.jsonl
- 3x_testcorpus/corpus/train_ef.jsonl
- 3x_testcorpus/corpus/train_eg.jsonl
- 3x_testcorpus/corpus/train_eh.jsonl
- 3x_testcorpus/corpus/train_ei.jsonl
- 3x_testcorpus/corpus/train_ej.jsonl
- 3x_testcorpus/corpus/train_ek.jsonl
- 3x_testcorpus/corpus/train_el.jsonl
- 3x_testcorpus/corpus/train_em.jsonl
- 3x_testcorpus/corpus/train_en.jsonl
- 3x_testcorpus/corpus/train_eo.jsonl
- 3x_testcorpus/corpus/train_ep.jsonl
- 3x_testcorpus/corpus/train_eq.jsonl
- 3x_testcorpus/corpus/train_er.jsonl
- 3x_testcorpus/corpus/train_es.jsonl
- 3x_testcorpus/corpus/train_et.jsonl
- 3x_testcorpus/corpus/train_eu.jsonl
- 3x_testcorpus/corpus/train_ev.jsonl
- 3x_testcorpus/corpus/train_ew.jsonl
- 3x_testcorpus/corpus/train_ex.jsonl
- 3x_testcorpus/corpus/train_ey.jsonl
- 3x_testcorpus/corpus/train_ez.jsonl
- 3x_testcorpus/corpus/train_fa.jsonl
- 3x_testcorpus/corpus/train_fb.jsonl
- 3x_testcorpus/corpus/train_fc.jsonl
- 3x_testcorpus/corpus/train_fd.jsonl
- 3x_testcorpus/corpus/train_fe.jsonl
- 3x_testcorpus/corpus/train_ff.jsonl
- 3x_testcorpus/corpus/train_fg.jsonl
- 3x_testcorpus/corpus/train_fh.jsonl
- 3x_testcorpus/corpus/train_fi.jsonl
- 3x_testcorpus/corpus/train_fj.jsonl
- 3x_testcorpus/corpus/train_fk.jsonl
- 3x_testcorpus/corpus/train_fl.jsonl
- 3x_testcorpus/corpus/train_fm.jsonl
- 3x_testcorpus/corpus/train_fn.jsonl
- 3x_testcorpus/corpus/train_fo.jsonl
- 3x_testcorpus/corpus/train_fp.jsonl
- 3x_testcorpus/corpus/train_fq.jsonl
- 3x_testcorpus/corpus/train_fr.jsonl
- 3x_testcorpus/corpus/train_fs.jsonl
- 3x_testcorpus/corpus/train_ft.jsonl
- 3x_testcorpus/corpus/train_fu.jsonl
- 3x_testcorpus/corpus/train_fv.jsonl
- 3x_testcorpus/corpus/train_fw.jsonl
- 3x_testcorpus/corpus/train_fx.jsonl
- 3x_testcorpus/corpus/train_fy.jsonl
- 3x_testcorpus/corpus/train_fz.jsonl
- 3x_testcorpus/corpus/train_ga.jsonl
- 3x_testcorpus/corpus/train_gb.jsonl
- 3x_testcorpus/corpus/train_gc.jsonl
- 3x_testcorpus/corpus/train_gd.jsonl
- 3x_testcorpus/corpus/train_ge.jsonl
- 3x_testcorpus/corpus/train_gf.jsonl
- 3x_testcorpus/corpus/train_gg.jsonl
- 3x_testcorpus/corpus/train_gh.jsonl
- 3x_testcorpus/corpus/train_gi.jsonl
- 3x_testcorpus/corpus/train_gj.jsonl
- 3x_testcorpus/corpus/train_gk.jsonl
- 3x_testcorpus/corpus/train_gl.jsonl
- 3x_testcorpus/corpus/train_gm.jsonl
- 3x_testcorpus/corpus/train_gn.jsonl
- 3x_testcorpus/corpus/train_go.jsonl
- 3x_testcorpus/corpus/train_gp.jsonl
- 3x_testcorpus/corpus/train_gq.jsonl
- 3x_testcorpus/corpus/train_gr.jsonl
- 3x_testcorpus/corpus/train_gs.jsonl
- 3x_testcorpus/corpus/train_gt.jsonl
- 3x_testcorpus/corpus/train_gu.jsonl
- 3x_testcorpus/corpus/train_gv.jsonl
- 3x_testcorpus/corpus/train_gw.jsonl
- 3x_testcorpus/corpus/train_gx.jsonl
- 3x_testcorpus/corpus/train_gy.jsonl
- 3x_testcorpus/corpus/train_gz.jsonl
- 3x_testcorpus/corpus/train_ha.jsonl
- 3x_testcorpus/corpus/train_hb.jsonl
- 3x_testcorpus/corpus/train_hc.jsonl
- 3x_testcorpus/corpus/train_hd.jsonl
- 3x_testcorpus/corpus/train_he.jsonl
- 3x_testcorpus/corpus/train_hf.jsonl
- 3x_testcorpus/corpus/train_hg.jsonl
- 3x_testcorpus/corpus/train_hh.jsonl
- 3x_testcorpus/corpus/train_hi.jsonl
- 3x_testcorpus/corpus/train_hj.jsonl
- 3x_testcorpus/corpus/train_hk.jsonl
- 3x_testcorpus/corpus/train_hl.jsonl
- 3x_testcorpus/corpus/train_hm.jsonl
- 3x_testcorpus/corpus/train_hn.jsonl
- 3x_testcorpus/corpus/train_ho.jsonl
- 3x_testcorpus/corpus/train_hp.jsonl
- 3x_testcorpus/corpus/train_hq.jsonl
- 3x_testcorpus/corpus/train_hr.jsonl
- 3x_testcorpus/corpus/train_hs.jsonl
- 3x_testcorpus/corpus/train_ht.jsonl
- 3x_testcorpus/corpus/train_hu.jsonl
- 3x_testcorpus/corpus/train_hv.jsonl
- 3x_testcorpus/corpus/train_hw.jsonl
- 3x_testcorpus/corpus/train_hx.jsonl
- 3x_testcorpus/corpus/train_hy.jsonl
- 3x_testcorpus/corpus/train_hz.jsonl
- 3x_testcorpus/corpus/train_ia.jsonl
- 3x_testcorpus/corpus/train_ib.jsonl
- 3x_testcorpus/corpus/train_ic.jsonl
- 3x_testcorpus/corpus/train_id.jsonl
- 3x_testcorpus/corpus/train_ie.jsonl
- 3x_testcorpus/corpus/train_if.jsonl
- 3x_testcorpus/corpus/train_ig.jsonl
- 3x_testcorpus/corpus/train_ih.jsonl
- 3x_testcorpus/corpus/train_ii.jsonl
- 3x_testcorpus/corpus/train_ij.jsonl
- 3x_testcorpus/corpus/train_ik.jsonl
- 3x_testcorpus/corpus/train_il.jsonl
- 3x_testcorpus/corpus/train_im.jsonl
- 3x_testcorpus/corpus/train_in.jsonl
- 3x_testcorpus/corpus/train_io.jsonl
- 3x_testcorpus/corpus/train_ip.jsonl
- 3x_testcorpus/corpus/train_iq.jsonl
- 3x_testcorpus/corpus/train_ir.jsonl
- 3x_testcorpus/corpus/train_is.jsonl
- 3x_testcorpus/corpus/train_it.jsonl
- 3x_testcorpus/corpus/train_iu.jsonl
- 3x_testcorpus/corpus/train_iv.jsonl
- 3x_testcorpus/corpus/train_iw.jsonl
- 3x_testcorpus/corpus/train_ix.jsonl
- 3x_testcorpus/corpus/train_iy.jsonl
- 3x_testcorpus/corpus/train_iz.jsonl
- 3x_testcorpus/corpus/train_ja.jsonl
- 3x_testcorpus/corpus/train_jb.jsonl
- 3x_testcorpus/corpus/train_jc.jsonl
- 3x_testcorpus/corpus/train_jd.jsonl
- 3x_testcorpus/corpus/train_je.jsonl
- 3x_testcorpus/corpus/train_jf.jsonl
- 3x_testcorpus/corpus/train_jg.jsonl
- 3x_testcorpus/corpus/train_jh.jsonl
- 3x_testcorpus/corpus/train_ji.jsonl
- 3x_testcorpus/corpus/train_jj.jsonl
- 3x_testcorpus/corpus/train_jk.jsonl
- 3x_testcorpus/corpus/train_jl.jsonl
- 3x_testcorpus/corpus/train_jm.jsonl
- 3x_testcorpus/corpus/train_jn.jsonl
- 3x_testcorpus/corpus/train_jo.jsonl
- 3x_testcorpus/corpus/train_jp.jsonl
- 3x_testcorpus/corpus/train_jq.jsonl
- 3x_testcorpus/corpus/train_jr.jsonl
- 3x_testcorpus/corpus/train_js.jsonl
- 3x_testcorpus/corpus/train_jt.jsonl
- 3x_testcorpus/corpus/train_ju.jsonl
- 3x_testcorpus/corpus/train_jv.jsonl
- 3x_testcorpus/corpus/train_jw.jsonl
- 3x_testcorpus/train_edu15_da.jsonl
- 3x_testcorpus/train_edu15_sv.jsonl
- 3x_testcorpus/train_edu2_ling1_no_clean_all.jsonl
- 3x_testcorpus/train_edu2_ling1_no_clean_spm_all_aug3.jsonl
- 3x_testcorpus/train_edu2_ling1_no_clean_spm_all_shots5.jsonl
- 3x_testcorpus/train_fineweb_en.jsonl
- 3x_testcorpus/train_nn.jsonl
- 4a_evalueted_noglotlid/clean_multi.py
- 4a_evalueted_noglotlid/edu15_da_question_rendered_eval_aug1_llama3.jsonl
- 4a_evalueted_noglotlid/edu15_da_question_rendered_eval_best_aug1.jsonl
- 4a_evalueted_noglotlid/edu15_da_question_rendered_eval_best.jsonl
- 4a_evalueted_noglotlid/edu15_reduced_da.jsonl
- 4a_evalueted_noglotlid/edu15_reduced_sv.jsonl
- 4a_evalueted_noglotlid/edu15_sv_question_rendered_eval_all_aug1.jsonl
- 4a_evalueted_noglotlid/edu15_sv_question_rendered_eval_aug1_llama3.jsonl
- 4a_evalueted_noglotlid/edu15_sv_question_rendered_eval_best.jsonl
- 4a_evalueted_noglotlid/edu2_ling1_no_clean_spm10_rendered_eval_best_aug2.jsonl
- 4a_evalueted_noglotlid/edu2_ling1_no_clean_spm10_rendered_eval_best_llama3.jsonl
- 4a_evalueted_noglotlid/fineweb_en_question_rendered_eval_all_aug1.jsonl
- 4a_evalueted_noglotlid/fineweb_en_question_rendered_eval_aug1_llama3.jsonl
- 4a_evalueted_noglotlid/fineweb_en_question_rendered_eval_best.jsonl
- 4a_evalueted_noglotlid/fineweb_en_reduced.jsonl
- 4a_evalueted_noglotlid/magpie_en_llama3.jsonl
- 4a_evalueted_noglotlid/magpie_no_llama3.jsonl
- 4a_evalueted_noglotlid/multi_magpie_english.jsonl
- 4a_evalueted_noglotlid/nn_question_rendered_eval_all_aug1.jsonl
- 4a_evalueted_noglotlid/nn_question_rendered_eval_all.jsonl
- 4a_evalueted_noglotlid/nn_question_rendered_eval_aug1_llama3.jsonl
- 4a_evalueted_noglotlid/nn_reduced.jsonl
- 4a_evalueted_noglotlid/stats.py
- 4a_evalueted_noglotlid/train_all_text_flash_complete_qa_eval_best_1M.jsonl
- 4a_evalueted_noglotlid/train_all_text_flash_complete_qa_eval_best_rest_llama3.jsonl
- 4a_evalueted_noglotlid/train_edu2_ling1_no_clean_spm_rendered_merged_best_aug2.jsonl
- 4a_evalueted_noglotlid/train_edu2_ling1_no_clean_spm_rendered_merged_best_shots5_100k.jsonl
- 4a_evalueted_noglotlid/train_edu2_ling1_no_text_clean_best.jsonl
- 4a_evalueted_noglotlid/train_play_llama3.jsonl
- 4a_evalueted_noglotlid/train_tinycode_llama3.jsonl
- 5a_cleaned_noglotlid/annotate_multi_glotlid.py
- 5a_cleaned_noglotlid/annotate_single_glotlid.py
- 5a_cleaned_noglotlid/edu15_da_question_rendered_eval_aug1_llama3.jsonl
- 5a_cleaned_noglotlid/edu15_da_question_rendered_eval_best_aug1.jsonl
- 5a_cleaned_noglotlid/edu15_da_question_rendered_eval_best.jsonl
- 5a_cleaned_noglotlid/edu15_reduced_da.jsonl
- 5a_cleaned_noglotlid/edu15_reduced_sv.jsonl
- 5a_cleaned_noglotlid/edu15_sv_question_rendered_eval_all_aug1.jsonl
- 5a_cleaned_noglotlid/edu15_sv_question_rendered_eval_aug1_llama3.jsonl
- 5a_cleaned_noglotlid/edu15_sv_question_rendered_eval_best.jsonl
- 5a_cleaned_noglotlid/edu2_ling1_no_clean_spm10_rendered_eval_best_aug2.jsonl
- 5a_cleaned_noglotlid/edu2_ling1_no_clean_spm10_rendered_eval_best_llama3.jsonl
- 5a_cleaned_noglotlid/fineweb_en_question_rendered_eval_all_aug1.jsonl
- 5a_cleaned_noglotlid/fineweb_en_question_rendered_eval_aug1_llama3.jsonl
- 5a_cleaned_noglotlid/fineweb_en_question_rendered_eval_best.jsonl
- 5a_cleaned_noglotlid/fineweb_en_reduced.jsonl
- 5a_cleaned_noglotlid/lid.176.ftz
- 5a_cleaned_noglotlid/magpie_en_llama3.jsonl
- 5a_cleaned_noglotlid/magpie_no_llama3.jsonl
- 5a_cleaned_noglotlid/multi_magpie_english.jsonl
- 5a_cleaned_noglotlid/nn_question_rendered_eval_all_aug1.jsonl
- 5a_cleaned_noglotlid/nn_question_rendered_eval_all.jsonl
- 5a_cleaned_noglotlid/nn_question_rendered_eval_aug1_llama3.jsonl
- 5a_cleaned_noglotlid/nn_reduced.jsonl
- 5a_cleaned_noglotlid/stats.py
- 5a_cleaned_noglotlid/train_all_text_flash_complete_qa_eval_best_1M.jsonl
- 5a_cleaned_noglotlid/train_all_text_flash_complete_qa_eval_best_rest_llama3.jsonl
- 5a_cleaned_noglotlid/train_edu2_ling1_no_clean_spm_rendered_merged_best_aug2.jsonl
- 5a_cleaned_noglotlid/train_edu2_ling1_no_clean_spm_rendered_merged_best_shots5_100k.jsonl
- 5a_cleaned_noglotlid/train_edu2_ling1_no_text_clean_best.jsonl
- 5a_cleaned_noglotlid/train_play_llama3.jsonl
- 5a_cleaned_noglotlid/train_tinycode_llama3.jsonl
- 5b_cleaned_glotlid/edu15_da_question_rendered_eval_aug1_llama3.jsonl
- 5b_cleaned_glotlid/edu15_da_question_rendered_eval_best_aug1.jsonl
- 5b_cleaned_glotlid/edu15_da_question_rendered_eval_best.jsonl
- 5b_cleaned_glotlid/edu15_reduced_da.jsonl
- 5b_cleaned_glotlid/edu15_reduced_sv.jsonl
- 5b_cleaned_glotlid/edu15_sv_question_rendered_eval_all_aug1.jsonl
- 5b_cleaned_glotlid/edu15_sv_question_rendered_eval_aug1_llama3.jsonl
- 5b_cleaned_glotlid/edu15_sv_question_rendered_eval_best.jsonl
- 5b_cleaned_glotlid/edu2_ling1_no_clean_spm10_rendered_eval_best_aug2.jsonl
- 5b_cleaned_glotlid/edu2_ling1_no_clean_spm10_rendered_eval_best_llama3.jsonl
- 5b_cleaned_glotlid/excerpt.csv
- 5b_cleaned_glotlid/fineweb_en_question_rendered_eval_all_aug1.jsonl
- 5b_cleaned_glotlid/fineweb_en_question_rendered_eval_aug1_llama3.jsonl
- 5b_cleaned_glotlid/fineweb_en_question_rendered_eval_best.jsonl
- 5b_cleaned_glotlid/fineweb_en_reduced.jsonl
- 5b_cleaned_glotlid/first_line_csv.py
- 5b_cleaned_glotlid/magpie_en_llama3.jsonl
- 5b_cleaned_glotlid/magpie_no_llama3.jsonl
- 5b_cleaned_glotlid/multi_magpie_english.jsonl
- 5b_cleaned_glotlid/nn_question_rendered_eval_all_aug1.jsonl
- 5b_cleaned_glotlid/nn_question_rendered_eval_all.jsonl
- 5b_cleaned_glotlid/nn_question_rendered_eval_aug1_llama3.jsonl
- 5b_cleaned_glotlid/nn_reduced.jsonl
- 5b_cleaned_glotlid/run_sem_dedup.py
- 5b_cleaned_glotlid/train_all_text_flash_complete_qa_eval_best_1M.jsonl
- 5b_cleaned_glotlid/train_all_text_flash_complete_qa_eval_best_rest_llama3.jsonl
- 5b_cleaned_glotlid/train_edu2_ling1_no_clean_spm_rendered_merged_best_aug2.jsonl
- 5b_cleaned_glotlid/train_edu2_ling1_no_clean_spm_rendered_merged_best_shots5_100k.jsonl
- 5b_cleaned_glotlid/train_edu2_ling1_no_text_clean_best.jsonl
- 5b_cleaned_glotlid/train_play_llama3.jsonl
- 5b_cleaned_glotlid/train_tinycode_llama3.jsonl
- 6c_cleaned_glotlid_semdedup/edu15_da_question_rendered_eval_aug1_llama3.jsonl
- 6c_cleaned_glotlid_semdedup/edu15_da_question_rendered_eval_best_aug1.jsonl
- 6c_cleaned_glotlid_semdedup/edu15_da_question_rendered_eval_best.jsonl
- create_tree.bash
- error_report.txt
- .gitignore
- LICENSE
- README.md

