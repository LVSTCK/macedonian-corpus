# 🇲🇰 Macedonian Corpus

This repository contains scripts, notebooks, and documentation to process, clean, and manage the [Macedonian Corpus Raw](https://huggingface.co/datasets/LVSTCK/macedonian-corpus-raw). 

---

## Overview

### **1. `filtering/`**
This folder contains the primary scripts for downloading, filtering, and preparing the clean version  of corpus (input: [raw corpus](https://huggingface.co/datasets/LVSTCK/macedonian-corpus-raw), output: [cleaned corpus](https://huggingface.co/datasets/LVSTCK/macedonian-corpus-cleaned). 

- **🧹 `filter.py`**
  - **Purpose:** Produces a cleaned version of the dataset (filtering process inspired by fineweb-2).
  - **Features:**
    - C4-like filtering (removing irrelevant lines, low-quality text, and placeholder content).
    - Gopher-like filtering (handling incomplete or overly repetitive documents).
    - High-confidence language detection for Macedonian text.
    - Sentence deduplication to avoid redundancy.
    - Personally Identifiable Information (PII) filtering.

- **📥 `download.py`**
  - **Purpose:** Downloads the raw dataset (`macedonian-corpus-raw`) from its source. It is advisable to split the dataset into chunks (using `split_data/`) for efficient multiprocessing.

- **🔀 `split_data/` (optional)**
  - **Purpose:** Contains split chunks of the downloaded corpus to exploit multiprocessing during filtering.

- **🧪 `test_language_model.py`**
  - **Purpose:** Evaluates the outputs of language detection models.
  - **Usage:** Useful for testing the language filtering logic.

- **👥 `dedup/minhash.py`**
  - **Purpose:** Second stage deduplication. (input: [cleaned](https://huggingface.co/datasets/LVSTCK/macedonian-corpus-cleaned), output: [cleaned and deduplicated](https://huggingface.co/datasets/LVSTCK/macedonian-corpus-cleaned-dedup) 

---

### **2. `process_data/`**
This folder contains notebooks used to process and unify various Macedonian text sources into the raw dataset.

- **📄 `common_voice.ipynb`**
  - **Purpose:** Extracts text from the [Common Voice dataset](https://commonvoice.mozilla.org/), specifically from `.tsv` files.

- **🔗 `consolidate_data.ipynb`**
  - **Purpose:** Unifies all data sources into a single raw dataset (`macedonian-corpus-raw`).
  - **Usage:** Combines text data from multiple sources (e.g., Common Voice, scraped PDFs, web crawls). For reference, see the dataset description [here](https://huggingface.co/datasets/LVSTCK/macedonian-corpus-raw).

---

### **3. `scraping/`**
This folder contains scripts for data collection through web scraping.

- **🖋️ `scrape_pdfs.py`**
  - **Purpose:** Scrapes text from PDFs.
  - **Usage:** The extracted content is processed through [MMORE](https://github.com/swiss-ai/mmore) and included in both the raw and cleaned datasets, with field 'source' == MMORE. 

---

## 🗃️ Macedonian Corpus - Cleaned Version

This repository contributes to the creation of the **Macedonian Corpus**, which aims to address the scarcity of high-quality Macedonian text data in NLP. The cleaned dataset applies heuristic filters and deduplication to ensure the quality of the text (NOTE: you have to download the data yourself, the links can be found in the [HuggingFace repo](https://huggingface.co/datasets/LVSTCK/macedonian-corpus-raw) under Data Sources).

---

## How to Use

### macedonian-corpus-raw:

1. **Scrape Additional Data:**
   - Use `scrape_pdfs.py` to collect additional text data from PDFs. 
   - Use your own data, such as local PDFs, DOCX, PPTX, TXT, spreadsheets, audio and video files to enrich the dataset. 

2. **Use MMORE to extract textual data from the files**

3. **Process Additional Data:**
   - Modify and run `consolidate_data.ipynb` to unify all data sources.
   - Since `macedonian-corpus-raw` is already unified, you can just append the newly collected data to the JSONL. 

### macedonian-corpus-cleaned: 

1. **Download the Dataset:**
   - If you dont have it locally, run `download.py` to retrieve the raw dataset.

2. **Filter the Dataset:**
   - Execute `filter.py` to produce the cleaned version of the dataset. Optionally, use `split_data/` for multiprocessing if handling large files. NOTE: Significant computational resources might be needed for this step, depending on number of workers and tasks chosen.  
   - You can modify the filtering according to your own needs (e.g. swap sentence deduplication with min hash deduplication). For more information see [datatrove](https://github.com/huggingface/datatrove). 

### macedonian-corpus-cleaned-deduplicated:

1. **Run MinHash Deduplication**:
   - Use your cleaned version of the dataset (or download it from HuggingFace) and just run `minhash.py` to reproduce the deduplicated version of the dataset (MinHashConfig can be changed according to needs). 

## How to Contribute?
You can contribute to the Macedonian corpus by:

1. **Digitalize Books and Materials**:  
   - Contribute by digitalizing books, documents, and other materials that are legally in the public domain. These digitalized materials can be used to expand the datasets.  
   - Ensure that the materials you contribute comply with copyright laws and are explicitly permitted for public use.

2. **Expand Data Collection**:  
   - Share other forms of Macedonian-language text data, such as articles, essays, or transcripts, that can legally be used for training or evaluating language models.  

3. **Encourage Institutional Participation**:  
   - We hope this initiative inspires institutions in Macedonia, such as libraries, universities, and research centers, to take part in the digitalization of Macedonian-language materials.  
   - The availability of such materials will enable the development of specialized software tailored to the needs of Macedonian speakers and researchers.


## 📝 TODO:
- Add `requirements.txt`
