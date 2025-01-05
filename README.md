# Macedonian Corpus 🇲🇰

This repository contains scripts, notebooks, and documentation to process, clean, and manage the [Macedonian Corpus Raw](https://huggingface.co/datasets/LVSTCK/macedonian-corpus-raw). 

---

## 🌟 Key Highlights
- 📚 **First consolidated Macedonian Corpus** for NLP research.
- 📊 Includes **3 versions** of the corpus:
  - [**Raw**](https://huggingface.co/datasets/LVSTCK/macedonian-corpus-raw): 37.6 GB, 3.53 billion words.
  - [**Cleaned**](https://huggingface.co/datasets/LVSTCK/macedonian-corpus-cleaned): 35.5 GB, 3.31 billion words (filtered for quality).
  - [**Cleaned + Deduplicated**](https://huggingface.co/datasets/LVSTCK/macedonian-corpus-cleaned-dedup): 16.78 GB, 1.47 billion words (high-quality, minimal redundancy).
- 🚀 Enables pretraining/fine-tuning LLMs, machine translation, and linguistic analysis.
- 🛠️ Built with state-of-the-art filtering and deduplication techniques.

#### **Raw**
| Origin                | Size (GB) | Words (B) | Percentage |
|-----------------------|-----------|-----------|------------|
| HPLT                 | 15.85     | 1.49      | 42.21%     |
| HuggingFace (fineweb-2) | 14.21     | 1.33      | 37.66%     |
| CLARIN (MaCoCu-mk 2.0) | 5.20      | 0.49      | 13.92%     |
| Wikipedia            | 0.78      | 0.07      | 1.96%      |
| Other (MMORE)        | 1.48      | 0.14      | 4.07%      |
| Common Voice         | 0.02      | 0.0018    | 0.05%      |
| SETimes Corpus       | 0.06      | 0.0044    | 0.13%      |
| **Total**            | **37.60** | **3.53**  | **100.00%** |

#### **Cleaned**
| Origin                | Size (GB) | Words (B) | Percentage |
|-----------------------|-----------|-----------|------------|
| HPLT                 | 15.51     | 1.45      | 43.72%     |
| HuggingFace (fineweb-2) | 14.13     | 1.31      | 39.62%     |
| CLARIN (MaCoCu-mk 2.0) | 5.14      | 0.48      | 14.57%     |
| Wikipedia            | 0.64      | 0.06      | 1.78%      |
| Other (MMORE)        | 0.04      | 0.004     | 0.12%      |
| Common Voice         | 0.02      | 0.002     | 0.05%      |
| SETimes Corpus       | 0.06      | 0.004     | 0.13%      |
| **Total**            | **35.54** | **3.31**  | **100.00%** |

#### **Cleaned + Deduplicated**
| Origin                | Size (GB) | Words (B) | Percentage |
|-----------------------|-----------|-----------|------------|
| HuggingFace (fineweb-2) | 7.85      | 0.73      | 49.55%     |
| HPLT                 | 5.80      | 0.54      | 36.87%     |
| CLARIN (MaCoCu-mk 2.0) | 1.94      | 0.18      | 12.39%     |
| Wikipedia            | 0.13      | 0.01      | 0.83%      |
| Other (MMORE)        | 0.04      | 0.004     | 0.25%      |
| Common Voice         | 0.02      | 0.002     | 0.12%      |
| **Total**            | **16.78** | **1.47**  | **100.00%** |

## 📚 Dataset Sources
The corpus is built by collecting and processing data from the following sources:

| **Source**                           | **Notes**                                                       | **Origin**                                                              |
|--------------------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------------|
| UKIM                                 | Books and dissertations from various topics                    | [UKIM Digital Library](https://ukim.edu.mk/en/nauka/infrastruktura/digitalna-biblioteka/), [UKIM Repository](https://repository.ukim.mk/) |
| Wikipedia (MK)                            | Macedonian Wikipedia dump                                       | [Wikipedia](https://mk.wikipedia.org)                                   |
| MANU                                 | Various publications from MANU                                  | [MANU](https://manu.edu.mk/)                                            |
| Institute for Macedonian Literature                        | E-library of the Institute for Macedonian Literature            | [E-biblioteka](https://e-biblioteka.mk/)                                |
| HuggingFace (fineweb-2)                            | Macedonian subset of FineWeb-2 (mkd_Cyrl)                       | [Hugging Face](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) |
| Common Voice (MK)                        | Macedonian sentences from the Common Voice dataset              | [Common Voice](https://commonvoice.mozilla.org/en)                      |
| CLARIN MaCoCu-mk 2.0                  | Web-crawled Macedonian texts                                    | [CLARIN](https://www.clarin.si/repository/xmlui/handle/11356/1801)      |
| Pollitecon (~90 books)               | Free eBooks on Macedonian themes by various publishers          | [Pollitecon](https://www.pollitecon.com/)                               |
| UKLO                                 | Resources from UKLO                                             | [UKLO](https://uklo.edu.mk/?lang=en)                                    |
| UGD                                  | Resources from UGD                                              | [UGD](https://www.ugd.edu.mk/en/home/)                                  |
| SETimes Corpus (MK-EN)               | Macedonian-English parallel corpus (only MK sentences used)     | [SETimes](https://github.com/stefan-it/nmt-en-mk?tab=readme-ov-file)    |
| HPLT (MK)                            | Macedonian subset of HPLT                                       | [HPLT](https://hplt-project.org/datasets/v2.0)                          |
| Institute of Macedonian Language     | Resources from the Institute of Macedonian Language "Krste Misirkov" | [IMJ](http://imj.ukim.edu.mk/)                                          |
| Official PE Gazette of North Macedonia | Official Gazette of North Macedonia                             | [slvesnik](https://www.slvesnik.com.mk/besplaten-pristap-do-izdanija.nspx) |

## 📋 Overview

### **1. `filtering/`**
This folder contains the primary scripts for downloading, filtering, and preparing the clean version  of corpus input: [raw corpus](https://huggingface.co/datasets/LVSTCK/macedonian-corpus-raw), output: [cleaned corpus](https://huggingface.co/datasets/LVSTCK/macedonian-corpus-cleaned). 

- **🧹 `filter.py`**
  - **Purpose:** Produces a cleaned version of the dataset (filtering process inspired by [fineweb-2](https://github.com/huggingface/fineweb-2)).
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

## How to Reproduce

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

## 🤝 How to Contribute?
You can contribute to the Macedonian corpus by:

1. **Digitalize Books and Materials**:  
   - Contribute by digitalizing books, documents, and other materials that are legally in the public domain. These digitalized materials can be used to expand the datasets.  
   - Ensure that the materials you contribute comply with copyright laws and are explicitly permitted for public use.

2. **Expand Data Collection**:  
   - Share other forms of Macedonian-language text data, such as articles, essays, or transcripts, that can legally be used for training or evaluating language models.  

3. **Encourage Institutional Participation**:  
   - We hope this initiative inspires institutions in Macedonia, such as libraries, universities, and research centers, to take part in the digitalization of Macedonian-language materials.  
   - The availability of such materials will enable the development of specialized software tailored to the needs of Macedonian speakers and researchers.
