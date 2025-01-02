""" To use this script it is first advisable to split the .jsonl file into smaller chunks for multiprocessing. A lot of memory might be needed to run this script. """

import re
import os 
import nltk

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.dedup import SentenceDedupFilter, SentenceDedupSignature, SentenceFindDedups
from datatrove.pipeline.dedup.sentence_dedup import SentDedupConfig
from datatrove.executor.base import PipelineExecutor
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.utils.lid import FT176LID, GlotLID, FastTextLID
from nltk.tokenize import sent_tokenize
from typing import Union

nltk.download('punkt')
nltk.download('punkt_tab')

CITATION_REGEX = re.compile(r"\[\d*]|\[edit]|\[citation needed]")
END_PUNCTUATION = (".", "?", "!", '"', "'")
ELLIPSIS = "..."
POLICY_SUBSTRINGS = [
    "terms of use",
    "privacy policy",
    "cookie policy",
    "uses cookies",
    "use of cookies",
    "use cookies",
]
BULLET_CHARS = ("-", "•", "*", "‣", "·") 
FINDER_WORKERS = 10  # this will speed up/parallelize step 2

# -------------------------------------------------------------
#  B) MACEDONIAN LANGUAGE FILTER
#   
# -------------------------------------------------------------
class CustomMacedonianFilter(BaseFilter):
    """
    A custom filter merging:
      - C4 rules: remove lines with "javascript", doc if "lorem ipsum" or curly brace,
        skip lines that don't end with terminal punct, skip lines < 3 words, remove lines with any word >1000 chars,
        skip lines containing policy keywords, etc.
      - Additional: skip doc if "{" found, skip doc if line with "lorem ipsum"; Not used in our pipeline.
      - Gopher-like rules: 
         * check alpha ratio: if fewer than 80% of words contain an alphabetic char, drop doc
         * bullet ratio: if >90% lines start with a bullet, drop doc
         * ellipsis ratio: if >30% lines end with "..."  => drop doc
      - Optionally, a language check: skip doc if fastText LID is too low
    """

    name = "Custom Macedonian Filter"
    def __init__(
        self,
        exclusion_writer=None,
        # C4-like settings
        remove_citations: bool = True,
        filter_no_terminal_punct: bool = True,
        min_words_per_line: int = 3,
        max_word_length: int = 1000,
        filter_lorem_ipsum: bool = True,
        filter_javascript: bool = True,
        filter_curly_bracket: bool = True,
        # Gopher-like settings
        min_alpha_word_ratio: float = 0.80,  # 80% of words must have at least 1 alpha 💪
        bullet_start_ratio_threshold: float = 0.90,  # 90% lines start with bullet => drop doc
        ellipsis_end_ratio_threshold: float = 0.30,  # 30% lines end with "..." => drop doc
        # language check
        do_language_check: bool = False,
        language_threshold: float = 0.65,
        languages: list[str] = None,  # e.g. ["mk"] for Macedonian
    ):
        super().__init__(exclusion_writer)
        self.remove_citations = remove_citations
        self.filter_no_terminal_punct = filter_no_terminal_punct
        self.min_words_per_line = min_words_per_line
        self.max_word_length = max_word_length
        self.filter_lorem_ipsum = filter_lorem_ipsum
        self.filter_javascript = filter_javascript
        self.filter_curly_bracket = filter_curly_bracket

        self.min_alpha_word_ratio = min_alpha_word_ratio
        self.bullet_start_ratio_threshold = bullet_start_ratio_threshold
        self.ellipsis_end_ratio_threshold = ellipsis_end_ratio_threshold

        self.do_language_check = do_language_check
        self.language_threshold = language_threshold
        self.languages = languages
        
        self.lid_model = FT176LID(languages = languages) # or FastTextLID() or FT176LID(); If you use GlotLID, you need to specify the language as ['mkd_Cyrl']
        # FT176LID returns e.g.: (('mk', 0.8015426397323608), {'mk': 0.8015426397323608}). You can play arround with test_language_model.py to see the output of each model

    def filter(self, doc: Document) -> Union[bool, tuple[bool, str]]:
        text = doc.text
        
        # ---- Language check ----
        if self.do_language_check and self.lid_model is not None:
            best_lang_pair, lang_pairs = self.lid_model.predict(doc)
            lang, lang_score = best_lang_pair
            doc.metadata["language"] = lang
            doc.metadata["language_score"] = lang_score
            
            if lang_score < self.language_threshold:
                return False, f"lang_score<{self.language_threshold}"

        ## split text into lines (like C4); should this be split lines or sentences? 
        # lines = text.splitlines() 
        ## split on sentence
        lines = sent_tokenize(text)
        
        kept_lines = []
        # counters for doc-level checks
        total_lines = 0
        bullet_starts = 0
        ellipsis_ends = 0

        for line in lines:
            total_lines += 1
            line = line.strip()

            if not line:
                self.stat_update("line-empty")
                continue

            # 1) any word too long?
            if self.max_word_length != -1:
                words = line.split()
                if any(len(word) > self.max_word_length for word in words):
                    self.stat_update("line-filter-too_long_word")
                    continue

            # 2) remove citations
            if self.remove_citations:
                line = CITATION_REGEX.sub("", line)

            # 3) check end punctuation
            if self.filter_no_terminal_punct and not line.endswith(END_PUNCTUATION):
                # lets allow "..." if we want it considered terminal?
                if not line.endswith(ELLIPSIS):
                    self.stat_update("line-filter-no_terminal_punc")
                    continue

            # 4) min words; if line has < 3 words, skip 
            words = line.split()
            if len(words) < self.min_words_per_line:
                self.stat_update("line-filter-too_few_words")
                continue

            # 5) doc-level filters if present in line
            line_lower = line.lower()
            if self.filter_lorem_ipsum and "lorem ipsum" in line_lower:
                return False, "lorem_ipsum"

            if self.filter_javascript and "javascript" in line_lower:
                self.stat_update("line-filter-javascript")
                continue

            if self.filter_curly_bracket and "{" in line:
                return False, "curly_bracket"

            # 6) cookies / policy substrings (line-level remove)
            if any(p in line_lower for p in POLICY_SUBSTRINGS):
                self.stat_update("line-filter-policy")
                continue

            # 7) track bullet / ellipsis
            if line.startswith(BULLET_CHARS):
                bullet_starts += 1
            if line.endswith(ELLIPSIS):
                ellipsis_ends += 1
            
            # 8) Because of the consolidation in data, especially important for MMORE 
            # check if doc is mmore 
            if doc.metadata.get("source") == "MMORE":
                sh_count = line.count("Ш")
                gj_count = line.count("Ѓ")
                kj_count = line.count("Ќ")
                if sh_count > 5 or gj_count > 5 or kj_count > 5:
                    self.stat_update("line-filter-sh-gj")
                    continue
                elif sh_count + gj_count + kj_count > 5:
                    self.stat_update("line-filter-sh-gj")
                    continue
            
            if "њњњ.себ.ее" in line:
                self.stat_update("line-filter-њњњ.себ.ее")
                continue


            # keep line
            kept_lines.append(line)
            self.stat_update("line-kept")

        # after we gather all lines, if none left, remove doc
        if not kept_lines:
            return False, "all_lines_filtered"

        # ---- Gopher-like doc-level checks ----
        # bullet ratio
        if total_lines > 0:
            bullet_ratio = bullet_starts / total_lines
            if bullet_ratio > self.bullet_start_ratio_threshold:
                return False, f"too_many_bullets({bullet_ratio:.2f})"

            ellipsis_ratio = ellipsis_ends / total_lines
            if ellipsis_ratio > self.ellipsis_end_ratio_threshold:
                return False, f"too_many_ellipsis({ellipsis_ratio:.2f})"

        ## TODO: I haven't tested this yet 
        # alpha ratio: we can do a quick doc-level check for alpha ratio 
        all_words = " ".join(kept_lines).split()
        if all_words:
            n_words = len(all_words)
            n_alpha_words = 0
            for w in all_words:
                # if any char is alpha => counts
                if any(c.isalpha() for c in w):
                    n_alpha_words += 1
            alpha_ratio = n_alpha_words / n_words
            if alpha_ratio < self.min_alpha_word_ratio:
                return False, f"below_alpha_ratio({alpha_ratio:.2f})"

        # Re-assemble the text from kept lines
        doc.text = "\n".join(kept_lines)
        return True


# -------------------------------------------------------------
#  E) BUILD THE PIPELINE

# -------------------------------------------------------------
"""
Builds a pipeline that:
    Stage 1)  
        reads the JSONL from `input_path`
        chunk large docs
        custom filter based on C4 and Gopher rules

    Stage 2)
        deduplication signature generation
    
    Stage 3)
        deduplication signature filtering
        write output 

    We are using sentence deduplication. Reference implementation: https://github.com/huggingface/datatrove/blob/main/examples/sentence_deduplication.py    
"""


# -------------------------------------------------------------
#  F) MAIN

# -------------------------------------------------------------
def main():
    input_path = "split_data/"
    output_base = "macedonian-corpus-cleaned"
    
    # Define intermediate folders
    stage1_output = os.path.join(output_base, "stage1")
    signatures_output = os.path.join(output_base, "signatures")
    dedup_output = os.path.join(output_base, "deduped")
    final_output = os.path.join(output_base, "final")
    
    # Ensure output directories exist
    os.makedirs(stage1_output, exist_ok=True)
    os.makedirs(signatures_output, exist_ok=True)
    os.makedirs(dedup_output, exist_ok=True)
    os.makedirs(final_output, exist_ok=True)
    
    sent_dedup_config = SentDedupConfig(
        n_sentences=3,
        split_sentences=True,  # set to False to split on \n instead
        only_dedup_in_index=True,
        min_doc_words=50,
    )

    pipeline_1 = [
        JsonlReader(
            data_folder=input_path,
            text_key="text",
            id_key=None,
            glob_pattern="*.jsonl",
            # limit=1000, # for debugging 
        ),
        TokensCounter(),
        CustomMacedonianFilter(
            remove_citations=True,
            filter_no_terminal_punct=False,
            min_words_per_line=3,
            max_word_length=1000,
            filter_lorem_ipsum=True,
            filter_javascript=True,
            filter_curly_bracket=True,
            min_alpha_word_ratio=0.80,
            bullet_start_ratio_threshold=0.90,
            ellipsis_end_ratio_threshold=0.30,
            do_language_check=True,
            languages=["mk"],
        ),
        TokensCounter(),
        JsonlWriter(output_folder=stage1_output),
        SentenceDedupSignature(output_folder=signatures_output, config=sent_dedup_config, finder_workers=FINDER_WORKERS),
    ]

    pipeline_2 = [SentenceFindDedups(data_folder=signatures_output, output_folder=dedup_output, config=sent_dedup_config)]

    pipeline_3 = [
        JsonlReader(
            data_folder=stage1_output,
            text_key="text",
            id_key=None,
            glob_pattern="*.gz",
        ),
        SentenceDedupFilter(data_folder=dedup_output, config=sent_dedup_config),
        PIIFormatter(),
        JsonlWriter(output_folder=final_output),
    ]

    try:
        print("Running Pipeline 1")
        executor_1: PipelineExecutor = LocalPipelineExecutor(
            pipeline=pipeline_1, workers=15, tasks=len(os.listdir(input_path))
        )
        print(executor_1.run())
    except MemoryError as e:
        print("OOM Error during Pipeline 1. Task failed:", str(e))
        # Optionally, retry with lower workers
        executor_1: PipelineExecutor = LocalPipelineExecutor(
            pipeline=pipeline_1, workers=5, tasks=len(os.listdir(input_path))
        )
        print("Retrying Pipeline 1 with reduced workers...")
        print(executor_1.run())
    except Exception as e:
        print("Error during Pipeline 1:", str(e))

    try:
        print("Running Pipeline 2")
        executor_2: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_2, workers=1)
        print(executor_2.run())
    except MemoryError as e:
        print("OOM Error during Pipeline 2. Task failed:", str(e))
    except Exception as e:
        print("Error during Pipeline 2:", str(e))

    try:
        print("Running Pipeline 3")
        executor_3: PipelineExecutor = LocalPipelineExecutor(
            pipeline=pipeline_3, workers=15, tasks=len(os.listdir(stage1_output))
        )
        print(executor_3.run())
    except MemoryError as e:
        print("OOM Error during Pipeline 3. Task failed:", str(e))
    except Exception as e:
        print("Error during Pipeline 3:", str(e))
    
    print(f"Done. Cleaned data is in: {final_output}/")

if __name__ == "__main__":
    main()
