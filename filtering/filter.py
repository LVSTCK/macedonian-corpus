""" To use this script it is first advisable to split the .gz files into smaller chunks. """


import re
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.pipeline.dedup import (
    MinhashDedupSignature,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.dedup.minhash import MinhashConfig
from datatrove.utils.hashing import HashConfig
from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.utils.lid import FT176LID, GlotLID, FastTextLID
from typing import Union
import spacy
from tqdm import tqdm 
import os 
import logging

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
# number of .gz files in the split_data folder for parallel processing
TOTAL_LINES = 100_000 # total number of lines in each .gz file

def get_minhash_config():
    return MinhashConfig(
        hash_config=HashConfig(hash_fc="sha1", precision=64),
        num_buckets=16,  # Ensure this matches your cluster setup
        hashes_per_bucket=8,
        n_grams=5,
    )


# -------------------------------------------------------------
#  A) CHUNKING BLOCK
# -------------------------------------------------------------
class ChunkerBlock(PipelineStep):
    """
    Uses spaCy to:
      1) segment the doc into sentences
      2) group consecutive sentences until we reach ~max_tokens
      3) yield a new chunk whenever we exceed that token budget
      4) optionally overlap chunk boundaries by `sentence_overlap`
         sentences, which can be helpful for contexts that
         rely on surrounding text
    """
    def __init__(
        self,
        spacy_model: str = "mk_core_news_lg",
        max_tokens: int = 4096,
        sentence_overlap: int = 0
    ):
        super().__init__()
        self.spacy_model = spacy_model
        self.max_tokens = max_tokens
        self.sentence_overlap = sentence_overlap
        self.nlp = spacy.load(self.spacy_model, disable=["ner", "parser"])
        self.nlp.add_pipe("sentencizer")
        self.nlp.max_length = 10_000_000 

    def run(self, data: DocumentsPipeline, rank=0, world_size=1):
        # Enable GPU; It has to be within each process in a multiprocessing environment, otherwise it tries to pickle objects that reference the GPU... tldr error is thrown
        spacy.prefer_gpu() 
        if spacy.require_gpu():
            print("GPU is enabled and ready.")
        else:
            print("Could not enable GPU. Check CUDA installation.")

        for doc in tqdm(data, desc="Chunking", total=TOTAL_LINES):
            try: 
                # if its not MMORE then skip 
                if doc.metadata.get("source") != "MMORE":
                    yield doc
                    continue

                text = doc.text
                # process with spaCy
                spacy_doc = self.nlp(text)
                # disable GPU
                spacy.require_cpu()
                # We'll store sentences as lists of tokens
                # so we can easily count them for the chunk budget
                sents_tokens = []
                for sent in spacy_doc.sents:
                    sent_tokens = [token.text for token in sent]
                    sents_tokens.append(sent_tokens)

                # chunk them up
                chunks = self._chunk_sents(
                    sents_tokens,
                    max_tokens=self.max_tokens,
                    sentence_overlap=self.sentence_overlap,
                )

                # yield each chunk as a new doc
                for i, chunk_tokens in enumerate(chunks):
                    chunk_text = " ".join(chunk_tokens).strip()
                    if chunk_text:
                        new_doc_id = f"{doc.id}_spacychunk_{i}"
                        new_doc = Document(
                            text=chunk_text,
                            id=new_doc_id,
                            metadata=dict(doc.metadata),  # copy original metadata
                        )
                        yield new_doc
            except Exception as e:
                logging.error(f"Error processing doc {doc.id}: {e}")

    def _chunk_sents(self, sents_tokens, max_tokens, sentence_overlap):
        """
        Group consecutive sentences until we exceed max_tokens,
        then yield a chunk. If sentence_overlap>0, then each
        new chunk will overlap the previous chunk by that many
        sentences.
        """
        chunks = []
        current_chunk = []
        current_len = 0

        for i, sent_toks in enumerate(sents_tokens):
            sent_len = len(sent_toks)
            if current_len + sent_len > max_tokens:
                # yield current chunk
                chunks.append([tok for toks in current_chunk for tok in toks])

                # If overlap is specified, copy the last `sentence_overlap` sents
                # to the next chunk. This can help preserve context across chunks.
                if sentence_overlap > 0:
                    overlap_slice = current_chunk[-sentence_overlap:]
                    current_chunk = overlap_slice[:]
                    current_len = sum(len(s) for s in current_chunk)
                else:
                    # start a fresh chunk
                    current_chunk = []
                    current_len = 0

            current_chunk.append(sent_toks)
            current_len += sent_len

        # leftover
        if current_chunk:
            chunks.append([tok for toks in current_chunk for tok in toks])

        return chunks


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
      - Additional: skip doc if "{" found, skip doc if line with "lorem ipsum"
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

        # split text into lines (like C4); should this be split lines or sentences? 
        lines = text.splitlines() 
        ## split on sentence if needed
        # lines = [line.text for line in self.nlp(text).sents] 
        
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

            # if self.filter_curly_bracket and "{" in line:
            #     return False, "curly_bracket"

            # 6) cookies / policy substrings (line-level remove)
            if any(p in line_lower for p in POLICY_SUBSTRINGS):
                self.stat_update("line-filter-policy")
                continue

            # 7) track bullet / ellipsis
            if line.startswith(BULLET_CHARS):
                bullet_starts += 1
            if line.endswith(ELLIPSIS):
                ellipsis_ends += 1

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

        ## Unused: I haven't tested this yet 
        ## alpha ratio: we can do a quick doc-level check for alpha ratio 
        # all_words = " ".join(kept_lines).split()
        # if all_words:
        #     n_words = len(all_words)
        #     n_alpha_words = 0
        #     for w in all_words:
        #         # if any char is alpha => counts
        #         if any(c.isalpha() for c in w):
        #             n_alpha_words += 1
        #     alpha_ratio = n_alpha_words / n_words
        #     if alpha_ratio < self.min_alpha_word_ratio:
        #         return False, f"below_alpha_ratio({alpha_ratio:.2f})"

        # Re-assemble the text from kept lines
        doc.text = "\n".join(kept_lines)
        return True


# -------------------------------------------------------------
#  C) CUSTOM DEDUP FILTER THAT PRESERVES fineweb-2
# -------------------------------------------------------------
class KeepFineWebMinhashDedupFilter(MinhashDedupFilter):
    """
    Extends MinhashDedupFilter so that if a cluster has
    a doc from 'fineweb-2', we keep that doc and remove the others.
    Otherwise, keep the first doc in the cluster.
    
    NOTE: We do this explicity after discussing with creators of fineweb-2 where they 
    informed us that fineweb is the most reliable source; hence, we want to keep it.
    By most reliable: they have trained multiple models on HPLT vs fineweb-2 and found 
    that training only on fineweb-2 yields the best results. HPLT and fineweb are both built from
    CommonCrawl, so, we assume that there will be duplicates between them.
    """

    def _should_remove(self, doc, cluster_id):
        # cluster docs
        if cluster_id not in self.clusters_by_id:
            return False  # no known cluster, keep

        cluster_doc_ids = self.clusters_by_id[cluster_id]

        # do we have a fineweb-2 doc?
        fineweb_ids = []
        for doc_id in cluster_doc_ids:
            info = self.hash_info_by_doc.get(doc_id, {})
            if info.get("source") == "fineweb-2":
                fineweb_ids.append(doc_id)

        if len(fineweb_ids) > 0:
            # if there's at least one fineweb doc, keep it,
            # remove everything else
            if doc.metadata.get("source") == "fineweb-2":
                return False  # keep
            else:
                return True  # remove
        else:
            # fallback to original minhash dedup logic:
            # keep doc with smallest doc_id, remove others
            first_doc_id = min(cluster_doc_ids)
            return doc.id != first_doc_id

# -------------------------------------------------------------
#  E) BUILD THE PIPELINE
# -------------------------------------------------------------
"""
Builds a pipeline that:
    Stage 1)  
        reads the JSONL from `input_path`
        chunk large docs
        custom filter

    Stage 2) minhash signature
    Stage 3) bucket
    Stage 4) cluster
    Stage 5) custom dedup filter that aims to keep fineweb-2 in case of duplicates
    
    Stage 6) 
        PII formatter
        writes out final data

    The output from Stage 1 is the input to Stage 2, and so on.
"""

def build_stage1_pipeline(input_path: str, stage1_output: str):
    pipeline = [
        JsonlReader(
            data_folder=input_path,
            text_key="text",
            id_key=None,
            compression="gzip",
            glob_pattern="*.gz",
        ),
        TokensCounter(),
        ChunkerBlock(max_tokens=4096, sentence_overlap=0),
        CustomMacedonianFilter(
            remove_citations=True,
            filter_no_terminal_punct=False,
            min_words_per_line=3,
            max_word_length=1000,
            filter_lorem_ipsum=True,
            filter_javascript=True,
            filter_curly_bracket=False,
            min_alpha_word_ratio=0.80,
            bullet_start_ratio_threshold=0.90,
            ellipsis_end_ratio_threshold=0.30,
            do_language_check=True,
            languages=["mk"],
        ),
        TokensCounter(),
        JsonlWriter(output_folder=stage1_output, output_filename="stage1_output.jsonl.gz"),
    ]
    return pipeline


def build_stage2_pipeline(stage1_output: str, signatures_output: str, minhash_conf: MinhashConfig):
    pipeline = [
        JsonlReader(
            data_folder=stage1_output,
            text_key="text",
            id_key=None,
            compression="gzip",
            glob_pattern="*.gz",
        ),
        MinhashDedupSignature(output_folder=signatures_output, config=minhash_conf),
    ]
    return pipeline


def build_stage3_pipeline(signatures_output: str, buckets_output: str, minhash_conf: MinhashConfig):
    pipeline = [
        MinhashDedupBuckets(
            input_folder=signatures_output,
            config=minhash_conf,
            output_folder=buckets_output,
        ),
    ]
    return pipeline

def build_stage4_pipeline(buckets_output: str, clusters_output: str, minhash_conf: MinhashConfig):
    pipeline = [
        MinhashDedupCluster(
            input_folder=buckets_output,
            output_folder=clusters_output,
            config=minhash_conf,
        ),
    ]
    return pipeline

def build_stage5_pipeline(clusters_output: str, dedup_output: str):
    pipeline = [
        KeepFineWebMinhashDedupFilter(input_folder=clusters_output),
        JsonlWriter(output_folder=dedup_output, output_filename="deduped_output.jsonl.gz"),
    ]
    return pipeline

def build_stage6_pipeline(dedup_output: str, final_output: str):
    pipeline = [
        JsonlReader(
            data_folder=dedup_output,
            text_key="text",
            id_key=None,
            compression="gzip",
            glob_pattern="*.gz",
        ),
        TokensCounter(),
        PIIFormatter(),
        JsonlWriter(output_folder=final_output, output_filename="final.jsonl.gz"),
    ]
    return pipeline

# -------------------------------------------------------------
#  F) MAIN
# -------------------------------------------------------------
def main():
    input_path = "split_data/"
    output_base = "macedonian-corpus-cleaned"
    
    # Define intermediate folders
    stage1_output = os.path.join(output_base, "stage1")
    signatures_output = os.path.join(output_base, "signatures")
    buckets_output = os.path.join(output_base, "buckets")
    clusters_output = os.path.join(output_base, "clusters")
    dedup_output = os.path.join(output_base, "deduped")
    final_output = os.path.join(output_base, "final")
    
    # Ensure output directories exist
    os.makedirs(stage1_output, exist_ok=True)
    os.makedirs(signatures_output, exist_ok=True)
    os.makedirs(buckets_output, exist_ok=True)
    os.makedirs(clusters_output, exist_ok=True)
    os.makedirs(dedup_output, exist_ok=True)
    os.makedirs(final_output, exist_ok=True)
    
    minhash_conf = get_minhash_config()
    
    # Stage 1
    logging.info("Starting Stage 1: Data Reading and Initial Processing")
    stage1_pipeline = build_stage1_pipeline(input_path, stage1_output)
    executor1 = LocalPipelineExecutor(
        pipeline=stage1_pipeline,
        tasks=len(os.listdir(input_path)),  
        workers=50, 
        logging_dir=os.path.join(output_base, "logs", "stage1"),
    )
    executor1.run()
    logging.info("Stage 1 completed.")
    
    # Stage 2
    logging.info("Starting Stage 2: Minhash Signature Generation")
    stage2_pipeline = build_stage2_pipeline(stage1_output, signatures_output, minhash_conf)
    executor2 = LocalPipelineExecutor(
        pipeline=stage2_pipeline,
        tasks=16,  # Number of buckets
        workers=50,
        logging_dir=os.path.join(output_base, "logs", "stage2"),
    )
    executor2.run()
    logging.info("Stage 2 completed.")
    
    # Stage 3
    logging.info("Starting Stage 3: Bucketing")
    stage3_pipeline = build_stage3_pipeline(signatures_output, buckets_output, minhash_conf)
    executor3 = LocalPipelineExecutor(
        pipeline=stage3_pipeline,
        tasks=16,  # Number of buckets
        workers=50,
        logging_dir=os.path.join(output_base, "logs", "stage3"),
    )
    executor3.run()
    logging.info("Stage 3 completed.")
    
    # Stage 4
    logging.info("Starting Stage 4: Clustering")
    stage4_pipeline = build_stage4_pipeline(buckets_output, clusters_output, minhash_conf)
    executor4 = LocalPipelineExecutor(
        pipeline=stage4_pipeline,
        tasks=1,  
        workers=1,
        logging_dir=os.path.join(output_base, "logs", "stage4"),
    )
    executor4.run()
    logging.info("Stage 4 completed.")
    
    # Stage 5
    logging.info("Starting Stage 5: Deduplication Filtering")
    stage5_pipeline = build_stage5_pipeline(clusters_output, dedup_output)
    executor5 = LocalPipelineExecutor(
        pipeline=stage5_pipeline,
        tasks=1, 
        workers=1,
        logging_dir=os.path.join(output_base, "logs", "stage5"),
    )
    executor5.run()
    logging.info("Stage 5 completed.")
    
    # Stage 6
    logging.info("Starting Stage 6: Post-Processing")
    stage6_pipeline = build_stage6_pipeline(dedup_output, final_output)
    executor6 = LocalPipelineExecutor(
        pipeline=stage6_pipeline,
        tasks=len(os.listdir(dedup_output)), 
        workers=50,
        logging_dir=os.path.join(output_base, "logs", "stage6"),
    )
    executor6.run()
    logging.info("Stage 6 completed.")
    
    print(f"Done. Cleaned data is in: {final_output}/")

if __name__ == "__main__":
    main()
