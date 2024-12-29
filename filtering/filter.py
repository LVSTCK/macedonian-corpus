#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.filters import LanguageFilter 
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
import spacy

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

        # We load spaCy once in `__init__` or lazily in `run()`.
        # However, note that in a multiprocessing environment,
        # you may want to re-load in each process if needed.
        self.nlp = spacy.load(self.spacy_model, disable=["ner", "parser"])
        # or keep parser if you do want parse-based sentence segmentation
        # just be mindful of speed

    def run(self, data: DocumentsPipeline, rank=0, world_size=1):
        for doc in data:
            text = doc.text
            # process with spaCy
            spacy_doc = self.nlp(text)

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
class NaiveMacedonianFilter(PipelineStep):
    """
    A minimal example: we keep docs if > 40% of chars are in the Cyrillic range.
    """
    def __init__(self, min_ratio=0.4):
        super().__init__()
        self.min_ratio = min_ratio

    def run(self, data: DocumentsPipeline, rank=0, world_size=1):
        for doc in data:
            text = doc.text
            cyr_chars = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
            ratio_cyr = cyr_chars / max(len(text), 1)
            if ratio_cyr >= self.min_ratio:
                yield doc


# -------------------------------------------------------------
#  C) CUSTOM DEDUP FILTER THAT PRESERVES fineweb-2
# -------------------------------------------------------------
class KeepFineWebMinhashDedupFilter(MinhashDedupFilter):
    """
    Extends MinhashDedupFilter so that if a cluster has
    a doc from 'fineweb-2', we keep that doc and remove the others.
    Otherwise, keep the first doc in the cluster.
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
def build_pipeline(input_path: str, output_path: str):
    """
    Builds a pipeline that:
     1) reads the JSONL from `input_path`
     2) chunk large docs
     3) naive Macedonian filter
     4) minhash signature
     5) bucket them
     6) cluster them
     7) custom dedup filter that keeps fineweb-2
     8) writes out final data
    """
    # configure the minhash
    minhash_conf = MinhashConfig(
        hash_config=HashConfig(hash_fc="sha1", precision=64),
        num_buckets=14,
        hashes_per_bucket=8,
        n_grams=5,
    )

    pipeline = [
        # 1) read from JSONL
        JsonlReader(
            data_folder=input_path,  # could be "macedonian_corpus_raw.jsonl"
            text_key="text",
            id_key=None,  # we’ll let DataTrove auto-generate or we can keep them from the file
            default_metadata={},  # might store e.g. { "source": "???" } if needed
            compression=None,
        ),

        # 2) chunk large docs
        ChunkerBlock(max_tokens=4096, sentence_overlap=0), # no need for overlap 

        # 3) naive Macedonian filter
        NaiveMacedonianFilter(min_ratio=0.5),

        # count tokens so we have stats pre-dedup
        TokensCounter(),

        # 4) minhash signature
        MinhashDedupSignature(output_folder=f"{output_path}/signatures", config=minhash_conf),
        # 5) bucket them
        MinhashDedupBuckets(input_folder=f"{output_path}/signatures", config=minhash_conf, output_folder=f"{output_path}/buckets"),
        # 6) cluster
        MinhashDedupCluster(
            input_folder=f"{output_path}/buckets",
            output_folder=f"{output_path}/remove_ids",
            config=minhash_conf,
        ),
        # 7) custom dedup filter that keeps fineweb-2 records
        KeepFineWebMinhashDedupFilter(input_folder=f"{output_path}/remove_ids"),

        # 8) write final
        JsonlWriter(
            output_folder=f"{output_path}/final",
            output_filename="${rank}.jsonl.gz",
        ),
    ]
    return pipeline


# -------------------------------------------------------------
#  F) MAIN
# -------------------------------------------------------------
def main():
    # point to your local input jsonl (or directory)
    input_path = "macedonian_corpus_raw.jsonl"
    # define where we'll store intermediate stuff and final
    output_path = "macedonian-cleaned-cleaned"

    pipeline = build_pipeline(input_path, output_path)

    # For a big dataset, you may set tasks=100 or more, one per shard
    # If your JSONL is a single file, tasks=1 might be simpler,
    # or you might split that JSONL into multiple shards first.
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=1,  # or more if you have multiple files
        workers=1,  # how many parallel CPU processes
        logging_dir=f"{output_path}/logs",
        skip_completed=True,
    )

    # This will run the entire pipeline locally
    executor.run()
    print(f"Done. Cleaned data is in: {output_path}/final/")

if __name__ == "__main__":
    main()
