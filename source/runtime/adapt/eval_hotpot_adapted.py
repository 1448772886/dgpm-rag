"""Adapted HotpotQA evaluation for the ReMindRAG reproduction (GLM backbone).

Based on eval/eval_Hotpot.py (official, untouched). Differences:
1. MeteredOpenaiAgent everywhere, incl. rewrite/judge agents (paper uses
   GPT-4o-mini backbones + GPT-4o judge; here GLM-5.3-Flash plays both,
   REMIND_BACKBONE / REMIND_JUDGE_MODEL env override).
2. GpuEmbedding (CUDA + memoization) instead of the CPU-only HgEmbedding.
3. Per-question LLM token usage recorded into outputs (paper's cost metric).
4. --do_update flag (official hardcodes False = "No Memorization" row; the
   Same-Query memorization protocol reuses the SAME test_name dir, which is
   the KG cache).
5. Refactored into build_components() + run_single() so a persistent-process
   runner (adapt/run_hotpot_inproc.py) can load the heavy models once and
   evaluate many questions in one process.

Run from repo root:  python adapt/eval_hotpot_adapted.py --title_index 0 ...
"""

import os

# MUST be set before transformers/huggingface_hub are imported anywhere in the
# process: model + remote code are fully cached locally, and every hub HEAD
# call otherwise burns minutes in serial 10s timeouts against blocked
# huggingface.co.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "adapt"))

import json
from datetime import datetime
from transformers import AutoTokenizer

from ReMindRag.chunking import NaiveChunker
from ReMindRag import ReMindRag
from adapt.gpu_embedding import GpuEmbedding, resolve_local_model
from adapt.metered_llm import MeteredOpenaiAgent

dataset_name = "hotpot_dev_distractor_v1.json"

RESPONSE_FORMAT = """
Origin Query: {query}
LLM Output: {output}

Reference Chunks: {chunks}
Reference Edges: {edges}

"""

ANS_CHECK_PROMPT = """
Given one question, there is a groundtruth and a predict answer.
Please decide whether they are the same or not in semantic.
Please only output True or False.
Question: {question}
groundtruth = {reference_answer}
predicted answer = {generated_output}

Only output one word(True or False), without any additional content.
"""

ANS_REWRITE_PROMPT = """
Please extract the answer part from the given input. The given input may contain irrelevant content. Please ignore this irrelevant information and only output the answer to the question within the input itself. Ensure that the answer content is exactly the same as the answer in the original text, without any modifications, additions, or deletions.

Original text: {og_output}
Question: {question}
Answer:
"""


def build_components(backbone: str = None, judge_model: str = None, build_backbone: str = None,
                     api_key_file: str = None, build_disable_thinking: bool = None,
                     traverse_disable_thinking: bool = None) -> dict:
    """Build all heavy, reusable objects once (agents, embedding, tokenizer, data).

    build_backbone: model for KG CONSTRUCTION (chunk_agent: entity/relation/title
    extraction) — may differ from the traversal/answer backbone, e.g. a stronger
    model builds the graph once and a cheaper one traverses many times.
    api_key_file: provider-specific key file (default adapt/api_key.json).
    build/traverse_disable_thinking: per-side thinking override (None = global env).
    """
    backbone = backbone or os.environ.get("REMIND_BACKBONE", "GLM-5.3-Flash")
    judge_model = judge_model or os.environ.get("REMIND_JUDGE_MODEL", backbone)
    build_backbone = build_backbone or os.environ.get("REMIND_BUILD_BACKBONE", backbone)
    key_file = api_key_file or os.environ.get("REMIND_API_KEY_FILE",
                                              os.path.join(ROOT, "adapt", "api_key.json"))

    with open(key_file, "r", encoding="utf-8") as f:
        api_data = json.load(f)
    api_entry = api_data[0] if isinstance(api_data, list) else api_data
    base_url, api_key = api_entry["base_url"], api_entry["api_key"]
    model_cache = os.path.join(ROOT, "model_cache")
    # Resolve the tokenizer/chunker to the same cached model directory used by
    # GpuEmbedding. Passing the Hub ID here triggers remote HEAD requests even
    # in the offline graph-build workflow, despite a complete local bundle.
    embedding_model_path = resolve_local_model(
        "nomic-ai/nomic-embed-text-v2-moe", model_cache)

    comp = {
        "backbone": backbone,
        "judge_model": judge_model,
        "build_backbone": build_backbone,
        "chunk_agent": MeteredOpenaiAgent(base_url, api_key, build_backbone,
                                          disable_thinking=build_disable_thinking),  # build model
        "kg_agent": MeteredOpenaiAgent(base_url, api_key, backbone,
                                       disable_thinking=traverse_disable_thinking),  # traversal model
        "generate_agent": MeteredOpenaiAgent(base_url, api_key, backbone),
        "ans_rewrite_agent": MeteredOpenaiAgent(base_url, api_key, judge_model),
        "ans_check_agent": MeteredOpenaiAgent(base_url, api_key, judge_model),
        "embedding": GpuEmbedding(
            "nomic-ai/nomic-embed-text-v2-moe",
            model_cache,
            device=os.environ.get("REMIND_EMBEDDING_DEVICE", "cuda"),
        ),
        "chunker": NaiveChunker(embedding_model_path, model_cache, max_token_length=750, context_sentence=0),
        "tokenizer": AutoTokenizer.from_pretrained(embedding_model_path, cache_dir=model_cache,
                                                   local_files_only=True, trust_remote_code=True),
    }

    cache_dir = os.path.join(ROOT, "eval", "dataset_cache")
    with open(os.path.join(cache_dir, "Hotpot", dataset_name), "r", encoding="utf-8") as f:
        comp["all_data"] = json.load(f)
    comp["similar_data"] = None
    comp["different_data"] = None
    return comp


def load_rewrite_data(comp, query_type: str):
    """Lazy-load rewrite files (produced by adapt/rewrite_questions.py)."""
    if query_type == "similar" and comp["similar_data"] is None:
        pth = os.path.join(ROOT, "eval", "dataset_cache", "Hotpot", "hotpot_dev_distractor_similar.json")
        with open(pth, "r", encoding="utf-8") as f:
            comp["similar_data"] = json.load(f)
    if query_type == "different" and comp["different_data"] is None:
        pth = os.path.join(ROOT, "eval", "dataset_cache", "Hotpot", "hotpot_dev_distractor_different.json")
        with open(pth, "r", encoding="utf-8") as f:
            comp["different_data"] = json.load(f)


def run_single(comp: dict, title_index: int, test_name: str, query_type: str = "origin",
               do_update: bool = False, max_jumps: int = 10,
               override_query: str = None, override_answer: str = None) -> dict:
    """Evaluate ONE question (independent KG per case, official protocol).

    override_query/override_answer: explicit query/gold bypassing dataset
    lookup — used for similar-phase fill-ins on LooGLE nightly KGs. When
    override_query is given, title_index selects the KG and evidence context.
    """
    all_data = comp["all_data"]
    if override_query is None:
        load_rewrite_data(comp, query_type)

        if query_type == "origin":
            query, ans = all_data[title_index]["question"], all_data[title_index]["answer"]
        elif query_type == "similar":
            query, ans = comp["similar_data"][title_index]["question"], comp["similar_data"][title_index]["answer"]
        else:
            query, ans = comp["different_data"][title_index]["question"], comp["different_data"][title_index]["answer"]
    else:
        query = override_query
        ans = override_answer if override_answer is not None else ""

    # LooGLE longdep/shortdep mode: single-document context comes from the
    # LooGLE dataset (one flat string), not HotpotQA's paragraph list.
    # Context arrives via a temp FILE (env var would exceed Windows' 32K limit
    # for long documents).
    loogle_ctx_file = os.environ.get("REMIND_LOOGLE_CONTEXT_FILE", "")
    loogle_title = os.environ.get("REMIND_LOOGLE_TITLE", "")

    evidence = all_data[title_index]["supporting_facts"]
    context = all_data[title_index]["context"]

    db_dir = os.path.join("eval", "database", test_name, str(title_index))  # relative: chroma hnsw breaks on non-ASCII absolute paths
    need_load_data = os.path.exists(db_dir)
    os.makedirs(db_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(db_dir, f"log_{timestamp}.log")

    rag = ReMindRag(
        logger_level=10,
        log_path=log_path,
        chunk_agent=comp["chunk_agent"],
        kg_agent=comp["kg_agent"],
        generate_agent=comp["generate_agent"],
        embedding=comp["embedding"],
        chunker=comp["chunker"],
        tokenizer=comp["tokenizer"],
        database_description="Database title: wiki",
        save_dir=db_dir,
        edge_weight_coefficient=0.1,
        strong_connection_threshold=0.5,
    )
    # Expose the active database to single-worker adapters that apply a
    # post-answer, cross-round memory gate (V10/C2). Normal callers ignore it.
    comp["_last_rag"] = rag

    if loogle_ctx_file:
        # LooGLE mode: one flat document per KG
        if not need_load_data or not rag.database.entity_collection.get()["ids"]:
            ctx_text = open(loogle_ctx_file, encoding="utf-8").read()
            print(f"{title_index} Load LooGLE Data ({len(ctx_text)} chars)")
            rag.load_content(ctx_text, "en")
        else:
            print(f"{title_index} --- LooGLE Data already loaded.")
    elif not need_load_data or not rag.database.entity_collection.get()["ids"]:
        print(f"{title_index} Load Data")
        for context_iter in context:
            rag.load_content(context_iter[0] + "\n" + "\n".join(context_iter[1]), "en")
    else:
        print(f"{title_index} --- Data already loaded.")

    print(f"{title_index} Handle question ({query_type}, do_update={do_update})")
    MeteredOpenaiAgent.reset_usage()

    raw_response, chunks, edges = rag.generate_response(
        chat_history=[], user_input=query, do_update=do_update, force_do_rag=True, max_jumps=max_jumps
    )
    response = RESPONSE_FORMAT.format(query=query, output=raw_response, chunks=str(chunks), edges=str(edges))
    usage_retrieval = MeteredOpenaiAgent.get_usage()

    skip_answer_judge = os.environ.get("REMIND_SKIP_ANSWER_REWRITE_JUDGE", "0") == "1"
    if skip_answer_judge:
        rewrite_response = ""
        ans_check_response = "SKIPPED"
        correct = None
        usage_total = usage_retrieval
    else:
        rewrite_response = comp["ans_rewrite_agent"].generate_response(
            "", [{"role": "user", "content": ANS_REWRITE_PROMPT.format(question=query, og_output=response)}]
        )
        ans_check_response = comp["ans_check_agent"].generate_response(
            "", [{"role": "user", "content": ANS_CHECK_PROMPT.format(
                question=query, reference_answer=ans, generated_output=rewrite_response)}]
        )
        usage_total = MeteredOpenaiAgent.get_usage()
        correct = ans_check_response.strip() == "True"

    result = {
        "title_index": title_index,
        "correct": bool(correct) if correct is not None else None,
        "judge_raw": ans_check_response,
        "tokens_retrieval": usage_retrieval,
        "tokens_total": usage_total,
    }
    with open(os.path.join(db_dir, "input.json"), "w", encoding="utf-8") as f:
        json.dump({
            "query": query, "response": response, "rewrite_response": rewrite_response,
            "real_ans": ans, "evidence": evidence, "check_response": ans_check_response,
            "meta": {"backbone": comp["backbone"], "judge_model": comp["judge_model"],
                     "question_type": query_type, "do_update": do_update,
                     "answer_rewrite_judge": "skipped" if skip_answer_judge else "enabled"},
            "usage": {"retrieval": usage_retrieval, "total": usage_total},
        }, f, ensure_ascii=False, indent=4)
    with open(os.path.join(db_dir, "result_adapted.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    status = "SKIPPED" if correct is None else "RIGHT" if correct else "WRONG"
    print(f"{title_index} OPENFORM {status} | retrieval tokens "
          f"{usage_retrieval['prompt_tokens']}+{usage_retrieval['completion_tokens']}")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run ReMindRag Test---Multi-Hop (adapted)")
    parser.add_argument("--title_index", type=int, required=True)
    parser.add_argument("--test_name", type=str, default="test")
    parser.add_argument("--question_type", type=str, default="origin", choices=["origin", "similar", "different"])
    parser.add_argument("--do_update", type=int, default=0, help="1 = enable memory update")
    args = parser.parse_args()

    comp = build_components()
    run_single(comp, args.title_index, args.test_name, args.question_type, bool(args.do_update))


if __name__ == "__main__":
    main()
