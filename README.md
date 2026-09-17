# Tumor Board Summarization

This repository provides prompts and code for:

1. Structuring and processing semi-structured tumor board case summaries and recommendation outcomes.
2. Processing and scoring fine-grained evaluation rubrics.
3. Generating case and outcome summaries using LLM APIs.
4. Evaluating reference and candidate summaries using LLM-as-a-judge methods and the `tb-fact` metric.

## Case Summarization

Case summaries contain information presented to the tumor board, including treatment options discussed during the meeting.

We define the case summarization task as a composition of four main sections:

- **summary**: A brief case summary and the reason for discussion.
- **oncological_history**: A semi-structured timeline of important patient events.
- **considered_options**: Potential next steps relevant to the patient.
- **supporting_data**: Excerpts from structured data relevant to the case discussion.

## Outcome Summarization

Tumor board outcomes summarize the board's discussion and recommendations, including next steps such as additional testing or treatment.

## Repository Structure

- `scripts/`: Data-processing scripts and demonstration notebooks.
- `data/`: Processed data.
- `code/`: Summary generation and evaluation code.
  - `prompts/`: YAML task prompts.
  - `config.yaml`: API configuration and credentials.
  - `tests/`: Unit tests for attribute extraction and classification conversions.
  - `tb_records.py`: Structures tumor board summaries, rubrics, and graded rubrics.
  - `tb_eval.py`: Implements evaluation methods.

The `scripts/tbeval-demo.ipynb` notebook demonstrates how to call the prompts and calculate `tb-fact` scores.

You can customize `tb-fact` by changing its weights. Evaluation can also use patient history as context instead of a reference summary.

## Evaluation Metrics

The repository supports several LLM-based evaluation methods using models such as GPT-4.1 and Qwen Plus:

- **LLMAsJudgeGeneric**: Rates a candidate summary against a reference on a 1–5 scale.
- **LLMAsJudge4Axes**: Rates a candidate against a reference on four dimensions: completeness, factual accuracy, relevance, and overall quality.
- **tb-fact**: A family of fact-based evaluation metrics.

### `tb-fact`

`tb-fact` decomposes reference and candidate summaries into facts using an expert-validated schema. It then:

1. Classifies entailment for each fact against the reference and source records.
2. Maps correlated reference and candidate facts.
3. Aggregates the results into precision and recall scores.

### `tb-fact-rf`

`tb-fact-rf` is the reference-free variant. It decomposes a candidate summary into facts and classifies each fact for entailment, hallucination, and importance using the patient history as context.