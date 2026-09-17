import pandas as pd
import argparse
import json
import gc
import torch
import yaml
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import yaml


def load_prompt_from_yaml(yaml_file: str) -> str:
    """
    Load prompt from YAML file and combine system_instructions and user_chat_prompt.
    
    Args:
        yaml_file: Path to the YAML configuration file
        
    Returns:
        Combined prompt string to be used as system prompt
    """
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    
    file_processor = config.get('FileProcessor', {})
    system_instructions = file_processor.get('system_instructions', '').strip()
    user_chat_prompt = file_processor.get('user_chat_prompt', '').strip()
    
    # Combine both parts with a newline separator
    combined_prompt = f"{system_instructions}\n\n{user_chat_prompt}"
    
    return combined_prompt


def clear_gpu_memory():
    """Clear GPU memory to prevent accumulation."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def load_json_data_as_df(json_file: str) -> pd.DataFrame:
    """
    Load JSON data directly as a pandas DataFrame.

    Expected JSON format: a list of dicts, e.g.
    [
      {
        "STUDY_ID": ...,
        "NOTE_ID": ...,
        "NOTE_TYPE": "...",
        "NOTE_TEXT_DEID": "..."
      },
      ...
    ]
    """
    df = pd.read_json(json_file)
    # Ensure required columns exist (you can adjust if some are missing)
    if "NOTE_TEXT_DEID" not in df.columns:
        raise ValueError("Input JSON must contain a 'NOTE_TEXT_DEID' field.")
    return df


def run_batch_annotation(note_df: pd.DataFrame, model_path: str, yaml_file: str):
    """Prepare prompts and initialize VLLM for tumor board extraction."""
    
    llm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": 4,  # adjust for your setup
        "gpu_memory_utilization": 0.85,
        "trust_remote_code": True,
        "enforce_eager": False,
        "disable_custom_all_reduce": True,
        "max_model_len": 32768,
        "disable_log_stats": True,
        "max_num_batched_tokens": 32768,
        "block_size": 32
    }
    
    llm = LLM(**llm_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.95,
        max_tokens=32768  # enough for JSON response
    )
    
    system_prompt = load_prompt_from_yaml(yaml_file)
    prompts = []
    
    # Only NOTE_TEXT_DEID is used as the model input
    for _, row in note_df.iterrows():
        note_text = row["NOTE_TEXT_DEID"]

        user_msg = note_text

        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]

        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)

    return prompts, llm, sampling_params


def process_in_batches(prompts: list, llm: LLM, sampling_params: SamplingParams, batch_size: int = 10) -> list:
    """Process prompts in batches with memory management."""
    
    all_outputs = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        batch_prompts = prompts[i:i+batch_size]
        batch_end = min(i+batch_size, len(prompts))
        print(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size} "
              f"(prompts {i+1}-{batch_end})")
        
        clear_gpu_memory()
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU memory: {memory_used:.2f}GB / {memory_total:.2f}GB used")
        
        try:
            batch_responses = llm.generate(batch_prompts, sampling_params=sampling_params)
            
            for response in batch_responses:
                if response.outputs:
                    text = response.outputs[0].text.strip()
                    all_outputs.append(text)
                else:
                    all_outputs.append("")
            
            del batch_responses
            clear_gpu_memory()
            
            print(f"Batch completed. Progress: {len(all_outputs)}/{len(prompts)}")
                
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            clear_gpu_memory()
            raise e
    
    return all_outputs


def process_outputs(outputs: list) -> list:
    """Strip any hidden chain-of-thought markers (</think>) if present."""
    return [x.split('</think>')[-1].strip() if '</think>' in x else x.strip() for x in outputs]


def parse_json_output(output_text: str) -> dict:
    """
    Parse JSON output from LLM response.

    Expected top-level structure:
    {
      "tumor_board_events": [
        {
          "tb_type": "...",
          "tb_date": [...],
          "recommendations": [...]
        },
        ...
      ]
    }
    """
    try:
        if '{' in output_text and '}' in output_text:
            start = output_text.find('{')
            end = output_text.rfind('}') + 1
            json_str = output_text[start:end]
            obj = json.loads(json_str)
        else:
            return {
                "tumor_board_events": [],
                "parse_error": f"No JSON braces found in: {output_text[:100]}..."
            }

        events = obj.get("tumor_board_events", [])
        if not isinstance(events, list):
            events = [events]

        # Optional: you could normalize each event's fields here if you want
        return {
            "tumor_board_events": events,
            "parse_error": ""
        }

    except Exception as e:
        return {
            "tumor_board_events": [],
            "parse_error": f"JSON parsing error: {str(e)}"
        }


def main(input_file: str, output_file: str, model_path: str, yaml_file: str) -> None:
    print("=" * 50)
    print(f"Loading JSON data from {input_file}")
    print(f"Loading prompt configuration from {yaml_file}")

    # Load notes as a DataFrame
    note_df = load_json_data_as_df(input_file)
    print(f"Processing {len(note_df)} notes from JSON file")

    # Basic stats (if available)
    if "STUDY_ID" in note_df.columns:
        print(f"Notes span {note_df['STUDY_ID'].nunique()} unique STUDY_IDs")

    # Prepare prompts and initialize model
    print("Initializing model and preparing prompts...")
    prompts, llm, sampling_params = run_batch_annotation(note_df, model_path, yaml_file)

    # Process in batches
    print("Starting batch processing with memory management...")
    outputs = process_in_batches(prompts, llm, sampling_params, batch_size=10)

    # Process outputs and parse JSON
    print("Processing outputs and parsing JSON responses...")
    processed_outputs = process_outputs(outputs)

    results = []
    for (_, row), output in zip(note_df.iterrows(), processed_outputs):
        parsed = parse_json_output(output)

        result = {
            # keep metadata for later analysis if present
            "STUDY_ID": row.get("STUDY_ID", None),
            "NOTE_ID": row.get("NOTE_ID", None),
            "NOTE_TYPE": row.get("NOTE_TYPE", None),
            # you can drop NOTE_TEXT_DEID here if you don't want it in the CSV
            "NOTE_TEXT_DEID": row["NOTE_TEXT_DEID"],
            "raw_output": output,
            # store tumor_board_events as a JSON string for easy analysis in pandas
            "tumor_board_events": json.dumps(parsed.get("tumor_board_events", [])),
            "parse_error": parsed.get("parse_error", "")
        }
        results.append(result)

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file} with {len(df_results)} rows.")

    print("\nQuick check: notes with any parsing error:")
    print(df_results["parse_error"].replace("", pd.NA).notna().sum())

    print("\nFinished tumor board extraction pipeline!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract tumor board information from clinical notes using VLLM'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSON file with NOTE_TEXT_DEID (and optional metadata like STUDY_ID/NOTE_ID)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file with tumor board extraction results')
    parser.add_argument('--model', type=str,
                        default='/home/bionlp/pubmodels/Qwen3-14B',
                        help='Path to local VLLM model')
    parser.add_argument('--yaml', type=str,
                        default='onco_files_processing.yaml',
                        help='Path to YAML file with prompt configuration (default: onco_files_processing.yaml)')

    args = parser.parse_args()
    main(args.input, args.output, args.model, args.yaml)
