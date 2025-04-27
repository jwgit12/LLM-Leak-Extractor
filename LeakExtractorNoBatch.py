import os
import json
import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def detect_device():
    """Detect the available hardware for inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_model(model_name):
    """Load the LLM model based on available hardware."""
    device = detect_device()
    print(f"Using device: {device}")

    # Configure quantization for memory efficiency
    if device == "cuda":
        # Use BitsAndBytes for CUDA to reduce memory usage
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
    else:
        # For MPS or CPU
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if device == "mps":
            model = model.to(torch.device("mps"))
        elif device == "cpu":
            model = model.to(torch.device("cpu"))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer, device


def process_lines(lines, model, tokenizer, temperature=0.1):
    """Process multiple lines of text (batch) through the LLM."""
    # Skip if all lines are empty
    if not any(line.strip() for line in lines):
        return []

    # Create a formatted multi-line input
    formatted_lines = "\n".join([f"Line {i + 1}: {line.strip()}" for i, line in enumerate(lines) if line.strip()])

    # Create a prompt that directly asks for JSON output
    prompt = f"""
    TASK: Extract every email / password pair from the section marked TEXT.

    RULES
    1. An *email/password pair* exists only when **both** appear in the same line.  
    2. Valid email: matches `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{{2,}}`.  
    3. Password: any non‑empty token that follows a typical delimiter  
       (`:`, `=`, space, “password”, “pwd”, “pass”, etc.).  
    4. If multiple pairs occur on the same line, output each pair separately.  
    5. Ignore everything except the required email/password strings.  
    6. Preserve original casing; do **not** trim internal characters.  

    OUTPUT FORMAT:  
    Return **only** a JSON array, minified (no extra spaces, new‑lines, or keys):  

    [
      {{"email":"<email1>","password":"<password1>"}},
      {{"email":"<email2>","password":"<password2>"}}
    ]

    If no valid pairs are found, return exactly `[]` (empty array).  
    Never add explanations, markdown, or trailing text.

    TEXT
    {formatted_lines}
    """

    # ---------- chat wrapper ----------
    chat_format = [
        {
            "role": "system",
            "content": (
                "You are an extraction engine. Follow the RULES strictly and "
                "respond with valid, minified JSON only—no prose or markdown."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        # Handle both chat and non-chat models
        if hasattr(tokenizer, "apply_chat_template"):
            # For chat models like Llama-2-chat
            formatted_prompt = tokenizer.apply_chat_template(chat_format, return_tensors="pt")
        else:
            # For non-chat models
            formatted_prompt = tokenizer(prompt, return_tensors="pt")

        # Move inputs to the appropriate device
        device = next(model.parameters()).device
        if isinstance(formatted_prompt, dict):
            inputs = {k: v.to(device) for k, v in formatted_prompt.items()}
        else:
            inputs = {"input_ids": formatted_prompt.to(device)}

        # Generate response
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,  # Increased for multiple outputs
                temperature=temperature,
                do_sample=temperature > 0,
            )


        # Handle both chat and non-chat models for decoding
        if hasattr(tokenizer, "apply_chat_template"):
            response_ids = output[0][len(inputs["input_ids"][0]):]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)

        else:
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            # Remove the prompt from the response if it's included
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
        # Extract just the JSON part from the response
        json_match = re.search(r'\[.*\]|\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                result = json.loads(json_str)
                # Handle both array and object responses
                if isinstance(result, list):
                    # Return only entries with email field
                    return [cred for cred in result if "email" in cred]
                elif isinstance(result, dict):
                    # Single entry result
                    if "email" in result:
                        return [{"email": result["email"], "password": result.get("password", "")}]
                    # "not found" format
                    elif result.get("found") is False:
                        return []
            except json.JSONDecodeError:
                # If JSON parsing fails, we return empty
                pass

        return []

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return []


def get_file_line_count(file_path):
    """Count the number of lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"Error counting lines in file {file_path}: {str(e)}")
        return 0


def deduplicate_credentials(credentials):
    """Remove duplicate credential entries."""
    seen = set()
    unique_credentials = []

    for cred in credentials:
        # Create a string representation for deduplication
        email = cred.get('email', '')
        password = cred.get('password', '')

        # Skip empty entries
        if not email:
            continue

        key = f"{email}-{password}"

        if key not in seen:
            seen.add(key)
            unique_credentials.append(cred)

    return unique_credentials


def process_database_dump(file_path, output_path, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", limit_lines=None,
                          temperature=0.1, batch_size=5):
    """Process the database dump file in batches and save the results."""
    # Load the model
    model, tokenizer, device = load_model(model_name)

    # Count lines for progress tracking
    total_lines = get_file_line_count(file_path)
    if limit_lines and limit_lines > 0:
        total_lines = min(total_lines, limit_lines)

    print(f"Processing {total_lines} lines from {file_path} in batches of {batch_size}")

    all_credentials = []
    line_count = 0
    current_batch = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, total=total_lines, desc="Processing lines"):
            # Add line to current batch
            if line.strip():
                current_batch.append(line.strip())

            line_count += 1

            # When batch is full or we're at the last line or reached limit, process it
            if len(current_batch) >= batch_size or line_count == total_lines or (
                    limit_lines and line_count >= limit_lines):
                if current_batch:  # Only process if we have lines
                    creds = process_lines(current_batch, model, tokenizer, temperature)
                    if creds:
                        all_credentials.extend(creds)

                    # Reset batch
                    current_batch = []

            # Optional: Save intermediate results periodically
            if line_count % 1000 == 0:
                print(f"Processed {line_count}/{total_lines} lines. Found {len(all_credentials)} credentials so far.")
                temp_output = {"leaked_credentials": all_credentials}
                try:
                    with open(f"{output_path}.temp", 'w') as f:
                        json.dump(temp_output, f, indent=2)
                except Exception as e:
                    print(f"Error saving intermediate results: {str(e)}")

            # Check if we've reached the limit
            if limit_lines and line_count >= limit_lines:
                break

    # Process any remaining lines in the last batch
    if current_batch:
        creds = process_lines(current_batch, model, tokenizer, temperature)
        if creds:
            all_credentials.extend(creds)

    # Deduplicate the credentials
    unique_credentials = deduplicate_credentials(all_credentials)

    # Save the final results
    output_data = {"leaked_credentials": unique_credentials}
    try:
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
    except Exception as e:
        print(f"Error saving final results to {output_path}: {str(e)}")
        # Try saving to a different location
        backup_path = f"{output_path}.backup"
        try:
            with open(backup_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Saved backup results to {backup_path}")
        except:
            print("Failed to save results.")

    print(
        f"Processing complete. Found {len(unique_credentials)} unique credential entries out of {len(all_credentials)} total.")
    return output_data


def search_credentials(output_file, search_term):
    """Search for credentials matching a specific term."""
    try:
        with open(output_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading output file {output_file}: {str(e)}")
        return []

    results = []
    for cred in data.get("leaked_credentials", []):
        if search_term.lower() in cred.get("email", "").lower():
            results.append(cred)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Process credential leak database dumps using LLM")
    parser.add_argument("--input", "-i", required=True, help="Path to the input database dump file")
    parser.add_argument("--output", "-o", required=True, help="Path to save the output JSON file")
    parser.add_argument("--model", "-m", default="google/gemma-3-4b-it",
                        help="HuggingFace model to use for extraction (default: google/gemma-3-4b-it)")
    parser.add_argument("--search", "-s", help="Search term to find in processed credentials")
    parser.add_argument("--limit", "-l", type=int, help="Limit processing to this many lines (for testing)")
    parser.add_argument("--temperature", "-t", type=float, default=0.1,
                        help="Temperature for model generation (default: 0.1, 0=deterministic)")
    parser.add_argument("--batch-size", "-b", type=int, default=5,
                        help="Number of lines to process in each prompt (default: 5)")

    args = parser.parse_args()

    if args.search and os.path.exists(args.output):
        # If we're just searching existing results
        results = search_credentials(args.output, args.search)
        print(f"Found {len(results)} credentials matching '{args.search}':")
        for cred in results:
            print(f"Email: {cred.get('email', 'N/A')}, Password: {cred.get('password', 'N/A')}")
    else:
        # Process the database dump
        process_database_dump(args.input, args.output, args.model, args.limit, args.temperature, args.batch_size)


if __name__ == "__main__":
    main()