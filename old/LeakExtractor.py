import os
import json
import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from concurrent.futures import ThreadPoolExecutor, as_completed


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


def extract_credentials_from_response(response):
    """Extract only email and password from model response."""
    # Try to extract email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, response)

    # Try to extract password patterns
    password_pattern = r'password[:\s]+([^\s,]+)'
    passwords = re.findall(password_pattern, response, re.IGNORECASE)

    # Also try more general patterns for passwords if the line contains an email
    if emails and not passwords:
        # Look for patterns like "email:user@example.com pass:12345"
        alt_password_pattern = r'pass(?:word)?[\s:]+([^\s,]+)'
        passwords = re.findall(alt_password_pattern, response, re.IGNORECASE)

        # If still no passwords, try looking for tokens after the email
        if not passwords and ":" in response:
            parts = response.split(":")
            for i, part in enumerate(parts):
                if emails[0] in part and i + 1 < len(parts):
                    potential_pass = parts[i + 1].strip().split()[0]
                    if potential_pass and not '@' in potential_pass:
                        passwords = [potential_pass]
                        break

    # If we found both email and password, create a credential entry
    if emails and passwords:
        return [{"email": emails[0], "password": passwords[0]}]
    elif emails:  # If we only found email
        return [{"email": emails[0], "password": ""}]
    else:
        return []


def process_line(line, model, tokenizer, temperature=0.1):
    """Process a single line of text through the LLM."""
    # Skip empty lines
    if not line.strip():
        return []

    # Create a simple prompt for the model
    prompt = f"""If this text contains an email and password, extract them. If not found, respond with "None".

Text: {line}

Format:
Email: [email]
Password: [password]
"""

    # For models that expect a chat format
    chat_format = [
        {"role": "system",
         "content": "You extract email and password pairs from text. Only respond if you find them, otherwise say None."},
        {"role": "user", "content": prompt}
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
                max_new_tokens=128,  # Shorter output is fine for email/password
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

        # If response indicates "None" or similar, return empty
        if re.search(r'\b(none|not found|no email|no password)\b', response.lower()):
            return []

        return extract_credentials_from_response(response)

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return []


def process_lines_batch(lines, model, tokenizer, max_workers=4, temperature=0.1):
    """Process multiple lines in parallel."""
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_line = {
            executor.submit(process_line, line, model, tokenizer, temperature): line
            for line in lines
        }

        for future in as_completed(future_to_line):
            try:
                creds = future.result()
                if creds:
                    results.extend(creds)
            except Exception as e:
                print(f"Error processing line: {str(e)}")

    return results


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


def process_database_dump(file_path, output_path, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", batch_size=100,
                          limit_lines=None, temperature=0.1):
    """Process the database dump file line by line and save the results."""
    # Load the model
    model, tokenizer, device = load_model(model_name)

    # Determine parallelization based on device
    max_workers = 1  # Default for CPU and MPS
    if device == "cuda":
        # Use more workers for CUDA
        max_workers = min(4, torch.cuda.device_count())

    # Count lines for progress tracking
    total_lines = get_file_line_count(file_path)
    if limit_lines and limit_lines > 0:
        total_lines = min(total_lines, limit_lines)

    print(f"Processing {total_lines} lines from {file_path}")

    all_credentials = []
    line_count = 0
    batch = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, total=total_lines, desc="Reading lines"):
            # Skip empty lines
            if not line.strip():
                continue

            batch.append(line.strip())
            line_count += 1

            # Process batch when it reaches the batch size
            if len(batch) >= batch_size:
                if max_workers > 1:
                    # Process in parallel
                    batch_results = process_lines_batch(batch, model, tokenizer, max_workers, temperature)
                else:
                    # Process sequentially
                    batch_results = []
                    for batch_line in batch:
                        creds = process_line(batch_line, model, tokenizer, temperature)
                        if creds:
                            batch_results.extend(creds)

                all_credentials.extend(batch_results)
                batch = []

                # Optional: Save intermediate results periodically
                if line_count % 1000 == 0:
                    print(
                        f"Processed {line_count}/{total_lines} lines. Found {len(all_credentials)} credentials so far.")
                    temp_output = {"leaked_credentials": all_credentials}
                    try:
                        with open(f"{output_path}.temp", 'w') as f:
                            json.dump(temp_output, f, indent=2)
                    except Exception as e:
                        print(f"Error saving intermediate results: {str(e)}")

            # Check if we've reached the limit
            if limit_lines and line_count >= limit_lines:
                break

    # Process any remaining lines in the batch
    if batch:
        if max_workers > 1:
            batch_results = process_lines_batch(batch, model, tokenizer, max_workers, temperature)
        else:
            batch_results = []
            for batch_line in batch:
                creds = process_line(batch_line, model, tokenizer, temperature)
                if creds:
                    batch_results.extend(creds)

        all_credentials.extend(batch_results)

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
    parser.add_argument("--model", "-m", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="HuggingFace model to use for extraction (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)")
    parser.add_argument("--search", "-s", help="Search term to find in processed credentials")
    parser.add_argument("--limit", "-l", type=int, help="Limit processing to this many lines (for testing)")
    parser.add_argument("--batch-size", "-b", type=int, default=100,
                        help="Number of lines to process in a batch (default: 100)")
    parser.add_argument("--temperature", "-t", type=float, default=0.1,
                        help="Temperature for model generation (default: 0.1, 0=deterministic)")

    args = parser.parse_args()

    if args.search and os.path.exists(args.output):
        # If we're just searching existing results
        results = search_credentials(args.output, args.search)
        print(f"Found {len(results)} credentials matching '{args.search}':")
        for cred in results:
            print(f"Email: {cred.get('email', 'N/A')}, Password: {cred.get('password', 'N/A')}")
    else:
        # Process the database dump
        process_database_dump(args.input, args.output, args.model, args.batch_size, args.limit, args.temperature)


if __name__ == "__main__":
    main()