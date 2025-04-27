
# Credential Leak Extractor

A tool for extracting email/password pairs from credential leak database dumps using LLMs.

## Repository Contents

- `LeakExtractorNoBatch.py`: Main script for extracting credentials from leak data
- `testdata.txt`: Large test dataset (~8GB) for credential extraction
- `testdata-small`: Smaller test dataset for quick testing
- `credentials.json`: Example output file with extracted credentials
- `stringparser.ipynb`: Jupyter notebook for development and testing
- `llm_cache`: Directory for caching LLM responses
- `old`: Directory containing older versions of the code

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CyberSecurityAI.git
cd CyberSecurityAI

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch transformers tqdm bitsandbytes
```

### Hardware Acceleration

The tool automatically detects and uses the best available hardware:

- **CUDA**: For NVIDIA GPUs, with 4-bit quantization for memory efficiency
- **MPS**: For Apple Silicon (M1/M2/M3) Macs
- **CPU**: As fallback when no GPU is available

No configuration is needed as the script automatically detects the available hardware.

## Usage

### Basic Usage

```bash
python LeakExtractorNoBatch.py --input testdata.txt --output credentials.json
```

### Advanced Usage

```bash
# Use a specific model with temperature control
python LeakExtractorNoBatch.py --input testdata.txt --output credentials.json --model google/gemma-3-4b-it --temperature 0.2

# Process only the first 1000 lines (for testing)
python LeakExtractorNoBatch.py --input testdata.txt --output test_results.json --limit 1000

# Search existing results for specific domain
python LeakExtractorNoBatch.py --output credentials.json --search "company.com"

# Adjust batch size for processing
python LeakExtractorNoBatch.py --input testdata.txt --output credentials.json --batch-size 10
```

## Command-Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--input` | `-i` | Path to the input database dump file |
| `--output` | `-o` | Path to save the output JSON file |
| `--model` | `-m` | HuggingFace model to use (default: google/gemma-3-4b-it) |
| `--search` | `-s` | Search term to find in processed credentials |
| `--limit` | `-l` | Limit processing to specified number of lines |
| `--temperature` | `-t` | Temperature for model generation (default: 0.1, 0=deterministic) |
| `--batch-size` | `-b` | Number of lines to process in each prompt (default: 5) |

## Output Format

The tool generates a JSON file with the following structure:

```json
{
  "leaked_credentials": [
    {
      "email": "user@example.com",
      "password": "password123"
    },
    {
      "email": "another@example.org",
      "password": "securepass!"
    }
  ]
}
```

Each entry in the `leaked_credentials` array contains an email and its associated password extracted from the input file. The tool automatically deduplicates credentials to avoid repetition.
