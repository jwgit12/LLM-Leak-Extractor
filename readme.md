
## TODO

- [ ] Add multiple line processing for context
- [ ] Get JSON template from Sebastian

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/credential-leak-processor.git
cd credential-leak-processor

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch transformers tqdm
```

## Usage

### Basic Usage

```bash
python processor.py --input leak_data.txt --output results.json
```

### Advanced Usage

```bash
# Use a specific model with temperature control
python processor.py --input breach_dump.txt --output results.json --model mistralai/Mistral-7B-Instruct-v0.2 --temperature 0.2

# Process only the first 1000 lines (for testing)
python processor.py --input large_dump.txt --output test_results.json --limit 1000

# Search existing results for specific domain
python processor.py --output results.json --search "company.com"
```

## Command-Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--input` | `-i` | Path to the input database dump file |
| `--output` | `-o` | Path to save the output JSON file |
| `--model` | `-m` | HuggingFace model to use (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0) |
| `--search` | `-s` | Search term to find in processed credentials |
| `--limit` | `-l` | Limit processing to specified number of lines |
| `--temperature` | `-t` | Temperature for model generation (default: 0.1, 0=deterministic) |

