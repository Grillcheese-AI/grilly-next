from transformers import AutoTokenizer

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
VOCAB_SIZE = 152453

print(f"Downloading tokenizer for {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print("Exporting vocabulary to vocab.txt...")
with open("vocab.txt", "w", encoding="utf-8") as f:
    for i in range(VOCAB_SIZE):
        # We try to decode the exact integer ID back to text
        try:
            token_str = tokenizer.decode([i])
            # If the token is empty (like a whitespace token), we use a placeholder
            if not token_str or token_str.isspace():
                token_str = f"<SPACE_{i}>"
        except Exception:
            # If it fails to decode, it's an unused/special token
            token_str = f"<UNK_{i}>"
        
        # Replace newlines so they don't break the text file formatting
        token_str = token_str.replace('\n', '<NEWLINE>')
        
        f.write(token_str + "\n")

print("Successfully exported vocab.txt!")