import torch
import re
from typing import List, Dict, Optional


class SimpleTokenizer:
    """Simple tokenizer for text prompts in SAR colorization task"""
    
    def __init__(self, vocab_size: int = 1000, max_length: int = 77):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Special tokens
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.start_token = "[START]"
        self.end_token = "[END]"
        
        # Initialize vocabulary
        self.vocab = {}
        self.inv_vocab = {}
        self._build_base_vocabulary()
        
    def _build_base_vocabulary(self):
        """Build base vocabulary with special tokens and common words"""
        # Special tokens
        special_tokens = [self.pad_token, self.unk_token, self.start_token, self.end_token]
        
        # Common words for our specific task
        task_words = [
            "colorise", "image", "region", "season",
            "tropical", "temperate", "arctic",
            "winter", "spring", "summer", "autumn", "fall",
            "north", "south", "east", "west",
            "land", "water", "ocean", "sea", "coast",
            "mountain", "forest", "desert", "urban", "rural"
        ]
        
        # Punctuation and common words
        common_words = [
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "through", "during",
            "before", "after", "above", "below", "between", "among", "under", "over",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might",
            "can", "this", "that", "these", "those", "i", "you", "he", "she", "it",
            "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
            "her", "its", "our", "their", "mine", "yours", "ours", "theirs",
            ",", ".", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", "-", "_"
        ]
        
        # Numbers
        numbers = [str(i) for i in range(100)]
        
        # Combine all tokens
        all_tokens = special_tokens + task_words + common_words + numbers
        
        # Add to vocabulary
        for i, token in enumerate(all_tokens):
            if i >= self.vocab_size:
                break
            self.vocab[token] = i
            self.inv_vocab[i] = token
            
        # Fill remaining slots with generic tokens
        for i in range(len(all_tokens), self.vocab_size):
            token = f"[TOKEN_{i}]"
            self.vocab[token] = i
            self.inv_vocab[i] = token
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization by splitting on whitespace and punctuation"""
        # Convert to lowercase
        text = text.lower()
        
        # Split on whitespace and common punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text)
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids"""
        tokens = self._tokenize(text)
        
        # Add start token
        token_ids = [self.vocab[self.start_token]]
        
        # Convert tokens to ids
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab[self.unk_token])
        
        # Add end token
        token_ids.append(self.vocab[self.end_token])
        
        # Pad or truncate to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            while len(token_ids) < self.max_length:
                token_ids.append(self.vocab[self.pad_token])
        
        return token_ids
    
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts"""
        token_ids = [self.encode(text) for text in texts]
        return torch.tensor(token_ids, dtype=torch.long)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token ids back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.inv_vocab:
                token = self.inv_vocab[token_id]
                if token not in [self.pad_token, self.start_token, self.end_token]:
                    tokens.append(token)
            else:
                tokens.append(self.unk_token)
        
        return ' '.join(tokens)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.vocab_size
    
    def get_pad_token_id(self) -> int:
        """Get padding token id"""
        return self.vocab[self.pad_token]
    
    def save_vocab(self, path: str):
        """Save vocabulary to file"""
        import json
        vocab_data = {
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'special_tokens': {
                'pad_token': self.pad_token,
                'unk_token': self.unk_token,
                'start_token': self.start_token,
                'end_token': self.end_token
            }
        }
        
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
    
    def load_vocab(self, path: str):
        """Load vocabulary from file"""
        import json
        
        with open(path, 'r') as f:
            vocab_data = json.load(f)
        
        self.vocab = vocab_data['vocab']
        self.vocab_size = vocab_data['vocab_size']
        self.max_length = vocab_data['max_length']
        
        special_tokens = vocab_data['special_tokens']
        self.pad_token = special_tokens['pad_token']
        self.unk_token = special_tokens['unk_token']
        self.start_token = special_tokens['start_token']
        self.end_token = special_tokens['end_token']
        
        # Rebuild inverse vocabulary
        self.inv_vocab = {v: k for k, v in self.vocab.items()}


def create_tokenizer_from_dataset(dataset_path: str, vocab_size: int = 1000) -> SimpleTokenizer:
    """Create tokenizer by analyzing dataset prompts"""
    import os
    import glob
    import pandas as pd
    from collections import Counter
    
    # Get all unique words from the dataset
    csv_files = glob.glob(os.path.join(dataset_path, "data_r_*.csv"))
    
    word_counter = Counter()
    regions = set()
    seasons = set()
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        regions.update(df['region'].unique())
        seasons.update(df['season'].unique())
        
        # Count words in prompts
        for _, row in df.iterrows():
            prompt = f"Colorise image, Region: {row['region']}, Season: {row['season']}"
            words = re.findall(r'\w+|[^\w\s]', prompt.lower())
            word_counter.update(words)
    
    # Create tokenizer with most common words
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    
    # Update vocabulary with dataset-specific words
    special_tokens = [tokenizer.pad_token, tokenizer.unk_token, tokenizer.start_token, tokenizer.end_token]
    most_common_words = [word for word, _ in word_counter.most_common(vocab_size - len(special_tokens))]
    
    # Rebuild vocabulary
    tokenizer.vocab = {}
    tokenizer.inv_vocab = {}
    
    all_tokens = special_tokens + most_common_words
    for i, token in enumerate(all_tokens):
        tokenizer.vocab[token] = i
        tokenizer.inv_vocab[i] = token
    
    return tokenizer


if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = SimpleTokenizer(vocab_size=100)
    
    test_texts = [
        "Colorise image, Region: tropical, Season: summer",
        "Colorise image, Region: arctic, Season: winter",
        "Colorise image, Region: temperate, Season: spring"
    ]
    
    print("Testing tokenizer:")
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"Original: {text}")
        print(f"Encoded: {encoded[:10]}...")  # Show first 10 tokens
        print(f"Decoded: {decoded}")
        print()
    
    # Test batch encoding
    batch_encoded = tokenizer.encode_batch(test_texts)
    print(f"Batch encoded shape: {batch_encoded.shape}")
