#!/usr/bin/env python3
# question_augmentation.py

import os
import json
import argparse
import re
from typing import List, Dict
import requests
import time
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

class Gemini25Augmenter:
    def __init__(self, max_length: int = 1024, retry_count: int = 3):
        """Initialize the question augmenter using Google's Gemini 2.5 API.
        
        Args:
            max_length: Maximum token length for generation
            retry_count: Number of times to retry failed API calls
        """
        self.model_name = "gemini-2.0-flash"
        self.max_length = max_length
        self.retry_count = retry_count
        
        # Get Google Gemini API token from environment variable
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
        
        # Configure the Gemini API
        genai.configure(api_key=self.google_api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(self.model_name)
        
        print(f"Using Google Gemini 2.5 API")
    
    def chunk_document(self, document: str, chunk_size: int = 4000) -> List[str]:
        """Split document into manageable chunks based on character count.
        
        Args:
            document: The document text
            chunk_size: Approximate character count per chunk
            
        Returns:
            List of document chunks
        """
        # Simple character-based chunking
        words = document.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    def extract_json_from_text(self, text: str) -> List[Dict[str, str]]:
        """Extract JSON from generated text with robust error handling.
        
        Args:
            text: Generated text that should contain JSON
            
        Returns:
            List of QA pairs extracted from the text
        """
        # Try standard JSON extraction first
        json_start = text.find("[")
        json_end = text.rfind("]") + 1
        
        if json_start != -1 and json_end > json_start:
            json_text = text[json_start:json_end]
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                fixed_json = self._fix_json(json_text)
                try:
                    return json.loads(fixed_json)
                except json.JSONDecodeError:
                    pass  # Continue to fallback methods
        
        # Fallback 1: Try regex to extract individual QA pairs
        qa_pairs = []
        pattern = r'"question"\s*:\s*"([^"]*)"\s*,\s*"answer"\s*:\s*"([^"]*)"'
        matches = re.findall(pattern, text)
        
        if matches:
            for question, answer in matches:
                qa_pairs.append({"question": question, "answer": answer})
            return qa_pairs
        
        # Fallback 2: Look for lines that might be Q&A
        lines = text.split('\n')
        q_prefixes = ["Q:", "Question:", "Q ", "Question "]
        a_prefixes = ["A:", "Answer:", "A ", "Answer "]
        
        current_question = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            is_question = False
            for prefix in q_prefixes:
                if line.startswith(prefix):
                    current_question = line[len(prefix):].strip()
                    is_question = True
                    break
                    
            if is_question:
                continue
                
            for prefix in a_prefixes:
                if line.startswith(prefix) and current_question:
                    answer = line[len(prefix):].strip()
                    qa_pairs.append({"question": current_question, "answer": answer})
                    current_question = None
                    break
        
        if qa_pairs:
            return qa_pairs
            
        # If we couldn't extract structured pairs, create minimal info
        if len(text.strip()) > 0:
            return [{"question": "Generated from text", "answer": text[:500] + "..." if len(text) > 500 else text}]
        
        return []
    
    def _fix_json(self, json_text: str) -> str:
        """Attempt to fix common JSON formatting errors."""
        # Fix unescaped quotes
        fixed = re.sub(r'(?<!\\)"(?=(.*?)".*?":")', r'\"', json_text)
        
        # Fix trailing commas in arrays
        fixed = re.sub(r',\s*]', ']', fixed)
        
        # Fix missing commas between objects
        fixed = re.sub(r'}\s*{', '},{', fixed)
        
        # Ensure the JSON is properly wrapped in an array
        if not fixed.startswith('['):
            fixed = '[' + fixed
        if not fixed.endswith(']'):
            fixed = fixed + ']'
            
        return fixed
    
    def generate_qa_pairs(self, chunk: str, num_pairs: int = 3000) -> List[Dict[str, str]]:
        """Generate question-answer pairs from a document chunk using Gemini 2.5 API.
        
        Args:
            chunk: Document chunk text
            num_pairs: Number of QA pairs to generate
            
        Returns:
            List of dictionaries containing questions and answers
        """
        prompt_template = f"""
Context:
{chunk}

Task: Based on the context provided above, generate {num_pairs} different question and answer pairs that cover key information. The questions should be diverse, covering different aspects of the content, and be natural and conversational, as if a user is asking a chatbot. Include both simple factual questions and more complex analytical questions.

The questions should:
1. Cover all important facts, concepts, and information in the text
4. Be clear, specific, and unambiguous
5. Sound natural, as if asked by a human user
6. Represent what real users would likely ask about MDGSpace
7. Focus only on information contained in the context

For answers:
    1. Provide comprehensive responses based solely on the information in the context
    2. For questions outside the scope of the provided context, create answers that politely explain the chatbot can only answer queries related to MDGSpace
    3. Include some examples of "I don't know" or "I can only answer questions about MDGSpace" responses for off-topic questions
    

Format your response as JSON with the following structure:
[
  {{
    "question": "Specific question from the content",
    "answer": "Detailed answer based on the content"
  }},
  ...
]

Generate exactly {num_pairs} unique question-answer pairs. Return only valid JSON without additional text or explanations outside the JSON structure.

JSON Response:
"""
        
        # Try with retries
        for attempt in range(self.retry_count):
            try:
                # Generate content with Gemini
                response = self.model.generate_content(prompt_template)
                
                # Extract text from response
                generated_text = response.text
                
                # Extract QA pairs from the generated text
                qa_pairs = self.extract_json_from_text(generated_text)
                
                if qa_pairs:
                    print(f"Successfully generated {len(qa_pairs)} QA pairs")
                    # Make sure we don't exceed the requested number of pairs
                    return qa_pairs[:num_pairs]
                else:
                    print("Failed to extract QA pairs from the model's response.")
                    print(f"Raw response: {generated_text[:200]}...")
                    # Try again if we have retries left
                    if attempt < self.retry_count - 1:
                        wait_time = 2 ** attempt
                        print(f"Retrying with a different prompt in {wait_time} seconds... (Attempt {attempt + 1}/{self.retry_count})")
                        time.sleep(wait_time)
                    else:
                        # Create a fallback QA pair if all extraction attempts fail
                        return [{"question": "What is this text about?", 
                                "answer": chunk[:200] + "..." if len(chunk) > 200 else chunk}]
                    
            except Exception as e:
                print(f"API request error: {e}")
                if attempt < self.retry_count - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{self.retry_count})")
                    time.sleep(wait_time)
                else:
                    # Create a fallback QA pair if all extraction attempts fail
                    return [{"question": "What is this text about?", 
                            "answer": chunk[:200] + "..." if len(chunk) > 200 else chunk}]
        
        # If all retries fail, return a fallback pair
        return [{"question": "What is this text about?", 
                "answer": chunk[:200] + "..." if len(chunk) > 200 else chunk}]
    
    def process_document(self, document_path: str, 
                         num_pairs_per_chunk: int = 1000) -> List[Dict[str, str]]:
        """Process a document and generate QA pairs.
        
        Args:
            document_path: Path to the document file
            num_pairs_per_chunk: Number of QA pairs to generate per chunk
            
        Returns:
            List of QA pairs generated from the document
        """
        print(f"Processing document: {document_path}")
        
        try:
            with open(document_path, 'r', encoding='utf-8') as file:
                document_text = file.read()
        except Exception as e:
            print(f"Error reading document {document_path}: {e}")
            return []
        
        chunks = self.chunk_document(document_text)
        print(f"Document split into {len(chunks)} chunks")
        
        all_qa_pairs = []
        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            qa_pairs = self.generate_qa_pairs(chunk, num_pairs_per_chunk)
            if qa_pairs:
                all_qa_pairs.extend(qa_pairs)
                # Save intermediate results to avoid losing all progress if an error occurs
                tmp_filename = f"tmp_qa_pairs_{i}.json"
                with open(tmp_filename, 'w', encoding='utf-8') as f:
                    json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
                print(f"Saved intermediate results to {tmp_filename}")
        
        return all_qa_pairs

def save_qa_pairs(qa_pairs: List[Dict[str, str]], output_file: str):
    """Save QA pairs to a JSON file, appending if the file exists.
    
    Args:
        qa_pairs: List of QA pairs to save
        output_file: Path to the output file
    """
    if not qa_pairs:
        print("Warning: No QA pairs to save.")
        return
    
    # Filter out placeholder entries
    filtered_qa_pairs = []
    for pair in qa_pairs:
        question = pair.get("question", "")
        answer = pair.get("answer", "")
        
        # Skip entries that contain placeholder text
        if ("Question " in question and question.strip().isdigit()) or \
           question == "Question 1" or \
           answer == "Answer 1" or \
           (answer.startswith("Answer ") and answer[7:].strip().isdigit()):
            continue
            
        filtered_qa_pairs.append(pair)
    
    print(f"Filtered out {len(qa_pairs) - len(filtered_qa_pairs)} placeholder entries")
    
    existing_data = []
    
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as file:
                existing_data = json.load(file)
                
            # Also filter existing data to remove placeholders
            existing_data = [pair for pair in existing_data if 
                           pair.get("question") != "Question 1" and 
                           pair.get("answer") != "Answer 1"]
                
        except json.JSONDecodeError:
            print(f"Warning: Existing file {output_file} contains invalid JSON. Creating new file.")
    
    # Add new QA pairs
    existing_data.extend(filtered_qa_pairs)
    
    # Save the updated data
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(filtered_qa_pairs)} new QA pairs to {output_file}")
    print(f"Total QA pairs in file: {len(existing_data)}")

def main():
    parser = argparse.ArgumentParser(description="Generate QA pairs from documents using Google's Gemini 2.5 API")
    parser.add_argument("--input", type=str, required=True, help="Path to input document or directory")
    parser.add_argument("--output", type=str, default="qa_dataset.json", help="Output JSON file path")
    parser.add_argument("--pairs-per-chunk", type=int, default=1000, help="Number of QA pairs per chunk")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries for failed API calls")
    
    args = parser.parse_args()
    
    augmenter = Gemini25Augmenter(retry_count=args.retries)
    
    if os.path.isdir(args.input):
        # Process all text files in directory
        all_qa_pairs = []
        for filename in os.listdir(args.input):
            if filename.endswith(('.txt', '.md', '.html', '.csv')):
                file_path = os.path.join(args.input, filename)
                qa_pairs = augmenter.process_document(file_path, args.pairs_per_chunk)
                all_qa_pairs.extend(qa_pairs)
                # Save after each file to prevent losing progress
                save_qa_pairs(all_qa_pairs, args.output)
    else:
        # Process single document
        qa_pairs = augmenter.process_document(args.input, args.pairs_per_chunk)
        save_qa_pairs(qa_pairs, args.output)
        
    # Process any intermediate files that were created
    intermediate_files = [f for f in os.listdir('.') if f.startswith('tmp_qa_pairs_') and f.endswith('.json')]
    if intermediate_files:
        print(f"Found {len(intermediate_files)} intermediate files. Cleaning up...")
        for f in intermediate_files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Error removing intermediate file {f}: {e}")

if __name__ == "__main__":
    main()