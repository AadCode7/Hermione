from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from config import LLM_MODEL, MAX_TOKENS, TEMPERATURE, CHARACTER_SYSTEM_PROMPT

class LLMComponents:
    def __init__(self):
        # Set device to CUDA if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Create text generation pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id
        )

    def generate_response(self, prompt):
        """Generate a response as Hermione"""
        try:
            # Create the full prompt with system message and chat history
            full_prompt = f"{CHARACTER_SYSTEM_PROMPT}\n\nUser: {prompt}\n\nHermione:"
            
            # Generate response
            response = self.pipeline(full_prompt)[0]['generated_text']
            
            # Extract just Hermione's response
            hermione_response = response.split("Hermione:")[-1].strip()
            
            # Ensure the response is not empty
            if not hermione_response:
                return "I'm sorry, I didn't quite catch that. Could you please repeat your question?"
            
            return hermione_response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I'm sorry, I'm having trouble processing that. Could you please try asking something else?" 