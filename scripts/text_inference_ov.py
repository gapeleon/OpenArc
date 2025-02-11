import torch
from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM
from typing import Optional, Dict, Any
import time
import psutil
import statistics
from dataclasses import dataclass
from typing import List

class InferenceCore:
    def __init__(self, device: str = 'cpu'):
        """Initialize with specified device"""
        self.model = None
        self.device = device  # OpenVINO handles devices differently

    def load_model(self, model_path: str) -> None:
        """Load and prepare the model"""
        self.model = OVModelForCausalLM.from_pretrained(
            model_path,
            device=self.device,
            trust_remote_code=True
        )

    def generate(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                generation_config: Dict[str, Any]) -> torch.Tensor:
        """Run model inference"""

        return self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            **generation_config
        )

class TextGeneration:
    def __init__(self, model_path: str, config: Optional[dict] = None):
        """Initialize with custom config"""
        # Add trust_remote_code for newer models
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Handle tokenizers without pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.default_config = {
            'max_new_tokens': 100,
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.95,
            'repetition_penalty': 1.2,
            'do_sample': True
        }
        
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)

        self.engine = InferenceCore(device='CPU')
        self.engine.load_model(model_path)

    def apply_chat_template(self, prompt: str) -> str:
        """Apply chat template to the prompt"""
        return f"User: {prompt}\nAI:"

    def generate_response(self, prompt: str) -> Optional[str]:
        """Generate a response using the current config"""
        chat_prompt = self.apply_chat_template(prompt)
        inputs = self.tokenizer(chat_prompt, return_tensors="pt", padding=True)
        
        output_ids = self.engine.generate(
            inputs["input_ids"],
            inputs["attention_mask"],
            self.config
        )
        
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


    def generate_response(self, prompt: str) -> Optional[str]:
        """Generate a response using the current config"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        output_ids = self.engine.generate(
            inputs["input_ids"],
            inputs["attention_mask"],
            self.config
        )
        
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

@dataclass
class GenerationMetrics:
    """Store metrics for a single inference"""
    tokens_generated: int
    generation_time: float
    memory_used: float
    tokens_per_second: float

class TorchProfile:
    def __init__(self, inference_engine: TextGeneration):
        self.engine = inference_engine
        self.metrics: List[GenerationMetrics] = []
        
    def _measure_memory(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in generated text"""
        return len(self.engine.tokenizer.encode(text))
    
    def run_profile(self, prompt: str, num_runs: int = 5) -> None:
        """Run multiple inferences and collect metrics"""
        print(f"\nRunning {num_runs} inference(s) for profiling...")
        
        for i in range(num_runs):
            start_time = time.time()
            start_memory = self._measure_memory()
            
            response = self.engine.generate_response(prompt)
            
            end_time = time.time()
            end_memory = self._measure_memory()
            
            if response:
                generation_time = end_time - start_time
                tokens_generated = self._count_tokens(response)
                memory_used = end_memory - start_memory
                tokens_per_second = tokens_generated / generation_time
                
                self.metrics.append(GenerationMetrics(
                    tokens_generated=tokens_generated,
                    generation_time=generation_time,
                    memory_used=memory_used,
                    tokens_per_second=tokens_per_second
                ))
                
                print(f"Run {i+1}/{num_runs} completed")
            else:
                print(f"Run {i+1}/{num_runs} failed")
    
    def print_report(self) -> None:
        """Print a detailed performance report"""
        if not self.metrics:
            print("No metrics available. Run profiling first.")
            return
        
        # Calculate statistics
        stats = {
            'tokens_generated': [m.tokens_generated for m in self.metrics],
            'generation_time': [m.generation_time for m in self.metrics],
            'memory_used': [m.memory_used for m in self.metrics],
            'tokens_per_second': [m.tokens_per_second for m in self.metrics]
        }
        
        print("\n" + "="*50)
        print("INFERENCE PERFORMANCE REPORT")
        print("="*50)
        
        print("\nSystem Information:")
        print(f"CPU Cores: {psutil.cpu_count()}")
        print(f"Total Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
        
        print("\nPerformance Metrics (averaged over {} runs):".format(len(self.metrics)))
        print(f"Average Tokens Generated: {statistics.mean(stats['tokens_generated']):.2f}")
        print(f"Average Generation Time: {statistics.mean(stats['generation_time']):.2f} seconds")
        print(f"Average Memory Used: {statistics.mean(stats['memory_used']):.2f} MB")
        print(f"Average Tokens/Second: {statistics.mean(stats['tokens_per_second']):.2f}")
        
        print("\nVariability Metrics:")
        print(f"Generation Time Std Dev: {statistics.stdev(stats['generation_time']):.2f} seconds")
        print(f"Tokens/Second Std Dev: {statistics.stdev(stats['tokens_per_second']):.2f}")
        
        print("\nRange Metrics:")
        print(f"Fastest Generation: {min(stats['generation_time']):.2f} seconds")
        print(f"Slowest Generation: {max(stats['generation_time']):.2f} seconds")
        print(f"Peak Memory Usage: {max(stats['memory_used']):.2f} MB")
        print("="*50)

def main():
    # Configuration
    model_path = "/media/ecomm/c0889304-9e30-4f04-b290-c7db463872c6/Models/Pytorch/DeepSeek-R1-Distill-Qwen-14B-int4-awq-ov"
    prompt = """
    
    You are R1-Qwen 14b, a medium LLM offering state of the art performance in an edge-safe package.
    Use the following instruction create a prompt that will challenge your teacher using knowledge only a stuent can obtain from his teacher.
    Draw on your training to determine what these facts are; however, try not to approach this as a math problem. Be critical, thoughful most importantl complete, since that is what you were trained to do. 
    Here are som constraints; you hare allowed unlimited thinking tokens, but only two-hundred tokens for your final answer.
    """
    num_runs = 3  # Number of inference runs for profiling
    
    # Optional: Custom configuration
    custom_config = {
        'max_new_tokens': 32000,
        'temperature': 0.6
    }
    
    # Initialize inference engine
    inference = TextGeneration(model_path, custom_config)
    
    # Initialize and run profiler
    profiler = TorchProfile(inference)
    profiler.run_profile(prompt, num_runs)
    profiler.print_report()
    
    # Optional: Print example generation
    print("\nExample Generation:")
    print(f"Prompt: {prompt}")
    print(f"Response: {inference.generate_response(prompt)}")

if __name__ == "__main__":
    main()

