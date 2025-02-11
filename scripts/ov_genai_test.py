#import openvino_genai
import openvino_genai
from openvino_genai import LLMPipeline
model_dir = "/mnt/Ironwolf-4TB/Models/Pytorch/Qwen2.5-0.5B-Instruct-ov"

def streamer(subword):
    print(subword, end='', flush=True)
    return False


def infer(model_dir: str):
    device = 'CPU'  # GPU can be used as well.
    pipe = openvino_genai.LLMPipeline(model_dir, device)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100

    pipe.start_chat()
    while True:
        try:
            prompt = input('question:\n')
        except EOFError:
            break
        pipe.generate(prompt, config, streamer)
        print('\n----------')
    pipe.finish_chat()