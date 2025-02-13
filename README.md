### OpenARC


# Welcome to OpenARC

OpenArc is a lightweight inference API backend for Optimum-Intel from Transformers to leverage hardware acceleration on Intel CPUs, GPUs and NPUs through the OpenVINO runtime.
It has been designed with text-to-text agentic use cases in mind. 

Generally OpenArc serves inference and integrates well with transformers!

Under the hood it's a strongly typed fastAPI implementation of [OVModelForCausalLM](https://huggingface.co/docs/optimum/main/en/intel/openvino/reference#optimum.intel.OVModelForCausalLM) from Optimum-Intel to make deploying inference use less of the same code, while reaping the benefits of fastAPI type safety and ease of use. Keep your application code seperate from inference code. 

Here are some features:

- **Strongly typed API with four endpoints**
	- /model/load: loads model and accepts ov_config
	- /model/unload: use gc to purge a loaded model
	- /generate/text: synchronous execution,  select sampling parameters, token limits : also returns a performance report
	- /status: see the loaded model 
- Each endpoint has a pydantic model keeping exposed parameters easy to maintain or extend.
- Native chat templating


 
| Pre Converted Models                                                                                                                                      | Compressed Weights |
| ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| [Echo9Zulu/Ministral-3b-instruct-int4_asym-ov](https://huggingface.co/Echo9Zulu/Ministral-3b-instruct-int4_asym-ov)                                   | 1.85 GB            |
| [Echo9Zulu/Llama-3.1-Tulu-3-8B-int4_asym-ov](https://huggingface.co/Echo9Zulu/Llama-3.1-Tulu-3-8B-int4_asym-ov/tree/main)                             | 4.68 GB            |
| [Echo9Zulu/Falcon3-10B-Instruct-int4_asym-ov](https://huggingface.co/Echo9Zulu/Falcon3-10B-Instruct-int4_asym-ov)                                     | 5.74 GB            |
| [Echo9Zulu/DeepSeek-R1-Distill-Qwen-14B-int4-awq-ov](https://huggingface.co/Echo9Zulu/DeepSeek-R1-Distill-Qwen-14B-int4-awq-ov/tree/main)             | 7.68 GB            |
| [Echo9Zulu/Phi-4-o1-int4_asym-awq-weight_quantization_error-ov](https://huggingface.co/Echo9Zulu/Phi-4-o1-int4_asym-awq-weight_quantization_error-ov) | 8.11 GB            |
| [Echo9Zulu/Qwen2.5-72B-Instruct-int4-ns-ov](https://huggingface.co/Echo9Zulu/Qwen2.5-72B-Instruct-int4-ns-ov/tree/main))                              | 39 GB              |

I have 40+ preconverted text generation models of larger variety that what's offered by the official repos for OpenVINO on my [HuggingFace](https://huggingface.co/Echo9Zulu).

You can easily craft conversion commands using my HF Space, [Optimum-CLI-Tool_tool](https://huggingface.co/spaces/Echo9Zulu/Optimum-CLI-Tool_tool)!

Learn more about model conversion and working with Intel Devices generally in the openvino_utilities notebook.

### Roadmap

Currently my plan is to transition from Optimum to OpenVINO GenAI since it handles everything here and much more, while being faster AND easier to install. Constantly rebuilding from source definetely doesn't brick

Planned Features

- Performance benchmarks
- Speculative decoding
- Better state management
- Lower level control of kv cache
- Speed from being one python layer above c++
- Multi-GPU












