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

## Usage

OpenArc offers a lightweight approach to decoupling machine learning code from application logic by adding an API layer to serve LLMs using hardware acceleration for Intel devices; in practice OpenArc offers a similar workflow to what's possible with Ollama, LM-Studio or OpenRouter. 


OpenArc emphasizes application level control over the context window. Exposing the conversation parameter from Transformers apply_chat_template method grant's complete control over what get's passed in and out of the model without any intervention required at the template level for text. I have not added the other parameters since I am interested in defining tool calls in other ways, potentially outside of the chat temple, such as the design philosophy of CodeAgent from smolagents where LLMs define their own tooling dynamically.


Since I started working with OpenVINO every implementation worth tinkering with has come from ad hoc experimentation; for example, no other projects support multi gpu natively; once I learned this I did not bother with any other projects. Moreover, the excellent openvino-notebooks repo maintains pace with SOTA. Back when Qwen2-VL came out the released notebook helped me implement a class barely a week after the model released which was not 





Use parameters defined in the [Pydanitc models here](https://github.com/SearchSavior/OpenArc/blob/main/src/engine/optimum_inference_core.py) to build a frontend and construct a request body based on inputs. You can even dump the pydantic models into a prompt and get strong boilerplate for your usecase since each BaseModel directly defines each type of request type. 




In practice 


## Converted Models

Find over 40+ preconverted text generation models of larger variety that what's offered by the official repos for OpenVINO on my [HuggingFace](https://huggingface.co/Echo9Zulu).

You can easily craft conversion commands using my HF Space, [Optimum-CLI-Tool_tool](https://huggingface.co/spaces/Echo9Zulu/Optimum-CLI-Tool_tool)!

 
| Models                                                                                                                                      | Compressed Weights |
| ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| [Ministral-3b-instruct-int4_asym-ov](https://huggingface.co/Echo9Zulu/Ministral-3b-instruct-int4_asym-ov)                                   | 1.85 GB            |
| [Hermes-3-Llama-3.2-3B-awq-ov](https://huggingface.co/Echo9Zulu/Hermes-3-Llama-3.2-3B-awq-ov)							| 1.8 GB |
[Llama-3.1-Tulu-3-8B-int4_asym-ov](https://huggingface.co/Echo9Zulu/Llama-3.1-Tulu-3-8B-int4_asym-ov/tree/main)                             | 4.68 GB            |
| [Falcon3-10B-Instruct-int4_asym-ov](https://huggingface.co/Echo9Zulu/Falcon3-10B-Instruct-int4_asym-ov)                                     | 5.74 GB            |
| [DeepSeek-R1-Distill-Qwen-14B-int4-awq-ov](https://huggingface.co/Echo9Zulu/DeepSeek-R1-Distill-Qwen-14B-int4-awq-ov/tree/main)             | 7.68 GB            |
| [Phi-4-o1-int4_asym-awq-weight_quantization_error-ov](https://huggingface.co/Echo9Zulu/Phi-4-o1-int4_asym-awq-weight_quantization_error-ov) | 8.11 GB            |
| [Mistral-Small-24B-Instruct-2501-int4_asym-ov](https://huggingface.co/Echo9Zulu/Mistral-Small-24B-Instruct-2501-int4_asym-ov)		| 12.9 GB	     |	
| [Qwen2.5-72B-Instruct-int4-ns-ov](https://huggingface.co/Echo9Zulu/Qwen2.5-72B-Instruct-int4-ns-ov/tree/main))                              | 39 GB              |


Learn more about model conversion and working with Intel Devices generally in the openvino_utilities notebook.



## Roadmap

Currently my plan is to transition from Optimum to OpenVINO GenAI since it handles everything here and much more, while being faster AND easier to install. 



## Resources
---
to learn more about how to leverage your Intel devices for Machine Learning:













