import gradio as gr
import requests
import json



OPENARC_URL =  "http://localhost:8000"

class Payload_Constructor:
    def __init__(self):
        self.generation_config = {}  # This will store actual values, not components

    def load_model(self, id_model, device, use_cache, export_model, num_streams, performance_hint, precision_hint):
        """
        Constructs and sends the load model request based on UI inputs
        
        Args:
            id_model (str): Model identifier or path
            device (str): Device selection for inference
            use_cache (bool): Whether to use cache
            export_model (bool): Whether to export the model
            num_streams (str): Number of inference streams
            performance_hint (str): Performance optimization strategy
            precision_hint (str): Model precision for computation
        """

        # Construct the payload matching the API format
        payload = {
            "load_config": {
                "id_model": id_model,
                "use_cache": use_cache,
                "device": device,
                "export_model": export_model
            },
            "ov_config": {
                "NUM_STREAMS": num_streams,
                "PERFORMANCE_HINT": performance_hint,
      #         "PRECISION_HINT": precision_hint
            }
        }

        # Clean up empty values
        ov_config = payload["ov_config"]
        ov_config = {k: v for k, v in ov_config.items() if v}
        if not ov_config:
            del payload["ov_config"]
        
        try:
            response = requests.post(
                f"{OPENARC_URL}/model/load",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload)
            )
            response.raise_for_status()
            return response.json(), f"Model loaded successfully: {response.json()}"
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}, f"Error loading model: {str(e)}"

    def unload_model(self):
        """
        Sends an unload model request to the API
        """
        try:
            response = requests.delete(
                f"{OPENARC_URL}/model/unload"
            )
            response.raise_for_status()
            return response.json(), f"Model unloaded successfully: {response.json()}"
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}, f"Error unloading model: {str(e)}"

    def generate_text(self, message, history):
        """
        Sends a text generation request to the API
        
        Args:
            message (str): The current message from the user
            history (list): List of previous messages from Gradio
        """
        # Convert Gradio history format to the API format
        conversation = []
        for item in history:
            # Each item in history is a message dict with 'role' and 'content'
            conversation.append(item)
        
        # Add the current message
        conversation.append({"role": "user", "content": message})
        
        # Construct payload with actual values from generation config
        payload = {
            "conversation": conversation,
            **self.generation_config
        }
        
        try:
            response = requests.post(
                f"{OPENARC_URL}/generate/text",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload)
            )
            response.raise_for_status()
            response_data = response.json()
            
            # Store the performance metrics
            if 'performance_metrics' in response_data:
                self.update_performance_metrics(response_data['performance_metrics'])
            
            # Return the complete assistant message object
            return response_data.get('generated_text', {})
        except requests.exceptions.RequestException as e:
            return {"role": "assistant", "content": f"Error: {str(e)}"}

    def status(self):
        """
        Checks the server status
        """
        try:
            response = requests.get(
                f"{OPENARC_URL}/status"
            )
            response.raise_for_status()
            return response.json(), f"Server status: {response.json()}"
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}, f"Error checking server status: {str(e)}"


class ChatUI:
    def __init__(self):
        self.demo = None
        self.chat_interface = None
        self.generation_config_components = {}  # Rename to clarify these are components
        self.payload_constructor = Payload_Constructor()
        self.setup_interface()

    def setup_generation_config(self):
        with gr.Accordion("**Generation Config**", open=False):
            gr.Markdown("")
            # Store the Gradio components
            self.generation_config_components.update({
                'max_new_tokens': gr.Slider(minimum=0, maximum=100, label="Maximum number of tokens to generate"),
                'temperature': gr.Slider(minimum=0, maximum=100, label="Temperature"),
                'top_k': gr.Slider(minimum=0, maximum=100, label="Top-k"),
                'top_p': gr.Slider(minimum=0, maximum=100, label="Top-p"),
                'repetition_penalty': gr.Slider(minimum=0, maximum=100, label="Repetition penalty"),
                'do_sample': gr.Checkbox(label="Do sample"),
                'num_return_sequences': gr.Slider(minimum=0, maximum=100, label="Number of sequences to return"),
                'pad_token_id': gr.Slider(minimum=0, maximum=100, label="pad token ID"),
                'eos_token_id': gr.Slider(minimum=0, maximum=100, label="eos token ID")
            })
            
            # Set up event handlers to update the payload constructor when values change
            for key, component in self.generation_config_components.items():
                component.change(
                    fn=lambda value, k=key: self.update_generation_config(k, value),
                    inputs=[component]
                )

    def update_generation_config(self, key, value):
        """Update the generation config in the payload constructor with actual values"""
        self.payload_constructor.generation_config[key] = value

    def setup_performance_metrics(self):
        with gr.Accordion("Performance Metrics", open=True):
            self.performance_metrics = gr.JSON(
                label="Latest Generation Metrics",
                value={
                    "generation_time": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "new_tokens": None,
                    "eval_time": None,
                    "tokens_per_second": None
                }
            )

    def chat_tab(self):
        with gr.Tab("Chat"):
            with gr.Row():
                # Chat interface on the left
                with gr.Column(scale=3):
                    self.chat_interface = gr.ChatInterface(
                        fn=lambda msg, history: self.handle_chat_response(msg, history),
                        type="messages",
                    )
                
                # Accordions on the right
                with gr.Column(scale=1):
                    self.setup_generation_config()
                    self.setup_performance_metrics()

    def handle_chat_response(self, msg, history):
        # Get the response from the model
        response = self.payload_constructor.generate_text(msg, history)
        
        # Update the performance metrics display
        if hasattr(self.payload_constructor, 'performance_metrics'):
            self.performance_metrics.value = self.payload_constructor.performance_metrics
        
        return response

    def openarc_loader(self):
        with gr.Tab("OpenArc Loader"):
            with gr.Row():
                with gr.Column(min_width=500, scale=1):
                    
                    id_model = gr.Textbox(
                        label="Model Identifier or Path",
                        placeholder="Enter model identifier or local path",
                        info="Enter the model's Hugging Face identifier or local path"
                    )
                    
                    device = gr.Dropdown(
                        choices=["AUTO", "CPU", "GPU.0", "GPU.1", "GPU.2", "AUTO:GPU.0,GPU.1", "AUTO:GPU.0,GPU.1,GPU.2"],
                        label="Device",
                        value="",
                        info="Select the device for model inference"
                    )
                    
                    use_cache = gr.Checkbox(
                        label="Use Cache",
                        value=True,
                        info="Enable cache for stateful models (disable for multi-GPU)"
                    )
                    
                    export_model = gr.Checkbox(
                        label="Export Model",
                        value=False,
                        info="Whether to export the model"
                    )
             
                    num_streams = gr.Textbox(
                        label="Number of Streams",
                        value="",
                        placeholder="Leave empty for default",
                        info="Number of inference streams (optional)"
                    )
                    
                    performance_hint = gr.Dropdown(
                        choices=["", "LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT"],
                        label="Performance Hint",
                        value="",
                        info="Select performance optimization strategy"
                    )
                    
                    precision_hint = gr.Dropdown(
                        choices=["", "auto", "fp32", "fp16", "int8"],
                        label="Precision Hint",
                        value="",
                        info="Select model precision for computation"
                    )

                    with gr.Row():
                        load_button = gr.Button("Load Model")
                        unload_button = gr.Button("Unload Model")
                        status_button = gr.Button("Check Status")

                # Second column (new)
                with gr.Column(min_width=300, scale=1):
                    with gr.Accordion("Debug Log", open=True):
                        debug_log = gr.JSON(
                            label="Log Output",
                            value={"message": "Debug information will appear here..."},
                        )

            # Connect the buttons to their respective methods
            load_button.click(
                fn=self.payload_constructor.load_model,
                inputs=[
                    id_model,
                    device,
                    use_cache,
                    export_model,
                    num_streams,
                    performance_hint,
                    precision_hint
                ],
                outputs=[debug_log]  # Add debug_log to outputs
            )
            
            unload_button.click(
                fn=self.payload_constructor.unload_model,
                inputs=None,
                outputs=[debug_log]  # Add debug_log to outputs
            )

            status_button.click(
                fn=self.payload_constructor.status,
                inputs=None,
                outputs=[debug_log]  # Add debug_log to outputs
            )

    def setup_interface(self):
        with gr.Blocks() as self.demo:
            with gr.Tabs():
                self.chat_tab()
                self.openarc_loader()

    def launch(self):
        self.demo.launch()

if __name__ == "__main__":
    app = ChatUI()
    app.launch()
