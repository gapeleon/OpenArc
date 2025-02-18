import gradio as gr
import requests
import json



OPENARC_URL = "http://localhost:8000"

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
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}

    def unload_model(self):
        """
        Sends an unload model request to the API
        """
        try:
            response = requests.delete(
                f"{OPENARC_URL}/model/unload"
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}

    def generate_text(self, message, history):
        """
        Sends a text generation request to the API
        
        Args:
            message (str): The current message from the user
            history (list): List of previous message pairs
        """
        # Convert the chat history into the expected conversation format
        conversation = []
        
        # Add history messages
        for user_msg, assistant_msg in history:
            conversation.extend([
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ])
        
        # Add the current message
        conversation.append({"role": "user", "content": message})
        
        # Construct payload with actual values from generation config
        payload = {
            "conversation": conversation,
            **self.generation_config  # Spread the actual values
        }
        
        # Remove None values from payload
        payload = {k: v for k, v in payload.items() if v is not None}
        
        try:
            response = requests.post(
                f"{OPENARC_URL}/generate/text",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload)
            )
            response.raise_for_status()
            return response.json().get('response', "Error: No response received")
        except requests.exceptions.RequestException as e:
            return f"Error: {str(e)}"

    def status(self):
        """
        Checks the server status
        """
        try:
            response = requests.get(
                f"{OPENARC_URL}/status"
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}


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

    def chat_tab(self):
        with gr.Tab("Chat"):
            with gr.Row():
                # Chat interface on the left
                with gr.Column(scale=3):
                    self.chat_interface = gr.ChatInterface(
                        fn=self.payload_constructor.generate_text,
                        type="messages",
                    )
                
                # Accordions on the right
                with gr.Column(scale=1):
                    self.setup_generation_config()
                    
                    with gr.Accordion("Accordion 2", open=False):
                        gr.Markdown("This is the content of Accordion 2.")
                        slider2 = gr.Slider(minimum=0, maximum=100, label="Slider 2")

    def openarc_loader(self):
        with gr.Tab("OpenArc Loader"):
            # Center the content with controlled width
            with gr.Column(min_width=600):
                gr.Markdown("""
                ### Model Configuration
                Configure basic model loading parameters
                """)
                
                # Text input for model identifier
                id_model = gr.Textbox(
                    label="Model Identifier or Path",
                    placeholder="Enter model identifier or local path",
                    info="Enter the model's Hugging Face identifier or local path"
                )
                
                # Device selection dropdown
                device = gr.Dropdown(
                    choices=["AUTO", "CPU", "GPU.0", "GPU.1", "GPU.2", "AUTO:GPU.0,GPU.1", "AUTO:GPU.0,GPU.1,GPU.2"],
                    label="Device",
                    value="",
                    info="Select the device for model inference"
                )
                
                # Checkboxes for boolean options
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
         
                # OpenVINO configuration inputs
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

                # Add buttons in a row
                with gr.Row():
                    load_button = gr.Button("Load Model")
                    unload_button = gr.Button("Unload Model")
                    status_button = gr.Button("Check Status")

                # Add a result area
                result = gr.JSON(label="Result")

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
                    outputs=result
                )
                
                unload_button.click(
                    fn=self.payload_constructor.unload_model,
                    inputs=None,
                    outputs=result
                )

                status_button.click(
                    fn=self.payload_constructor.status,
                    inputs=None,
                    outputs=result
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
