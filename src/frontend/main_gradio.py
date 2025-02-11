import json
import requests
import gradio as gr

# URL of the FastAPI endpoint
API_URL = "http://localhost:8000/generate/text"

def send_chat(user_message, chat_history, conversation_state, config, last_assistant):
    """
    Send the new user message along with the conversation and config to the API.
    The API returns a full conversation context in its last assistant message.
    To avoid appending the whole context to the chat display, we compare the new
    assistant reply with the previously displayed one and extract only the new portion.
    """
    if conversation_state is None:
        conversation_state = []
    if chat_history is None:
        chat_history = []
    
    # Append the new user message to the conversation payload.
    conversation_state.append({"role": "user", "content": user_message})
    
    # Build payload using the current configuration.
    payload = {
        "conversation": conversation_state,
        "max_new_tokens": config.get("max_new_tokens", 128),
        "temperature": config.get("temperature", 0.7),
        "top_k": config.get("top_k", 50),
        "top_p": config.get("top_p", 0.9),
        "repetition_penalty": config.get("repetition_penalty", 1.0),
        "do_sample": config.get("do_sample", True),
        "num_return_sequences": config.get("num_return_sequences", 1),
        "pad_token_id": config.get("pad_token_id", None),
        "eos_token_id": config.get("eos_token_id", 2),
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        response_json = response.json()
        
        # Assume the API returns a "conversation" field with a list of messages.
        if "conversation" in response_json:
            full_assistant_message = response_json["conversation"][-1]["content"]
            # If we already have a previous assistant message and the new one starts with it,
            # then extract only the new part.
            if last_assistant and full_assistant_message.startswith(last_assistant):
                new_assistant_message = full_assistant_message[len(last_assistant):].strip()
            else:
                new_assistant_message = full_assistant_message
            # Update the conversation state with the API's returned conversation.
            conversation_state = response_json["conversation"]
        elif "generated_text" in response_json:
            # If the API returns a single generated text.
            full_assistant_message = response_json["generated_text"]
            if last_assistant and full_assistant_message.startswith(last_assistant):
                new_assistant_message = full_assistant_message[len(last_assistant):].strip()
            else:
                new_assistant_message = full_assistant_message
            conversation_state.append({"role": "assistant", "content": full_assistant_message})
        else:
            new_assistant_message = "Error: Unexpected response format."
            conversation_state.append({"role": "assistant", "content": new_assistant_message})
    except Exception as e:
        new_assistant_message = f"Error: {str(e)}"
        conversation_state.append({"role": "assistant", "content": new_assistant_message})
    
    # Append only the new (user, assistant) pair to the chat history.
    chat_history.append((user_message, new_assistant_message))
    
    # Update the last assistant state to the full assistant reply (for future diffing).
    last_assistant = full_assistant_message if 'full_assistant_message' in locals() else last_assistant
    
    return chat_history, chat_history, conversation_state, last_assistant

# --- Configuration update functions (each field has its own input and button) ---
def update_max_new_tokens(value, config):
    config = config or {}
    try:
        config["max_new_tokens"] = int(value)
        return config, f"max_new_tokens updated to {value}"
    except Exception as e:
        return config, f"Error: {str(e)}"

def update_temperature(value, config):
    config = config or {}
    try:
        config["temperature"] = float(value)
        return config, f"temperature updated to {value}"
    except Exception as e:
        return config, f"Error: {str(e)}"

def update_top_k(value, config):
    config = config or {}
    try:
        config["top_k"] = int(value)
        return config, f"top_k updated to {value}"
    except Exception as e:
        return config, f"Error: {str(e)}"

def update_top_p(value, config):
    config = config or {}
    try:
        config["top_p"] = float(value)
        return config, f"top_p updated to {value}"
    except Exception as e:
        return config, f"Error: {str(e)}"

def update_repetition_penalty(value, config):
    config = config or {}
    try:
        config["repetition_penalty"] = float(value)
        return config, f"repetition_penalty updated to {value}"
    except Exception as e:
        return config, f"Error: {str(e)}"

def update_do_sample(value, config):
    config = config or {}
    try:
        config["do_sample"] = bool(value)
        return config, f"do_sample updated to {value}"
    except Exception as e:
        return config, f"Error: {str(e)}"

def update_num_return_sequences(value, config):
    config = config or {}
    try:
        config["num_return_sequences"] = int(value)
        return config, f"num_return_sequences updated to {value}"
    except Exception as e:
        return config, f"Error: {str(e)}"

def update_pad_token_id(value, config):
    config = config or {}
    try:
        if value == "" or value.lower() == "none":
            config["pad_token_id"] = None
            out = "None"
        else:
            config["pad_token_id"] = int(value)
            out = value
        return config, f"pad_token_id updated to {out}"
    except Exception as e:
        return config, f"Error: {str(e)}"

def update_eos_token_id(value, config):
    config = config or {}
    try:
        config["eos_token_id"] = int(value)
        return config, f"eos_token_id updated to {value}"
    except Exception as e:
        return config, f"Error: {str(e)}"

# --- Default configuration ---
default_config = {
    "max_new_tokens": 128,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "do_sample": True,
    "num_return_sequences": 1,
    "pad_token_id": None,
    "eos_token_id": 2,
}

# --- Build the Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("## Chat Interface with Generation Configuration")
    
    with gr.Row():
        with gr.Column(scale=2):
            # Chatbot shows conversation as a list of (user, assistant) pairs.
            chatbot = gr.Chatbot(label="Chat History")
            with gr.Row():
                user_message = gr.Textbox(
                    show_label=True,
                    placeholder="Enter your message here",
                    label="Your Message"
                )
                send_button = gr.Button("Send")
        with gr.Column(scale=1):
            gr.Markdown("### Generation Configuration")
            with gr.Row():
                max_new_tokens_in = gr.Number(value=128, label="max_new_tokens")
                max_new_tokens_btn = gr.Button("Set max_new_tokens")
                max_new_tokens_status = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                temperature_in = gr.Slider(0.0, 2.0, value=0.7, step=0.01, label="temperature")
                temperature_btn = gr.Button("Set temperature")
                temperature_status = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                top_k_in = gr.Number(value=50, label="top_k")
                top_k_btn = gr.Button("Set top_k")
                top_k_status = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                top_p_in = gr.Slider(0.0, 1.0, value=0.9, step=0.01, label="top_p")
                top_p_btn = gr.Button("Set top_p")
                top_p_status = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                repetition_penalty_in = gr.Number(value=1.0, label="repetition_penalty")
                repetition_penalty_btn = gr.Button("Set repetition_penalty")
                repetition_penalty_status = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                do_sample_in = gr.Checkbox(value=True, label="do_sample")
                do_sample_btn = gr.Button("Set do_sample")
                do_sample_status = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                num_return_sequences_in = gr.Number(value=1, label="num_return_sequences")
                num_return_sequences_btn = gr.Button("Set num_return_sequences")
                num_return_sequences_status = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                pad_token_id_in = gr.Textbox(value="", label="pad_token_id (leave empty for None)")
                pad_token_id_btn = gr.Button("Set pad_token_id")
                pad_token_id_status = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                eos_token_id_in = gr.Number(value=2, label="eos_token_id")
                eos_token_id_btn = gr.Button("Set eos_token_id")
                eos_token_id_status = gr.Textbox(label="Status", interactive=False)
    
    # --- Hidden states ---
    # chat_history_state: list of (user, assistant) pairs to display in the Chatbot widget.
    # conversation_state: the full conversation (list of message dicts) sent to the API.
    # config_state: holds the generation configuration.
    # last_assistant_state: holds the last full assistant message (for diffing purposes).
    chat_history_state = gr.State([])
    conversation_state = gr.State([])
    config_state = gr.State(default_config)
    last_assistant_state = gr.State("")  # initially empty
    
    # --- Wire up the Send button ---
    send_button.click(
        fn=send_chat,
        inputs=[user_message, chat_history_state, conversation_state, config_state, last_assistant_state],
        outputs=[chatbot, chat_history_state, conversation_state, last_assistant_state],
        queue=True,
    )
    
    # --- Wire up configuration update buttons ---
    max_new_tokens_btn.click(
        fn=update_max_new_tokens,
        inputs=[max_new_tokens_in, config_state],
        outputs=[config_state, max_new_tokens_status]
    )
    temperature_btn.click(
        fn=update_temperature,
        inputs=[temperature_in, config_state],
        outputs=[config_state, temperature_status]
    )
    top_k_btn.click(
        fn=update_top_k,
        inputs=[top_k_in, config_state],
        outputs=[config_state, top_k_status]
    )
    top_p_btn.click(
        fn=update_top_p,
        inputs=[top_p_in, config_state],
        outputs=[config_state, top_p_status]
    )
    repetition_penalty_btn.click(
        fn=update_repetition_penalty,
        inputs=[repetition_penalty_in, config_state],
        outputs=[config_state, repetition_penalty_status]
    )
    do_sample_btn.click(
        fn=update_do_sample,
        inputs=[do_sample_in, config_state],
        outputs=[config_state, do_sample_status]
    )
    num_return_sequences_btn.click(
        fn=update_num_return_sequences,
        inputs=[num_return_sequences_in, config_state],
        outputs=[config_state, num_return_sequences_status]
    )
    pad_token_id_btn.click(
        fn=update_pad_token_id,
        inputs=[pad_token_id_in, config_state],
        outputs=[config_state, pad_token_id_status]
    )
    eos_token_id_btn.click(
        fn=update_eos_token_id,
        inputs=[eos_token_id_in, config_state],
        outputs=[config_state, eos_token_id_status]
    )

if __name__ == "__main__":
    demo.launch()
