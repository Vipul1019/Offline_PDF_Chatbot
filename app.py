import gradio as gr
from rag_pipeline import RAGPipeline
import config

# ── Global pipeline instance ──────────────────────────────────────────────────
rag = RAGPipeline()


# ── Helper: fetch Ollama models ───────────────────────────────────────────────

def refresh_models():
    models = rag.get_available_models()
    if not models:
        return gr.update(choices=[config.DEFAULT_MODEL], value=config.DEFAULT_MODEL)
    return gr.update(choices=models, value=models[0])


# ── PDF processing ────────────────────────────────────────────────────────────

def process_pdf(pdf_file, model_name):
    if pdf_file is None:
        return "Please upload a PDF file first."

    if not rag.is_ollama_running():
        return "Ollama is not running. Start it with: ollama serve"

    rag.set_model(model_name)

    try:
        num_chunks = rag.ingest_pdf(pdf_file)
        return (
            f"Ready! | {rag.loaded_pdf} | "
            f"{num_chunks} chunks | Model: {model_name}"
        )
    except Exception as e:
        return f"Error processing PDF:\n{e}"


# ── Chat (streaming) — Gradio 6 message format ────────────────────────────────
# history is a list of {"role": "user"/"assistant", "content": "..."}

def chat(message, history, model_name):
    message = message.strip()
    if not message:
        yield history, ""
        return

    if not rag.is_ollama_running():
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Ollama is not running. Please start it with: ollama serve"},
        ]
        yield history, ""
        return

    rag.set_model(model_name)

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]

    for token in rag.stream_query(message):
        history[-1]["content"] += token
        yield history, ""


# ── Clear helpers ─────────────────────────────────────────────────────────────

def clear_everything():
    rag.clear()
    return [], "", "All data cleared. Upload a new PDF to start fresh."

def clear_chat():
    return [], ""


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Offline PDF Chatbot") as demo:

    gr.Markdown(
        "# Offline PDF Chatbot\n"
        "Chat with your PDF documents using a **local LLM via Ollama** — fully offline."
    )

    with gr.Row():

        # Left sidebar
        with gr.Column(scale=1):

            gr.Markdown("### Model")
            model_dropdown = gr.Dropdown(
                choices=rag.get_available_models() or [config.DEFAULT_MODEL],
                value=(rag.get_available_models() or [config.DEFAULT_MODEL])[0],
                label="Ollama Model",
                interactive=True,
            )
            refresh_btn = gr.Button("Refresh model list", size="sm")

            gr.Markdown("### Document")
            pdf_upload = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                file_count="single",
            )
            process_btn = gr.Button("Process PDF", variant="primary")
            status_box = gr.Textbox(
                label="Status",
                value="No document loaded.",
                interactive=False,
                lines=2,
            )

            gr.Markdown("### Actions")
            with gr.Row():
                clear_chat_btn = gr.Button("Clear Chat", size="sm")
                clear_all_btn  = gr.Button("Clear All Data", size="sm", variant="stop")

            gr.Markdown(
                "---\n"
                "**Quick start:**\n"
                "1. Pull a model: `ollama pull llama3.2`\n"
                "2. Upload your PDF and click Process\n"
                "3. Start chatting!"
            )

        # Chat area
        with gr.Column(scale=3):
            gr.Markdown("### Chat")
            chatbot = gr.Chatbot(
                label="",
                height=500,
                buttons=["copy"],
            )
            msg_input = gr.Textbox(
                placeholder="Ask a question about your PDF...  (Enter to send)",
                label="Your question",
                lines=2,
            )
            send_btn = gr.Button("Send", variant="primary")

    # Event wiring
    refresh_btn.click(fn=refresh_models, outputs=model_dropdown)

    process_btn.click(
        fn=process_pdf,
        inputs=[pdf_upload, model_dropdown],
        outputs=status_box,
    )

    send_btn.click(
        fn=chat,
        inputs=[msg_input, chatbot, model_dropdown],
        outputs=[chatbot, msg_input],
    )

    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot, model_dropdown],
        outputs=[chatbot, msg_input],
    )

    clear_chat_btn.click(fn=clear_chat, outputs=[chatbot, msg_input])
    clear_all_btn.click(fn=clear_everything, outputs=[chatbot, msg_input, status_box])


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Offline PDF Chatbot")
    print("  Opening at: http://127.0.0.1:7860")
    print("=" * 50 + "\n")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
