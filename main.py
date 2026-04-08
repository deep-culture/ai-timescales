import sounddevice as sd
import queue
import threading
import torch
from LLaDA.generate import generate
from LLaDA.patch_llada import patch_model
from transformers import AutoTokenizer, AutoModel
from kokoro import KPipeline

pipeline = KPipeline(lang_code='a')
word_queue = queue.Queue()

def tts_worker(wait=False):
    """Background thread for Kokoro + sounddevice"""
    while True:
        word = word_queue.get()

        if word is None:  # Poison pill to stop thread cleanly
            break

        # Generate audio for the single word (sped up slightly)
        generator = pipeline(word, voice='af_heart', speed=1.2)

        for _, _, audio_array in generator:
            # Kokoro outputs numpy arrays at 24kHz.
            # sounddevice plays it instantly.
            sd.play(audio_array, samplerate=24000)

            # Block this background thread until the word finishes playing
            # so words don't overlap, while the main transformer loop keeps running.
            if wait:
                sd.wait()

        word_queue.task_done()

# DIFFUSION LANGUAGE MODEL
text_buffer = ""

def sonify_tokens(tokens, attention=None):
    # tokens: int tensor of shape [batch, seq_len]
    token_ids = tokens.flatten().tolist()
    for token_id in token_ids:
        new_token = tokenizer.decode([token_id], skip_special_tokens=False)
        word_queue.put(new_token.strip())


def load_dllm(model_id='GSAI-ML/LLaDA-8B-Instruct'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation='eager'  # important
    ).to(device).eval()

    # This is important--patch the model so it actually returns attention weights
    model = patch_model(model)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    return model, tokenizer, device


def chat_dllm(model, tokenizer, device, gen_length=32, steps=4):
    conversation_num = 0
    while True:
        user_input = input("Enter your question: ")

        m = [{"role": "user", "content": user_input}]

        user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(user_input)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        if conversation_num == 0:
            prompt = input_ids
        else:
            prompt = torch.cat([prompt, input_ids[:, 1:]], dim=1)

        print("Generating...")
        x, step_tokens, step_attentions = generate(
            model, prompt, steps=steps, gen_length=gen_length,
            block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence'
        )

        # --- Per-step inspection ---
        prompt_len = prompt.shape[1]
        for step_idx, (tokens, attn) in enumerate(zip(step_tokens, step_attentions)):
            # Decode only the generated portion; MASK token (126336) won't decode to readable text
            tokens_decoded = tokenizer.batch_decode(tokens[:, prompt_len:], skip_special_tokens=False)[0]
            print(f"  Step {step_idx + 1} tokens : {tokens_decoded}")
            print(f"  Step {step_idx + 1} attn   : shape={tuple(attn.shape)}, mean={attn.mean():.4f}")

            sonify_tokens(tokens)

        answer = tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)[0]
        print(f"Bot's reply: {answer}")

        # remove the <EOS>
        prompt = x[x != 126081].unsqueeze(0)
        conversation_num += 1
        print('-----------------------------------------------------------------------')


if __name__ == "__main__":
    threading.Thread(target=tts_worker, daemon=True).start()

    word_queue.put("Hello!")

    word_queue.join()  # Block main thread until all remaining words are spoken
    word_queue.put(None)  # Send poison pill to close the background thread safely
    #model, tokenizer, device = load_dllm()
    #chat_dllm(model, tokenizer, device)