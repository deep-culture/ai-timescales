import torch

from generate import generate
from transformers import AutoTokenizer, AutoModel
from patch_llada import patch_model


def chat():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModel.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation='eager'
    ).to(device).eval()
    model = patch_model(model)
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    gen_length = 32
    steps = 4
    print('*' * 66)
    print(f'**  Answer Length: {gen_length}  |  Sampling Steps: {steps}  **')
    print('*' * 66)

    conversation_num = 0
    while True:
        user_input = input("Enter your question: ")

        m = [{"role": "user", "content": user_input}]
        print("Tokenizing")
        user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(user_input)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        if conversation_num == 0:
            prompt = input_ids
        else:
            prompt = torch.cat([prompt, input_ids[:, 1:]], dim=1)

        print("Generating")
        x, step_tokens, step_attentions = generate(
            model, prompt, steps=steps, gen_length=gen_length,
            block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence'
        )

        # --- Per-step inspection ---
        prompt_len = prompt.shape[1]
        for step_idx, (tokens, attn) in enumerate(zip(step_tokens, step_attentions)):
            # Decode only the generated portion; MASK token (126336) won't decode to readable text
            decoded = tokenizer.batch_decode(tokens[:, prompt_len:], skip_special_tokens=False)[0]
            print(f"  Step {step_idx + 1} tokens : {decoded}")
            print(f"  Step {step_idx + 1} attn   : shape={tuple(attn.shape)}, mean={attn.mean():.4f}")

        answer = tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)[0]
        print(f"Bot's reply: {answer}")

        # remove the <EOS>
        prompt = x[x != 126081].unsqueeze(0)
        conversation_num += 1
        print('-----------------------------------------------------------------------')


if __name__ == "__main__":
    chat()
