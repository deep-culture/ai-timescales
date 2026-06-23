# EigenzAIt

A dual-model comparison UI that streams token generation from an autoregressive and a diffusion language model (LLaDA) side by side, with real-time TTS vocalization.

<em>Click for audio</em>
<a href="https://deep-culture.org/eigenzait/ar-lm-inference.mp4" target="_blank"><img width="1274" height="880" alt="ar-lm-generation" src="https://github.com/user-attachments/assets/89fa4f29-3aa6-460f-89d8-7744b6775be5" /></a>

<a href="https://deep-culture.org/eigenzait/dlm-inference.mp4" target="_blank"><img width="1282" height="882" alt="dlm-generation" src="https://github.com/user-attachments/assets/80e00290-9ff9-4678-b069-6d9679b702c4" /></a>


## Requirements

- Python 3.10–3.12
- Node.js 18+ and npm
- A HuggingFace token with access to [LLaMA 3.1 9B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) for best comparison with the diffusion model (LLaDa 8B-Instruct), or another AR-LM of your choosing.

---

## Installation

### 1. Python dependencies

Check the right PyTorch build for your system at [pytorch.org](https://pytorch.org/get-started/locally/), install it first, then:

```bash
pip install -r requirements.txt
```

### 2. Frontend dependencies

```bash
cd frontend
npm install
```
If you need to upgrade Node.js, do

```bash
curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.33.8/install.sh | bash
nvm install node
````

### 3. Environment variables

Create `.env`:

```env
HF_TOKEN=hf_...          # HuggingFace token
API_KEY=your_secret      # shared secret between frontend proxy and backend
# Optional — skip loading a model:
# DO_NOT_LOAD_AR=true
# DO_NOT_LOAD_DIFFUSION=true
```

Copy `.env-example` into `.env` and set the credentials.

You'll need a HuggingFace token with access to the [LLaMA 3.2](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) model. Request access on the model page if you don't have it yet.

Set `LOGIN_USER` and `LOGIN_PASSWORD` in `.env` if you want to enable the optional login gate.

---

## Running

### 1. Start the backend

```bash
cd backend
uvicorn backend:app --host 127.0.0.1 --port 8000 --workers 1
```

### 2. Start the frontend dev server

```bash
cd frontend
npm run dev
```

Then open **http://localhost:8080** in your browser.

The Vite dev server proxies all `/api/*` requests to the backend at `http://127.0.0.1:8000`, so no CORS configuration is needed.

### Production build (optional)

```bash
cd frontend
npm run build      # outputs to frontend/dist/
npm run preview    # serves the built output locally
```
