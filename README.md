# Manned_T2 — your space companion

Manned is a lightweight Node.js chat application for interstellar travel, spacecraft, astronomy, and space questions. It serves the browser UI and a small OpenAI-compatible API proxy for Qwen (or another compatible provider).

## Run locally

```bash
npm install
QWEN_API_KEY=your-key npm start
```

Open `http://localhost:8080`.

Optional configuration:

- `PORT` — server port, defaults to `8080`
- `QWEN_API_KEY` — API key for the configured provider
- `QWEN_MODEL` — model name, defaults to `qwen-plus`
- `QWEN_BASE_URL` — OpenAI-compatible API base URL, defaults to DashScope
- `SYSTEM_PROMPT` — override Manned's space companion instructions

The API is:

- `GET /health`
- `POST /api/chat` with `{ "messages": [{ "role": "user", "content": "..." }] }`

## Deploy to Google Cloud Run

This is a Node.js service using the included Dockerfile.

```bash
gcloud builds submit --tag gcr.io/$GOOGLE_CLOUD_PROJECT/manned-t2
gcloud run deploy manned-t2 \
  --image gcr.io/$GOOGLE_CLOUD_PROJECT/manned-t2 \
  --region us-central1 \
  --set-env-vars QWEN_MODEL=qwen-plus \
  --set-env-vars QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --set-env-vars QWEN_API_KEY=your-key \
  --allow-unauthenticated
```

Cloud Run supplies `PORT`; the server listens on `0.0.0.0` and is ready for the container health check.

## Adding other models later

The server already uses an OpenAI-compatible request shape. To switch providers without changing the UI, set `QWEN_BASE_URL`, `QWEN_MODEL`, and the matching API key environment variable wiring in `server.js`. A future provider adapter can be added behind the same `/api/chat` route without changing the frontend.

## Project structure

```text
server.js          Node.js + Express server and chat API
static/index.html  Manned browser interface
Dockerfile         Cloud Run container definition
package.json       Node dependencies and scripts
```

Never commit API keys. Use Cloud Run Secret Manager integration for production secrets rather than putting credentials in source control.

## Scripts

```bash
npm start   # production server
npm run dev # Node watch mode
```

The project uses Node.js 20 or newer and Express 5.
