import express from "express"
import path from "node:path"
import { fileURLToPath } from "node:url"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const app = express()
const port = Number(process.env.PORT || 8080)
const model = "openrouter/free"
const baseUrl = (process.env.OPENROUTER_BASE_URL || "https://openrouter.ai/api/v1").replace(/\/$/, "")
const apiKey = (process.env.OPENROUTER_API_KEY || process.env.QWEN_API_KEY || "")
  .trim()
  .replace(/^['"]|['"]$/g, "")
const systemPrompt = process.env.SYSTEM_PROMPT || "You are manned, a thoughtful space companion for interstellar travel and space questions. Be clear, warm, concise, scientifically grounded, and honest about uncertainty. Keep the conversation focused on space when appropriate."

function configurationError() {
  return "Your space companion is not connected yet. Set an active OpenRouter API key as OPENROUTER_API_KEY, then restart the Node server."
}

app.use(express.json({ limit: "1mb" }))
const staticDir = path.join(__dirname, "static")
app.use(express.static(staticDir))
app.get("/", (_req, res) => res.sendFile(path.join(staticDir, "index.html")))

app.get("/health", (_req, res) => res.json({ status: "ok", model }))

app.post("/api/chat", async (req, res) => {
  const messages = Array.isArray(req.body?.messages) ? req.body.messages : []
  if (!messages.length) return res.status(400).json({ error: "messages must not be empty" })
  if (!apiKey) return res.status(503).json({ error: configurationError() })

  try {
    const response = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
      signal: AbortSignal.timeout(45000),
      body: JSON.stringify({
        model,
        messages: [{ role: "system", content: systemPrompt }, ...messages],
        temperature: 0.7,
        top_p: 0.9,
        max_tokens: Math.min(Number(req.body?.max_tokens || 512), 2048),
      }),
    })
    const data = await response.json().catch(() => ({}))
    const providerError = data?.error
    const providerErrorMessage = typeof providerError === "string"
      ? providerError
      : providerError?.message || providerError?.code || data?.message || "OpenRouter request failed"
    if (!response.ok) {
      if (response.status === 401 || response.status === 403) {
        return res.status(502).json({ error: "OpenRouter rejected the API key. Use an active OpenRouter key, save it as OPENROUTER_API_KEY, and restart the Node server." })
      }
      return res.status(response.status).json({ error: String(providerErrorMessage) })
    }
    const content = data?.choices?.[0]?.message?.content
    const reply = Array.isArray(content)
      ? content.map((part) => typeof part === "string" ? part : part?.text || "").join("").trim()
      : typeof content === "string" ? content.trim() : content?.text ? String(content.text).trim() : ""
    if (!reply) return res.status(502).json({ error: "OpenRouter returned an empty response. Check the selected model and API account." })
    res.json({ reply })
  } catch (error) {
    console.error("[manned] OpenRouter request failed", error)
    res.status(502).json({ error: "Unable to reach OpenRouter. Check the API key, model availability, and network connection." })
  }
})

app.get("/{*splat}", (_req, res) => res.sendFile(path.join(__dirname, "static", "index.html")))
if (!process.env.VERCEL) {
  app.listen(port, "0.0.0.0", () => console.log(`[manned] listening on ${port}`))

  process.on("SIGTERM", () => process.exit(0))
  process.on("SIGINT", () => process.exit(0))
}

export default app
