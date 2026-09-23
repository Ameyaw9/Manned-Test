import express from "express"
import path from "node:path"
import { fileURLToPath } from "node:url"

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const app = express()
const port = Number(process.env.PORT || 8080)
const model = process.env.QWEN_MODEL || "qwen-plus"
const baseUrl = (process.env.QWEN_BASE_URL || "https://dashscope.aliyuncs.com/compatible-mode/v1").replace(/\/$/, "")
const apiKey = process.env.QWEN_API_KEY || process.env.DASHSCOPE_API_KEY
const systemPrompt = process.env.SYSTEM_PROMPT || "You are manned, a thoughtful space companion for interstellar travel and space questions. Be clear, warm, concise, scientifically grounded, and honest about uncertainty. Keep the conversation focused on space when appropriate."

function configurationError() {
  return "Qwen is not connected yet. Set QWEN_API_KEY (or DASHSCOPE_API_KEY) in the environment, then restart the Node server."
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
    const data = await response.json()
    if (!response.ok) return res.status(response.status).json({ error: data?.error?.message || "Qwen request failed" })
    res.json({ reply: data.choices?.[0]?.message?.content?.trim() || "" })
  } catch (error) {
    console.error("[manned] Qwen request failed", error)
    res.status(502).json({ error: "Unable to reach Qwen" })
  }
})

app.get("/{*splat}", (_req, res) => res.sendFile(path.join(__dirname, "static", "index.html")))
app.listen(port, "0.0.0.0", () => console.log(`[manned] listening on ${port}`))

process.on("SIGTERM", () => process.exit(0))
process.on("SIGINT", () => process.exit(0))

export default app
