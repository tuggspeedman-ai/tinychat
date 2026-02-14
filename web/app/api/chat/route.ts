export async function POST(request: Request) {
  const { messages, temperature, max_tokens, top_k } = await request.json();

  const apiUrl = process.env.MODAL_API_URL;
  const apiKey = process.env.MODAL_API_KEY;

  if (!apiUrl) {
    return new Response(
      JSON.stringify({ error: "MODAL_API_URL not configured" }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }

  const body: Record<string, unknown> = {
    messages,
    temperature: temperature ?? 0.4,
    max_tokens: max_tokens ?? 512,
    top_k: top_k ?? 40,
  };

  const response = await fetch(apiUrl, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const text = await response.text();
    return new Response(
      JSON.stringify({ error: `Backend error: ${response.status}`, detail: text }),
      { status: response.status, headers: { "Content-Type": "application/json" } }
    );
  }

  return new Response(response.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
