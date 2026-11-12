const API_BASE = '/api/v1'

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options)
  if (!response.ok) {
    let detail = `Request failed (${response.status})`
    try {
      const payload = await response.json()
      if (payload?.detail) detail = payload.detail
    } catch {
      // keep HTTP status fallback
    }
    throw new Error(detail)
  }
  if (response.status === 204) return null
  return response.json()
}

export function fetchHealth() {
  return requestJson('/health')
}

export function listDocuments() {
  return requestJson('/documents')
}

export function getDocument(documentId) {
  return requestJson(`/documents/${documentId}`)
}

export function getDocumentStatus(documentId) {
  return requestJson(`/documents/${documentId}/status`)
}

export function getDocumentFileUrl(documentId) {
  return `${API_BASE}/documents/${documentId}/file`
}

export async function uploadDocument(file) {
  const formData = new FormData()
  formData.append('file', file)
  return requestJson('/documents/upload', {
    method: 'POST',
    body: formData,
  })
}

export function reingestDocument(documentId) {
  return requestJson(`/documents/${documentId}/reingest`, {
    method: 'POST',
  })
}

export function deleteDocument(documentId) {
  return requestJson(`/documents/${documentId}`, {
    method: 'DELETE',
  })
}

export function listSessions() {
  return requestJson('/sessions')
}

export function createSession(payload) {
  return requestJson('/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function getSession(sessionId) {
  return requestJson(`/sessions/${sessionId}`)
}

export function updateSession(sessionId, payload) {
  return requestJson(`/sessions/${sessionId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function deleteSession(sessionId) {
  return requestJson(`/sessions/${sessionId}`, {
    method: 'DELETE',
  })
}

/**
 * Stream LangGraph SSE events from POST /sessions/{id}/chat/stream.
 * Normalizes CRLF line endings emitted by sse-starlette before splitting frames.
 */
export async function streamChat({ sessionId, query, documentIds, signal, onEvent }) {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({
      query,
      document_ids: documentIds ?? null,
    }),
    signal,
  })

  if (!response.ok) {
    let detail = `Chat request failed (${response.status})`
    try {
      const err = await response.json()
      if (err?.detail) detail = err.detail
    } catch {
      // ignore non-JSON error body
    }
    throw new Error(detail)
  }

  if (!response.body) {
    throw new Error('Streaming response body is unavailable.')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  const processFrame = (frame) => {
    const trimmed = frame.trim()
    if (!trimmed) return
    let eventName = 'message'
    const dataLines = []
    for (const line of trimmed.split('\n')) {
      if (line.startsWith('event:')) {
        eventName = line.slice(6).trim()
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trim())
      }
    }
    if (dataLines.length === 0) return
    const rawData = dataLines.join('\n')
    try {
      onEvent(eventName, JSON.parse(rawData))
    } catch {
      onEvent(eventName, rawData)
    }
  }

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')
    let splitIdx = buffer.indexOf('\n\n')
    while (splitIdx !== -1) {
      const frame = buffer.slice(0, splitIdx)
      buffer = buffer.slice(splitIdx + 2)
      processFrame(frame)
      splitIdx = buffer.indexOf('\n\n')
    }
  }

  if (buffer.trim()) {
    processFrame(buffer)
  }
}

