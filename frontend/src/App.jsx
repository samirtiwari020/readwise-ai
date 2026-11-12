import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  createSession,
  deleteDocument,
  deleteSession,
  fetchHealth,
  getDocument,
  getDocumentStatus,
  getSession,
  listDocuments,
  listSessions,
  reingestDocument,
  streamChat,
  updateSession,
  uploadDocument,
} from './api/client'
import { ChatPane } from './components/ChatPane'
import { PdfViewerPane } from './components/PdfViewerPane'
import { Sidebar } from './components/Sidebar'
import { useToast } from './components/toastContext'
import { TopBar } from './components/TopBar'

export default function App() {
  const { pushToast } = useToast()

  const [health, setHealth] = useState(null)
  const [documents, setDocuments] = useState([])
  const [sessions, setSessions] = useState([])
  const [activeSession, setActiveSession] = useState(null)
  const [messages, setMessages] = useState([])
  const [scopedDocumentIds, setScopedDocumentIds] = useState([])

  const [loadingDocs, setLoadingDocs] = useState(true)
  const [loadingSessions, setLoadingSessions] = useState(true)
  const [loadingMessages, setLoadingMessages] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [streamingState, setStreamingState] = useState(null)

  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false)
  const [isPdfOpen, setIsPdfOpen] = useState(false)
  const [activePdfDoc, setActivePdfDoc] = useState(null)
  const [pdfPage, setPdfPage] = useState(1)
  const [activeCitation, setActiveCitation] = useState(null)

  const abortRef = useRef(null)

  const completedDocs = useMemo(
    () => documents.filter((d) => d.ingestion_status === 'completed'),
    [documents]
  )

  const scopedDocs = useMemo(
    () => completedDocs.filter((d) => scopedDocumentIds.includes(d.id)),
    [completedDocs, scopedDocumentIds]
  )

  const loadSessionDetail = useCallback(
    async (sessionId) => {
      setLoadingMessages(true)
      try {
        const detail = await getSession(sessionId)
        setActiveSession(detail)
        setMessages(detail.messages || [])
        setScopedDocumentIds(detail.selected_document_ids || [])
      } catch (err) {
        pushToast({ type: 'error', title: 'Failed to load session', description: err.message })
      } finally {
        setLoadingMessages(false)
      }
    },
    [pushToast]
  )

  // Initial hydration
  useEffect(() => {
    let mounted = true
    async function bootstrap() {
      const [healthRes, docsRes, sessionsRes] = await Promise.allSettled([
        fetchHealth(),
        listDocuments(),
        listSessions(),
      ])

      if (!mounted) return

      if (healthRes.status === 'fulfilled') {
        setHealth(healthRes.value)
      }
      if (docsRes.status === 'fulfilled') {
        setDocuments(docsRes.value.documents || [])
      } else {
        pushToast({
          type: 'error',
          title: 'Backend unreachable',
          description: 'Start FastAPI on port 8000 to connect your library.',
        })
      }
      setLoadingDocs(false)

      if (sessionsRes.status === 'fulfilled') {
        const loadedSessions = sessionsRes.value.sessions || []
        setSessions(loadedSessions)
        if (loadedSessions.length > 0) {
          await loadSessionDetail(loadedSessions[0].id)
        }
      }
      setLoadingSessions(false)
    }
    bootstrap()
    return () => {
      mounted = false
    }
  }, [loadSessionDetail, pushToast])

  // Poll pending / processing document ingestion progress
  useEffect(() => {
    const busyDocs = documents.filter(
      (doc) => doc.ingestion_status === 'pending' || doc.ingestion_status === 'processing'
    )
    if (busyDocs.length === 0) return undefined

    const timer = setInterval(async () => {
      const updates = await Promise.allSettled(busyDocs.map((doc) => getDocumentStatus(doc.id)))
      for (let i = 0; i < updates.length; i++) {
        const res = updates[i]
        const target = busyDocs[i]
        if (res.status !== 'fulfilled') continue
        const statusData = res.value

        if (statusData.ingestion_status === 'completed') {
          try {
            const fullDoc = await getDocument(target.id)
            setDocuments((prev) => prev.map((d) => (d.id === target.id ? fullDoc : d)))
          } catch {
            setDocuments((prev) =>
              prev.map((d) => (d.id === target.id ? { ...d, ...statusData } : d))
            )
          }
          pushToast({
            type: 'success',
            title: 'PDF Indexed',
            description: `${target.title} (${statusData.page_count || '?'} pages) is ready for Q&A.`,
          })
        } else if (statusData.ingestion_status === 'failed') {
          setDocuments((prev) =>
            prev.map((d) => (d.id === target.id ? { ...d, ...statusData } : d))
          )
          pushToast({
            type: 'error',
            title: `Ingestion failed: ${target.title}`,
            description: statusData.error_message || 'Unable to process PDF.',
          })
        } else {
          setDocuments((prev) =>
            prev.map((d) => (d.id === target.id ? { ...d, ...statusData } : d))
          )
        }
      }
    }, 1500)

    return () => clearInterval(timer)
  }, [documents, pushToast])

  // Escape closes mobile drawer or PDF viewer
  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.key !== 'Escape') return
      if (isMobileSidebarOpen) setIsMobileSidebarOpen(false)
      else if (isPdfOpen) setIsPdfOpen(false)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [isMobileSidebarOpen, isPdfOpen])

  const handleUploadFiles = async (files) => {
    setUploading(true)
    try {
      for (const file of files) {
        const result = await uploadDocument(file)
        const doc = result.document
        setDocuments((prev) => {
          const exists = prev.some((item) => item.id === doc.id)
          return exists ? prev.map((item) => (item.id === doc.id ? doc : item)) : [doc, ...prev]
        })
        if (result.deduplicated) {
          pushToast({
            type: 'info',
            title: 'Duplicate PDF skipped',
            description: `${doc.title} is already in your library (matched by SHA-256).`,
          })
        } else {
          pushToast({
            type: 'info',
            title: 'Indexing started',
            description: `Processing ${doc.title} page by page...`,
          })
        }
      }
    } catch (err) {
      pushToast({ type: 'error', title: 'Upload failed', description: err.message })
    } finally {
      setUploading(false)
    }
  }

  const syncScopeToSession = async (nextScope) => {
    if (!activeSession) return
    try {
      const updated = await updateSession(activeSession.id, {
        selected_document_ids: nextScope,
      })
      setActiveSession((prev) => (prev ? { ...prev, ...updated } : updated))
      setSessions((prev) => prev.map((s) => (s.id === updated.id ? updated : s)))
    } catch (err) {
      pushToast({ type: 'error', title: 'Could not save document scope', description: err.message })
    }
  }

  const handleToggleScope = (docId) => {
    const nextScope = scopedDocumentIds.includes(docId)
      ? scopedDocumentIds.filter((id) => id !== docId)
      : [...scopedDocumentIds, docId]
    setScopedDocumentIds(nextScope)
    syncScopeToSession(nextScope)
  }

  const handleClearScope = () => {
    setScopedDocumentIds([])
    syncScopeToSession([])
  }

  const handleOpenPdf = async (doc, targetPage = 1, citation = null) => {
    setActiveCitation(citation)
    setPdfPage(targetPage || 1)
    setActivePdfDoc(doc)
    setIsPdfOpen(true)

    if (!doc.toc_json) {
      try {
        const full = await getDocument(doc.id)
        setActivePdfDoc(full)
        setDocuments((prev) => prev.map((item) => (item.id === full.id ? full : item)))
      } catch {
        // non-fatal if TOC fetch fails
      }
    }
  }

  const handleSelectCitation = async (citation) => {
    const existing = documents.find((d) => d.id === citation.document_id)
    if (existing) {
      await handleOpenPdf(existing, citation.page_start, citation)
      return
    }
    try {
      const fetched = await getDocument(citation.document_id)
      await handleOpenPdf(fetched, citation.page_start, citation)
    } catch (err) {
      pushToast({ type: 'error', title: 'Document unavailable', description: err.message })
    }
  }

  const handleReingestDoc = async (docId) => {
    try {
      const updated = await reingestDocument(docId)
      setDocuments((prev) => prev.map((d) => (d.id === docId ? { ...d, ...updated } : d)))
      pushToast({ type: 'info', title: 'Re-indexing queued', description: 'Processing PDF again.' })
    } catch (err) {
      pushToast({ type: 'error', title: 'Re-ingest failed', description: err.message })
    }
  }

  const handleDeleteDoc = async (docId) => {
    try {
      await deleteDocument(docId)
      setDocuments((prev) => prev.filter((d) => d.id !== docId))
      const nextScope = scopedDocumentIds.filter((id) => id !== docId)
      setScopedDocumentIds(nextScope)
      if (activePdfDoc?.id === docId) {
        setActivePdfDoc(null)
        setActiveCitation(null)
        setIsPdfOpen(false)
      }
      pushToast({ type: 'success', title: 'Document deleted' })
    } catch (err) {
      pushToast({ type: 'error', title: 'Delete failed', description: err.message })
    }
  }

  const handleCreateSession = async () => {
    try {
      const created = await createSession({
        title: 'New Research Session',
        selected_document_ids: scopedDocumentIds,
      })
      setSessions((prev) => [created, ...prev])
      setActiveSession(created)
      setMessages([])
      setIsMobileSidebarOpen(false)
    } catch (err) {
      pushToast({ type: 'error', title: 'Could not create session', description: err.message })
    }
  }

  const handleRenameSession = async (sessionId, title) => {
    try {
      const updated = await updateSession(sessionId, { title })
      setSessions((prev) => prev.map((s) => (s.id === sessionId ? updated : s)))
      if (activeSession?.id === sessionId) {
        setActiveSession((prev) => (prev ? { ...prev, title: updated.title } : updated))
      }
    } catch (err) {
      pushToast({ type: 'error', title: 'Rename failed', description: err.message })
    }
  }

  const handleDeleteSession = async (sessionId) => {
    try {
      await deleteSession(sessionId)
      const remaining = sessions.filter((s) => s.id !== sessionId)
      setSessions(remaining)
      if (activeSession?.id === sessionId) {
        if (remaining.length > 0) {
          await loadSessionDetail(remaining[0].id)
        } else {
          setActiveSession(null)
          setMessages([])
        }
      }
      pushToast({ type: 'success', title: 'Session deleted' })
    } catch (err) {
      pushToast({ type: 'error', title: 'Could not delete session', description: err.message })
    }
  }

  const handleSendMessage = async (queryText) => {
    let targetSession = activeSession
    if (!targetSession) {
      try {
        targetSession = await createSession({
          title: 'New Research Session',
          selected_document_ids: scopedDocumentIds,
        })
        setSessions((prev) => [targetSession, ...prev])
        setActiveSession(targetSession)
      } catch (err) {
        pushToast({ type: 'error', title: 'Session error', description: err.message })
        return
      }
    }

    const tempUserMsgId = `temp-${Date.now()}`
    setMessages((prev) => [
      ...prev,
      { id: tempUserMsgId, role: 'user', content: queryText, created_at: new Date().toISOString() },
    ])
    setStreamingState({
      stage: 'decomposing',
      detail: 'Analyzing query and scoping documents...',
      subQueries: [],
      content: '',
      citations: [],
    })

    const controller = new AbortController()
    abortRef.current = controller

    try {
      await streamChat({
        sessionId: targetSession.id,
        query: queryText,
        documentIds: scopedDocumentIds.length > 0 ? scopedDocumentIds : null,
        signal: controller.signal,
        onEvent: (eventName, data) => {
          if (eventName === 'user_message') {
            setMessages((prev) => prev.map((m) => (m.id === tempUserMsgId ? data : m)))
          } else if (eventName === 'status') {
            setStreamingState((prev) =>
              prev ? { ...prev, stage: data.stage, detail: data.detail } : prev
            )
          } else if (eventName === 'trace') {
            setStreamingState((prev) =>
              prev ? { ...prev, subQueries: data.sub_queries || [] } : prev
            )
          } else if (eventName === 'token') {
            setStreamingState((prev) =>
              prev ? { ...prev, content: `${prev.content}${data.delta || ''}` } : prev
            )
          } else if (eventName === 'citations') {
            setStreamingState((prev) =>
              prev ? { ...prev, citations: Array.isArray(data) ? data : [] } : prev
            )
          } else if (eventName === 'done') {
            const finalMsg = data.assistant_message
            setMessages((prev) => [...prev, finalMsg])
            setStreamingState(null)
            // Refresh session list so auto-generated title appears immediately
            listSessions()
              .then((res) => {
                const updatedList = res.sessions || []
                setSessions(updatedList)
                const matched = updatedList.find((s) => s.id === targetSession.id)
                if (matched) setActiveSession((prev) => (prev ? { ...prev, ...matched } : matched))
              })
              .catch(() => {})
          } else if (eventName === 'error') {
            setStreamingState(null)
            pushToast({
              type: 'error',
              title: 'Research agent error',
              description: data?.detail || 'Failed to generate grounded response.',
            })
          }
        },
      })
    } catch (err) {
      if (err.name !== 'AbortError') {
        pushToast({ type: 'error', title: 'Chat failed', description: err.message })
      }
      setStreamingState(null)
    } finally {
      abortRef.current = null
    }
  }

  const handleStopStreaming = () => {
    abortRef.current?.abort()
    setStreamingState(null)
  }

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-slate-50 text-slate-900">
      <TopBar
        health={health}
        activeSession={activeSession}
        scopedCount={scopedDocumentIds.length}
        completedCount={completedDocs.length}
        isPdfOpen={isPdfOpen}
        onToggleSidebar={() => setIsMobileSidebarOpen((v) => !v)}
        onTogglePdf={() => setIsPdfOpen((v) => !v)}
      />

      <div className="flex min-h-0 flex-1 overflow-hidden">
        <Sidebar
          isMobileOpen={isMobileSidebarOpen}
          onCloseMobile={() => setIsMobileSidebarOpen(false)}
          documents={documents}
          sessions={sessions}
          activeSessionId={activeSession?.id || null}
          scopedDocumentIds={scopedDocumentIds}
          activePdfDocId={activePdfDoc?.id || null}
          loadingDocs={loadingDocs}
          loadingSessions={loadingSessions}
          uploading={uploading}
          onUploadFiles={handleUploadFiles}
          onToggleScope={handleToggleScope}
          onClearScope={handleClearScope}
          onOpenPdf={(doc) => handleOpenPdf(doc, 1, null)}
          onReingestDoc={handleReingestDoc}
          onDeleteDoc={handleDeleteDoc}
          onCreateSession={handleCreateSession}
          onSelectSession={loadSessionDetail}
          onRenameSession={handleRenameSession}
          onDeleteSession={handleDeleteSession}
        />

        <ChatPane
          messages={messages}
          loadingMessages={loadingMessages}
          streamingState={streamingState}
          completedDocs={completedDocs}
          scopedDocs={scopedDocs}
          onSendMessage={handleSendMessage}
          onStopStreaming={handleStopStreaming}
          onSelectCitation={handleSelectCitation}
          onToggleMobileSidebar={() => setIsMobileSidebarOpen(true)}
        />

        <PdfViewerPane
          isOpen={isPdfOpen}
          doc={activePdfDoc}
          pageNumber={pdfPage}
          activeCitation={activeCitation}
          onPageChange={setPdfPage}
          onDismissCitation={() => setActiveCitation(null)}
          onClose={() => setIsPdfOpen(false)}
        />
      </div>
    </div>
  )
}
