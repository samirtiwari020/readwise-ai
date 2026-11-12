import { useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  AlertTriangle,
  BookOpen,
  Check,
  CheckSquare,
  Eye,
  FileText,
  Library,
  Loader2,
  MessageSquare,
  Pencil,
  Plus,
  RefreshCw,
  Square,
  Trash2,
  UploadCloud,
  X,
} from 'lucide-react'

function formatFileSize(bytes) {
  if (!bytes) return '0 MB'
  const mb = bytes / (1024 * 1024)
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`
}

function SidebarSkeleton() {
  return (
    <div className="space-y-2.5 p-3">
      {[1, 2, 3].map((item) => (
        <div
          key={item}
          className="animate-pulse rounded-xl border border-slate-200/70 bg-white p-3"
        >
          <div className="h-3.5 w-3/4 rounded bg-slate-200" />
          <div className="mt-2 h-2.5 w-1/2 rounded bg-slate-100" />
        </div>
      ))}
    </div>
  )
}

function DocumentCard({
  doc,
  isScoped,
  isViewing,
  onToggleScope,
  onOpenPdf,
  onReingest,
  onDelete,
}) {
  const [confirmDelete, setConfirmDelete] = useState(false)
  const isReady = doc.ingestion_status === 'completed'
  const isFailed = doc.ingestion_status === 'failed'
  const isBusy = doc.ingestion_status === 'pending' || doc.ingestion_status === 'processing'

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.96 }}
      transition={{ duration: 0.16 }}
      className={`group rounded-xl border p-3 transition-all ${
        isViewing
          ? 'border-indigo-300 bg-indigo-50/40 shadow-xs'
          : isScoped
            ? 'border-indigo-200/90 bg-white shadow-xs'
            : 'border-slate-200/80 bg-white hover:border-slate-300 hover:shadow-xs'
      }`}
    >
      <div className="flex items-start gap-2.5">
        <button
          type="button"
          disabled={!isReady}
          onClick={() => onToggleScope(doc.id)}
          aria-label={isScoped ? `Remove ${doc.title} from search scope` : `Scope search to ${doc.title}`}
          title={isReady ? 'Include in search scope' : 'Wait for indexing to finish'}
          className={`mt-0.5 shrink-0 rounded transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
            !isReady
              ? 'cursor-not-allowed text-slate-300'
              : isScoped
                ? 'text-indigo-600 hover:text-indigo-700'
                : 'text-slate-400 hover:text-slate-600'
          }`}
        >
          {isScoped ? <CheckSquare className="h-4 w-4" /> : <Square className="h-4 w-4" />}
        </button>

        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-2">
            <h4 className="truncate text-xs font-semibold text-slate-900" title={doc.title}>
              {doc.title}
            </h4>
            {isReady && (
              <span className="shrink-0 rounded-full bg-emerald-50 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-700">
                Indexed
              </span>
            )}
            {isBusy && (
              <span className="inline-flex shrink-0 items-center gap-1 rounded-full bg-indigo-50 px-1.5 py-0.5 text-[10px] font-semibold text-indigo-700">
                <Loader2 className="h-2.5 w-2.5 animate-spin" />
                {Math.round(doc.progress_percent || 0)}%
              </span>
            )}
            {isFailed && (
              <span className="inline-flex shrink-0 items-center gap-1 rounded-full bg-rose-50 px-1.5 py-0.5 text-[10px] font-semibold text-rose-700">
                <AlertTriangle className="h-2.5 w-2.5" />
                Failed
              </span>
            )}
          </div>

          <p className="mt-1 flex items-center gap-2 text-[11px] text-slate-500">
            <span>{doc.page_count ? `${doc.page_count.toLocaleString()} pages` : formatFileSize(doc.file_size_bytes)}</span>
            {doc.chunk_count > 0 && <span>· {doc.chunk_count.toLocaleString()} chunks</span>}
          </p>

          {isBusy && (
            <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
              <div
                className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 transition-all duration-300"
                style={{ width: `${Math.max(8, doc.progress_percent || 8)}%` }}
              />
            </div>
          )}

          {isFailed && doc.error_message && (
            <p className="mt-1.5 line-clamp-2 text-[11px] text-rose-600" title={doc.error_message}>
              {doc.error_message}
            </p>
          )}

          <div className="mt-2 flex items-center justify-between gap-1">
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => onOpenPdf(doc)}
                className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] font-medium text-slate-600 transition-colors hover:bg-slate-100 hover:text-indigo-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
              >
                <Eye className="h-3 w-3" />
                View
              </button>
              {isFailed && (
                <button
                  type="button"
                  onClick={() => onReingest(doc.id)}
                  className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] font-medium text-amber-700 transition-colors hover:bg-amber-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
                >
                  <RefreshCw className="h-3 w-3" />
                  Retry
                </button>
              )}
            </div>

            {confirmDelete ? (
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => {
                    onDelete(doc.id)
                    setConfirmDelete(false)
                  }}
                  className="rounded-md bg-rose-600 px-2 py-0.5 text-[10px] font-semibold text-white hover:bg-rose-700"
                >
                  Confirm
                </button>
                <button
                  type="button"
                  onClick={() => setConfirmDelete(false)}
                  className="rounded-md px-1.5 py-0.5 text-[10px] font-medium text-slate-500 hover:bg-slate-100"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => setConfirmDelete(true)}
                aria-label={`Delete ${doc.title}`}
                className="rounded-lg p-1 text-slate-400 opacity-75 transition-colors hover:bg-rose-50 hover:text-rose-600 group-hover:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  )
}

function SessionCard({ session, isActive, onSelect, onRename, onDelete }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(session.title)

  const commitRename = () => {
    const clean = draft.trim()
    if (clean && clean !== session.title) {
      onRename(session.id, clean)
    }
    setEditing(false)
  }

  return (
    <div
      className={`group flex items-center justify-between gap-2 rounded-xl border px-3 py-2.5 transition-all ${
        isActive
          ? 'border-indigo-200 bg-indigo-50/70 text-indigo-950 shadow-xs'
          : 'border-transparent bg-white text-slate-700 hover:border-slate-200 hover:bg-slate-50/70'
      }`}
    >
      {editing ? (
        <div className="flex flex-1 items-center gap-1.5">
          <input
            type="text"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') commitRename()
              if (e.key === 'Escape') {
                setDraft(session.title)
                setEditing(false)
              }
            }}
            autoFocus
            className="w-full rounded-lg border border-indigo-300 bg-white px-2 py-1 text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
          <button
            type="button"
            onClick={commitRename}
            aria-label="Save session title"
            className="rounded-md bg-indigo-600 p-1 text-white hover:bg-indigo-700"
          >
            <Check className="h-3 w-3" />
          </button>
        </div>
      ) : (
        <>
          <button
            type="button"
            onClick={() => onSelect(session.id)}
            className="flex min-w-0 flex-1 items-center gap-2 text-left focus-visible:outline-none"
          >
            <MessageSquare
              className={`h-3.5 w-3.5 shrink-0 ${
                isActive ? 'text-indigo-600' : 'text-slate-400'
              }`}
            />
            <span className="truncate text-xs font-medium">{session.title}</span>
          </button>
          <div className="flex shrink-0 items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
            <button
              type="button"
              onClick={() => {
                setDraft(session.title)
                setEditing(true)
              }}
              aria-label="Rename session"
              className="rounded-md p-1 text-slate-400 hover:bg-white hover:text-slate-700"
            >
              <Pencil className="h-3 w-3" />
            </button>
            <button
              type="button"
              onClick={() => onDelete(session.id)}
              aria-label="Delete session"
              className="rounded-md p-1 text-slate-400 hover:bg-rose-50 hover:text-rose-600"
            >
              <Trash2 className="h-3 w-3" />
            </button>
          </div>
        </>
      )}
    </div>
  )
}

function SidebarContent({
  documents,
  sessions,
  activeSessionId,
  scopedDocumentIds,
  activePdfDocId,
  loadingDocs,
  loadingSessions,
  uploading,
  onUploadFiles,
  onToggleScope,
  onClearScope,
  onOpenPdf,
  onReingestDoc,
  onDeleteDoc,
  onCreateSession,
  onSelectSession,
  onRenameSession,
  onDeleteSession,
  onCloseMobile,
}) {
  const [tab, setTab] = useState('library')
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef(null)

  const handleDrop = (event) => {
    event.preventDefault()
    setDragOver(false)
    const files = Array.from(event.dataTransfer.files || []).filter((f) =>
      f.name.toLowerCase().endsWith('.pdf')
    )
    if (files.length > 0) onUploadFiles(files)
  }

  return (
    <div className="flex h-full flex-col bg-slate-50/70">
      {/* Header switcher */}
      <div className="flex items-center justify-between gap-2 border-b border-slate-200/80 bg-white px-3 py-2.5">
        <div className="grid flex-1 grid-cols-2 rounded-xl bg-slate-100 p-1">
          <button
            type="button"
            onClick={() => setTab('library')}
            className={`inline-flex items-center justify-center gap-1.5 rounded-lg py-1.5 text-xs font-semibold transition-all ${
              tab === 'library'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Library className="h-3.5 w-3.5" />
            Library ({documents.length})
          </button>
          <button
            type="button"
            onClick={() => setTab('sessions')}
            className={`inline-flex items-center justify-center gap-1.5 rounded-lg py-1.5 text-xs font-semibold transition-all ${
              tab === 'sessions'
                ? 'bg-white text-slate-900 shadow-xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <MessageSquare className="h-3.5 w-3.5" />
            Chats ({sessions.length})
          </button>
        </div>

        {onCloseMobile && (
          <button
            type="button"
            onClick={onCloseMobile}
            aria-label="Close sidebar"
            className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 lg:hidden"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {tab === 'library' ? (
        <div className="flex min-h-0 flex-1 flex-col">
          {/* Upload Dropzone */}
          <div className="p-3 pb-2">
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,application/pdf"
              multiple
              onChange={(e) => {
                const picked = Array.from(e.target.files || [])
                if (picked.length > 0) onUploadFiles(picked)
                e.target.value = ''
              }}
              className="hidden"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault()
                setDragOver(true)
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              disabled={uploading}
              className={`flex w-full cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-4 py-4 text-center transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
                dragOver
                  ? 'border-indigo-500 bg-indigo-50/70'
                  : 'border-slate-300/90 bg-white hover:border-indigo-300 hover:bg-indigo-50/30'
              }`}
            >
              <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-50 text-indigo-600">
                {uploading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <UploadCloud className="h-4 w-4" />
                )}
              </div>
              <p className="text-xs font-semibold text-slate-800">
                {uploading ? 'Uploading PDF...' : 'Upload PDF Books & Papers'}
              </p>
              <p className="mt-0.5 text-[11px] text-slate-500">
                Drag & drop or click · Supports 1,000+ pages
              </p>
            </button>
          </div>

          {scopedDocumentIds.length > 0 && (
            <div className="mx-3 mb-1 flex items-center justify-between rounded-xl border border-indigo-200/80 bg-indigo-50/70 px-3 py-1.5 text-[11px]">
              <span className="font-medium text-indigo-800">
                {scopedDocumentIds.length} doc{scopedDocumentIds.length > 1 ? 's' : ''} scoped
              </span>
              <button
                type="button"
                onClick={onClearScope}
                className="font-semibold text-indigo-700 hover:underline"
              >
                Search all
              </button>
            </div>
          )}

          {/* Document Cards */}
          <div className="flex-1 overflow-y-auto px-3 pb-4 pt-1">
            {loadingDocs ? (
              <SidebarSkeleton />
            ) : documents.length === 0 ? (
              <div className="mt-6 flex flex-col items-center rounded-2xl border border-dashed border-slate-200 bg-white/70 px-4 py-8 text-center">
                <BookOpen className="h-8 w-8 text-slate-300" />
                <p className="mt-2.5 text-xs font-semibold text-slate-700">
                  Your library is empty
                </p>
                <p className="mt-1 text-[11px] leading-relaxed text-slate-500">
                  Upload textbooks, research papers, or technical PDFs to start page-grounded Q&A.
                </p>
              </div>
            ) : (
              <div className="space-y-2.5">
                <AnimatePresence initial={false}>
                  {documents.map((doc) => (
                    <DocumentCard
                      key={doc.id}
                      doc={doc}
                      isScoped={scopedDocumentIds.includes(doc.id)}
                      isViewing={activePdfDocId === doc.id}
                      onToggleScope={onToggleScope}
                      onOpenPdf={onOpenPdf}
                      onReingest={onReingestDoc}
                      onDelete={onDeleteDoc}
                    />
                  ))}
                </AnimatePresence>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="p-3 pb-2">
            <button
              type="button"
              onClick={onCreateSession}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 px-3.5 py-2.5 text-xs font-semibold text-white shadow-sm shadow-indigo-600/20 transition-all hover:brightness-105 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
            >
              <Plus className="h-4 w-4" />
              New Research Session
            </button>
          </div>

          <div className="flex-1 overflow-y-auto px-3 pb-4 pt-1">
            {loadingSessions ? (
              <SidebarSkeleton />
            ) : sessions.length === 0 ? (
              <div className="mt-6 flex flex-col items-center rounded-2xl border border-dashed border-slate-200 bg-white/70 px-4 py-8 text-center">
                <FileText className="h-8 w-8 text-slate-300" />
                <p className="mt-2.5 text-xs font-semibold text-slate-700">No saved sessions</p>
                <p className="mt-1 text-[11px] text-slate-500">
                  Create a session to preserve multi-turn citations and traces.
                </p>
              </div>
            ) : (
              <div className="space-y-1.5">
                {sessions.map((session) => (
                  <SessionCard
                    key={session.id}
                    session={session}
                    isActive={session.id === activeSessionId}
                    onSelect={(id) => {
                      onSelectSession(id)
                      onCloseMobile?.()
                    }}
                    onRename={onRenameSession}
                    onDelete={onDeleteSession}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export function Sidebar({ isMobileOpen, onCloseMobile, ...props }) {
  return (
    <>
      {/* Desktop persistent sidebar */}
      <aside
        aria-label="Document Library and Research Sessions"
        className="hidden w-80 shrink-0 border-r border-slate-200/80 lg:flex lg:flex-col"
      >
        <SidebarContent {...props} />
      </aside>

      {/* Mobile / Tablet slide-over drawer */}
      <AnimatePresence>
        {isMobileOpen && (
          <div className="fixed inset-0 z-40 flex lg:hidden">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={onCloseMobile}
              className="fixed inset-0 bg-slate-900/35 backdrop-blur-xs"
            />
            <motion.aside
              initial={{ x: '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: '-100%' }}
              transition={{ type: 'spring', damping: 28, stiffness: 300 }}
              aria-label="Mobile Library Drawer"
              className="relative z-10 flex h-full w-80 max-w-[85vw] flex-col border-r border-slate-200 bg-white shadow-2xl"
            >
              <SidebarContent {...props} onCloseMobile={onCloseMobile} />
            </motion.aside>
          </div>
        )}
      </AnimatePresence>
    </>
  )
}

