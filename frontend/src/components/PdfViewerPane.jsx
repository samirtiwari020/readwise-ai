import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  BookOpen,
  ChevronLeft,
  ChevronRight,
  Quote,
  X,
  ZoomIn,
  ZoomOut,
} from 'lucide-react'
import { Document as PdfDocument, Page, pdfjs } from 'react-pdf'

import { getDocumentFileUrl } from '../api/client'

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url
).toString()

function PdfViewerContent({
  doc,
  pageNumber,
  activeCitation,
  onPageChange,
  onDismissCitation,
  onClose,
}) {
  const [numPages, setNumPages] = useState(doc?.page_count || null)
  const [zoom, setZoom] = useState(1.0)
  const [draftPage, setDraftPage] = useState(null)
  const [containerWidth, setContainerWidth] = useState(460)
  const [loadError, setLoadError] = useState(null)
  const viewportRef = useRef(null)

  useEffect(() => {
    const el = viewportRef.current
    if (!el) return undefined
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const width = Math.max(260, Math.floor(entry.contentRect.width - 32))
        setContainerWidth(width)
      }
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  const clampPage = (target) => {
    const parsed = Number.parseInt(target, 10)
    setDraftPage(null)
    if (Number.isNaN(parsed)) return
    const upper = numPages || parsed
    const next = Math.max(1, Math.min(upper, parsed))
    onPageChange(next)
  }

  return (
    <div className="flex h-full flex-col bg-slate-100/70">
      {/* Viewer Header */}
      <div className="flex items-center justify-between gap-2 border-b border-slate-200 bg-white px-3.5 py-2.5">
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-xs font-semibold text-slate-900">
            {doc ? doc.title : 'PDF Source Reader'}
          </h3>
          {doc && (
            <p className="text-[11px] text-slate-500">
              {numPages ? `${numPages.toLocaleString()} pages` : 'Loading document...'}
            </p>
          )}
        </div>

        {doc && Array.isArray(doc.toc_json) && doc.toc_json.length > 0 && (
          <select
            aria-label="Jump to chapter"
            value=""
            onChange={(e) => {
              if (e.target.value) onPageChange(Number(e.target.value))
            }}
            className="hidden max-w-[150px] rounded-lg border border-slate-200 bg-slate-50 px-2 py-1 text-[11px] text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 sm:block"
          >
            <option value="">Jump to section...</option>
            {doc.toc_json.slice(0, 60).map((entry, idx) => (
              <option key={`${idx}-${entry.title}`} value={entry.page}>
                {'\u00A0'.repeat(Math.max(0, (entry.level - 1) * 2))}
                {entry.title} (p. {entry.page})
              </option>
            ))}
          </select>
        )}

        <button
          type="button"
          onClick={onClose}
          aria-label="Close PDF reader"
          className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100 hover:text-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Controls Bar */}
      {doc && (
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200/80 bg-white/90 px-3.5 py-2 text-xs">
          <div className="flex items-center gap-1">
            <button
              type="button"
              disabled={pageNumber <= 1}
              onClick={() => clampPage(pageNumber - 1)}
              aria-label="Previous page"
              className="rounded-lg border border-slate-200 p-1 text-slate-600 hover:bg-slate-50 disabled:opacity-40"
            >
              <ChevronLeft className="h-3.5 w-3.5" />
            </button>
            <form
              onSubmit={(e) => {
                e.preventDefault()
                clampPage(draftPage ?? String(pageNumber))
              }}
              className="flex items-center gap-1"
            >
              <input
                type="text"
                inputMode="numeric"
                value={draftPage ?? String(pageNumber || 1)}
                onChange={(e) => setDraftPage(e.target.value)}
                onBlur={() => {
                  if (draftPage !== null) clampPage(draftPage)
                }}
                aria-label="Current page number"
                className="w-12 rounded-md border border-slate-200 px-1.5 py-0.5 text-center font-mono text-xs text-slate-900 focus:border-indigo-400 focus:outline-none"
              />
              <span className="font-mono text-[11px] text-slate-500">
                / {numPages || '—'}
              </span>
            </form>
            <button
              type="button"
              disabled={numPages !== null && pageNumber >= numPages}
              onClick={() => clampPage(pageNumber + 1)}
              aria-label="Next page"
              className="rounded-lg border border-slate-200 p-1 text-slate-600 hover:bg-slate-50 disabled:opacity-40"
            >
              <ChevronRight className="h-3.5 w-3.5" />
            </button>
          </div>

          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => setZoom((z) => Math.max(0.6, +(z - 0.15).toFixed(2)))}
              aria-label="Zoom out"
              className="rounded-lg border border-slate-200 p-1 text-slate-600 hover:bg-slate-50"
            >
              <ZoomOut className="h-3.5 w-3.5" />
            </button>
            <span className="w-11 text-center font-mono text-[11px] text-slate-600">
              {Math.round(zoom * 100)}%
            </span>
            <button
              type="button"
              onClick={() => setZoom((z) => Math.min(2.0, +(z + 0.15).toFixed(2)))}
              aria-label="Zoom in"
              className="rounded-lg border border-slate-200 p-1 text-slate-600 hover:bg-slate-50"
            >
              <ZoomIn className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      )}

      {/* Cited Excerpt Callout */}
      <AnimatePresence initial={false}>
        {activeCitation && (
          <motion.div
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            className="mx-3 mt-3 rounded-xl border border-indigo-200 bg-indigo-50/90 p-3 text-xs shadow-2xs"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex items-center gap-1.5 font-semibold text-indigo-900">
                <Quote className="h-3.5 w-3.5 text-indigo-600" />
                <span>
                  Citation [{activeCitation.citation_index}] · Page {activeCitation.page_start}
                  {activeCitation.chapter ? ` · ${activeCitation.chapter}` : ''}
                </span>
              </div>
              <button
                type="button"
                onClick={onDismissCitation}
                aria-label="Dismiss excerpt banner"
                className="rounded p-0.5 text-indigo-500 hover:bg-indigo-100"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
            <p className="mt-1.5 leading-relaxed text-indigo-950/90">
              “{activeCitation.quote_snippet}”
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* PDF Canvas Viewport */}
      <div ref={viewportRef} className="flex-1 overflow-auto p-4">
        {!doc ? (
          <div className="flex h-full flex-col items-center justify-center rounded-2xl border border-dashed border-slate-300 bg-white/70 p-6 text-center">
            <BookOpen className="h-9 w-9 text-slate-300" />
            <p className="mt-3 text-xs font-semibold text-slate-700">No PDF Selected</p>
            <p className="mt-1 max-w-xs text-[11px] leading-relaxed text-slate-500">
              Click any inline citation badge <span className="font-mono text-indigo-600">[1]</span> or
              select View on a document in the library to jump straight to the source page.
            </p>
          </div>
        ) : loadError ? (
          <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-xs text-rose-700">
            Failed to load PDF preview: {loadError}
          </div>
        ) : (
          <div className="flex justify-center">
            <PdfDocument
              file={getDocumentFileUrl(doc.id)}
              onLoadSuccess={({ numPages: total }) => setNumPages(total)}
              onLoadError={(err) => setLoadError(err?.message || 'Unable to parse PDF')}
              loading={
                <div className="h-[520px] w-full max-w-md animate-pulse rounded-xl bg-white shadow-sm" />
              }
            >
              <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-md">
                <Page
                  pageNumber={pageNumber}
                  width={Math.round(containerWidth * zoom)}
                  loading={
                    <div className="flex h-[520px] w-[420px] items-center justify-center bg-white text-xs text-slate-400">
                      Rendering page {pageNumber}...
                    </div>
                  }
                />
              </div>
            </PdfDocument>
          </div>
        )}
      </div>
    </div>
  )
}

export function PdfViewerPane({ isOpen, ...props }) {
  const viewerKey = props.doc?.id || 'empty-pdf'

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Desktop split pane */}
          <motion.section
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: '42%', opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: 'easeOut' }}
            aria-label="PDF Source Viewer"
            className="hidden h-full min-w-[360px] max-w-[680px] shrink-0 overflow-hidden border-l border-slate-200/80 xl:block"
          >
            <PdfViewerContent key={viewerKey} {...props} />
          </motion.section>

          {/* Mobile / Tablet / Laptop overlay drawer */}
          <div className="fixed inset-0 z-40 flex justify-end xl:hidden">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={props.onClose}
              className="fixed inset-0 bg-slate-900/35 backdrop-blur-xs"
            />
            <motion.section
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 28, stiffness: 300 }}
              aria-label="Mobile PDF Source Viewer"
              className="relative z-10 h-full w-full max-w-xl border-l border-slate-200 bg-white shadow-2xl"
            >
              <PdfViewerContent key={viewerKey} {...props} />
            </motion.section>
          </div>
        </>
      )}
    </AnimatePresence>
  )
}
