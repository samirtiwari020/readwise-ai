import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import {
  ArrowUp,
  BookOpen,
  FileSearch,
  Layers,
  Loader2,
  Scale,
  Sparkles,
  Square,
} from 'lucide-react'

import { AgentTracePanel } from './AgentTracePanel'
import { MarkdownWithCitations } from './MarkdownWithCitations'

const STARTER_PROMPTS = [
  {
    icon: Scale,
    title: 'Cross-Document Comparison',
    prompt: 'Compare the core architectures and trade-offs explained across the selected documents.',
  },
  {
    icon: FileSearch,
    title: 'Page-Grounded Deep Dive',
    prompt: 'What are the key definitions and theoretical guarantees introduced in Chapter 1?',
  },
  {
    icon: Layers,
    title: 'Structured Synthesis',
    prompt: 'Summarize the main practical recommendations in a table with exact page citations.',
  },
]

const STAGE_ORDER = ['decomposing', 'retrieving', 'grading', 'synthesizing', 'verifying']

function MessageSkeleton() {
  return (
    <div className="mx-auto max-w-3xl space-y-5 p-6">
      <div className="ml-auto h-12 w-2/3 animate-pulse rounded-2xl bg-indigo-100/70" />
      <div className="space-y-3 rounded-2xl border border-slate-200 bg-white p-5">
        <div className="h-4 w-1/3 animate-pulse rounded bg-slate-200" />
        <div className="h-3.5 w-full animate-pulse rounded bg-slate-100" />
        <div className="h-3.5 w-5/6 animate-pulse rounded bg-slate-100" />
        <div className="h-3.5 w-3/4 animate-pulse rounded bg-slate-100" />
      </div>
    </div>
  )
}

function CitationSourceCards({ citations, onSelectCitation }) {
  if (!citations || citations.length === 0) return null

  return (
    <div className="mt-4 border-t border-slate-100 pt-3">
      <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
        Verified Sources ({citations.length})
      </p>
      <div className="grid gap-2 sm:grid-cols-2">
        {citations.map((cite) => {
          const pageLabel =
            cite.page_start === cite.page_end
              ? `p. ${cite.page_start}`
              : `pp. ${cite.page_start}-${cite.page_end}`
          return (
            <button
              key={`${cite.citation_index}-${cite.chunk_id}`}
              type="button"
              onClick={() => onSelectCitation?.(cite)}
              className="group flex items-start gap-2.5 rounded-xl border border-slate-200/90 bg-slate-50/70 p-2.5 text-left transition-all hover:border-indigo-300 hover:bg-indigo-50/40 hover:shadow-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
            >
              <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-indigo-600 font-mono text-[11px] font-bold text-white">
                {cite.citation_index}
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-1.5">
                  <span className="truncate text-xs font-semibold text-slate-800 group-hover:text-indigo-800">
                    {cite.document_title}
                  </span>
                  <span className="shrink-0 rounded-md bg-white px-1.5 py-0.5 font-mono text-[10px] font-medium text-slate-600 ring-1 ring-slate-200/80">
                    {pageLabel}
                  </span>
                </div>
                {cite.chapter && (
                  <p className="mt-0.5 truncate text-[11px] font-medium text-indigo-600/90">
                    {cite.chapter}
                  </p>
                )}
                <p className="mt-1 line-clamp-2 text-[11px] leading-relaxed text-slate-500">
                  {cite.quote_snippet}
                </p>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}

export function ChatPane({
  messages,
  loadingMessages,
  streamingState,
  completedDocs,
  scopedDocs,
  onSendMessage,
  onStopStreaming,
  onSelectCitation,
  onToggleMobileSidebar,
}) {
  const [input, setInput] = useState('')
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages.length, streamingState?.content, streamingState?.stage])

  const canSend = input.trim().length > 0 && !streamingState && completedDocs.length > 0

  const handleSubmit = (e) => {
    e?.preventDefault()
    if (!canSend) return
    onSendMessage(input.trim())
    setInput('')
  }

  return (
    <main className="flex min-w-0 flex-1 flex-col bg-slate-50/40">
      {/* Scrollable conversation viewport */}
      <div className="flex-1 overflow-y-auto px-3 py-5 sm:px-6">
        {loadingMessages ? (
          <MessageSkeleton />
        ) : messages.length === 0 && !streamingState ? (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.25 }}
            className="mx-auto flex max-w-2xl flex-col items-center justify-center py-10 text-center sm:py-16"
          >
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-600 to-violet-600 text-white shadow-md shadow-indigo-600/20">
              <Sparkles className="h-6 w-6" />
            </div>
            <h1 className="font-display text-xl font-bold tracking-tight text-slate-900 sm:text-2xl">
              Grounded Research Over Your Library
            </h1>
            <p className="mt-2 max-w-lg text-sm leading-relaxed text-slate-600">
              Ask multi-hop questions or compare concepts across 1,000+ page PDFs. Every claim is
              verified and linked directly to its exact page.
            </p>

            {completedDocs.length === 0 ? (
              <div className="mt-6 rounded-2xl border border-amber-200 bg-amber-50/70 px-5 py-4 text-xs text-amber-900">
                <p className="font-semibold">No indexed PDFs ready yet</p>
                <p className="mt-1 text-amber-800/90">
                  Upload a PDF in the Library sidebar and wait for indexing to reach 100%.
                </p>
                <button
                  type="button"
                  onClick={onToggleMobileSidebar}
                  className="mt-3 inline-flex items-center gap-1.5 rounded-lg bg-amber-900 px-3 py-1.5 text-xs font-semibold text-white lg:hidden"
                >
                  <BookOpen className="h-3.5 w-3.5" />
                  Open Library
                </button>
              </div>
            ) : (
              <div className="mt-8 grid w-full gap-3 text-left sm:grid-cols-3">
                {STARTER_PROMPTS.map((item) => {
                  const Icon = item.icon
                  return (
                    <button
                      key={item.title}
                      type="button"
                      onClick={() => onSendMessage(item.prompt)}
                      className="group flex flex-col rounded-2xl border border-slate-200/90 bg-white p-4 shadow-2xs transition-all hover:-translate-y-0.5 hover:border-indigo-300 hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
                    >
                      <div className="mb-2.5 flex h-8 w-8 items-center justify-center rounded-xl bg-indigo-50 text-indigo-600 transition-colors group-hover:bg-indigo-600 group-hover:text-white">
                        <Icon className="h-4 w-4" />
                      </div>
                      <h3 className="text-xs font-semibold text-slate-900">{item.title}</h3>
                      <p className="mt-1 text-[11px] leading-relaxed text-slate-500">
                        {item.prompt}
                      </p>
                    </button>
                  )
                })}
              </div>
            )}
          </motion.div>
        ) : (
          <div className="mx-auto max-w-3xl space-y-5">
            {messages.map((msg) =>
              msg.role === 'user' ? (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.18 }}
                  className="flex justify-end"
                >
                  <div className="max-w-[85%] rounded-2xl rounded-tr-sm bg-gradient-to-br from-indigo-600 to-indigo-700 px-4 py-3 text-sm leading-relaxed text-white shadow-sm shadow-indigo-600/15">
                    {msg.content}
                  </div>
                </motion.div>
              ) : (
                <motion.article
                  key={msg.id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.2 }}
                  className="rounded-2xl border border-slate-200/90 bg-white p-4 shadow-xs sm:p-5"
                >
                  <AgentTracePanel trace={msg.agent_trace} />
                  <MarkdownWithCitations
                    content={msg.content}
                    citations={msg.citations}
                    onSelectCitation={onSelectCitation}
                  />
                  <CitationSourceCards
                    citations={msg.citations}
                    onSelectCitation={onSelectCitation}
                  />
                </motion.article>
              )
            )}

            {/* Live SSE Streaming Progress Card */}
            {streamingState && (
              <motion.article
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className="rounded-2xl border border-indigo-200 bg-white p-4 shadow-sm sm:p-5"
              >
                <div className="mb-3 flex flex-wrap items-center justify-between gap-2 rounded-xl bg-indigo-50/70 px-3 py-2 text-xs">
                  <div className="flex items-center gap-2 font-medium text-indigo-800">
                    <Loader2 className="h-3.5 w-3.5 animate-spin text-indigo-600" />
                    <span>{streamingState.detail || 'Running agentic RAG workflow...'}</span>
                  </div>
                  <div className="hidden items-center gap-1 sm:flex">
                    {STAGE_ORDER.map((stage) => {
                      const activeIdx = STAGE_ORDER.indexOf(streamingState.stage)
                      const stepIdx = STAGE_ORDER.indexOf(stage)
                      const done = activeIdx > stepIdx
                      const active = activeIdx === stepIdx
                      return (
                        <span
                          key={stage}
                          className={`h-1.5 w-5 rounded-full transition-colors ${
                            done
                              ? 'bg-emerald-500'
                              : active
                                ? 'bg-indigo-600 animate-pulse'
                                : 'bg-slate-200'
                          }`}
                          title={stage}
                        />
                      )
                    })}
                  </div>
                </div>

                {streamingState.subQueries?.length > 0 && (
                  <div className="mb-3 flex flex-wrap gap-1.5">
                    {streamingState.subQueries.map((sq, i) => (
                      <span
                        key={`${i}-${sq}`}
                        className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] text-slate-600"
                      >
                        {sq}
                      </span>
                    ))}
                  </div>
                )}

                {streamingState.content ? (
                  <>
                    <MarkdownWithCitations
                      content={streamingState.content}
                      citations={streamingState.citations}
                      onSelectCitation={onSelectCitation}
                    />
                    <CitationSourceCards
                      citations={streamingState.citations}
                      onSelectCitation={onSelectCitation}
                    />
                  </>
                ) : (
                  <div className="space-y-2.5 py-1">
                    <div className="h-3.5 w-4/5 animate-pulse rounded bg-slate-100" />
                    <div className="h-3.5 w-2/3 animate-pulse rounded bg-slate-100" />
                  </div>
                )}
              </motion.article>
            )}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Sticky Composer */}
      <div className="border-t border-slate-200/80 bg-white/90 px-3 py-3 backdrop-blur-md sm:px-6">
        <form
          onSubmit={handleSubmit}
          className="mx-auto max-w-3xl rounded-2xl border border-slate-200 bg-white p-2 shadow-sm transition-shadow focus-within:border-indigo-400 focus-within:ring-4 focus-within:ring-indigo-500/10"
        >
          <div className="mb-1.5 flex flex-wrap items-center gap-1.5 px-2 pt-0.5 text-[11px] text-slate-500">
            <span className="font-medium text-slate-400">Scope:</span>
            {scopedDocs.length > 0 ? (
              scopedDocs.map((doc) => (
                <span
                  key={doc.id}
                  className="inline-flex items-center rounded-md bg-indigo-50 px-2 py-0.5 font-medium text-indigo-700"
                >
                  {doc.title}
                </span>
              ))
            ) : (
              <span className="rounded-md bg-slate-100 px-2 py-0.5 font-medium text-slate-600">
                All indexed documents ({completedDocs.length})
              </span>
            )}
          </div>

          <div className="flex items-end gap-2">
            <textarea
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSubmit()
                }
              }}
              placeholder={
                completedDocs.length > 0
                  ? 'Ask a research question or compare concepts across documents...'
                  : 'Upload & index a PDF in the sidebar to begin...'
              }
              disabled={completedDocs.length === 0}
              aria-label="Research query input"
              className="max-h-36 min-h-[42px] flex-1 resize-none bg-transparent px-2.5 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none disabled:cursor-not-allowed"
            />
            {streamingState ? (
              <button
                type="button"
                onClick={onStopStreaming}
                aria-label="Stop response"
                className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-rose-600 text-white transition-colors hover:bg-rose-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500"
              >
                <Square className="h-3.5 w-3.5 fill-current" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!canSend}
                aria-label="Send query"
                className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-indigo-600 text-white shadow-xs transition-all hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
              >
                <ArrowUp className="h-4 w-4" />
              </button>
            )}
          </div>
        </form>
      </div>
    </main>
  )
}

