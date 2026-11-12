import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { ChevronDown, GitBranch, Search, ShieldCheck, Timer } from 'lucide-react'

const QUERY_TYPE_LABELS = {
  single_fact: 'Single Fact',
  multi_hop: 'Multi-Hop Synthesis',
  cross_document_compare: 'Cross-Doc Comparison',
  summary: 'Structured Summary',
  out_of_scope: 'Out of Scope',
}

export function AgentTracePanel({ trace }) {
  const [expanded, setExpanded] = useState(false)

  if (!trace) return null

  const typeLabel = QUERY_TYPE_LABELS[trace.query_type] || trace.query_type || 'Agentic RAG'
  const subQueries = Array.isArray(trace.sub_queries) ? trace.sub_queries : []
  const steps = Array.isArray(trace.steps) ? trace.steps : []

  return (
    <div className="mb-3 rounded-xl border border-slate-200/80 bg-slate-50/70">
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        aria-expanded={expanded}
        className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-xs text-slate-600 transition-colors hover:text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 rounded-xl"
      >
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-1 rounded-md bg-indigo-50 px-2 py-0.5 font-medium text-indigo-700">
            <GitBranch className="h-3 w-3" />
            {typeLabel}
          </span>
          <span className="inline-flex items-center gap-1 text-slate-500">
            <Search className="h-3 w-3" />
            {trace.reranked_chunks ?? 0} chunks
          </span>
          <span className="inline-flex items-center gap-1 text-emerald-700">
            <ShieldCheck className="h-3 w-3" />
            {trace.verified_citations ?? 0} verified citations
          </span>
          {trace.latency_ms != null && (
            <span className="hidden items-center gap-1 font-mono text-[11px] text-slate-400 sm:inline-flex">
              <Timer className="h-3 w-3" />
              {trace.latency_ms}ms
            </span>
          )}
        </div>
        <ChevronDown
          className={`h-3.5 w-3.5 shrink-0 text-slate-400 transition-transform duration-200 ${
            expanded ? 'rotate-180' : ''
          }`}
        />
      </button>

      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
            className="overflow-hidden"
          >
            <div className="space-y-2.5 border-t border-slate-200/70 px-3 py-2.5 text-xs">
              {subQueries.length > 0 && (
                <div>
                  <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                    Decomposed Sub-Queries
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {subQueries.map((sq, idx) => (
                      <span
                        key={`${idx}-${sq}`}
                        className="rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-slate-700"
                      >
                        {sq}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {steps.length > 0 && (
                <div>
                  <p className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                    Execution Graph Steps
                  </p>
                  <div className="flex flex-wrap items-center gap-1.5 font-mono text-[11px] text-slate-600">
                    {steps.map((step, idx) => (
                      <span key={`${idx}-${step}`} className="inline-flex items-center gap-1">
                        <span className="rounded bg-slate-200/70 px-1.5 py-0.5">{step}</span>
                        {idx < steps.length - 1 && <span className="text-slate-400">→</span>}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

