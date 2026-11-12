import {
  BookOpenCheck,
  Cpu,
  Menu,
  PanelRightClose,
  PanelRightOpen,
  Sparkles,
} from 'lucide-react'

export function TopBar({
  health,
  activeSession,
  scopedCount,
  completedCount,
  isPdfOpen,
  onToggleSidebar,
  onTogglePdf,
}) {
  const isHealthy = health?.database_connected
  const gpuLabel = health?.cuda_available
    ? health.device_name?.replace('NVIDIA GeForce ', '').replace(' Laptop GPU', '') || 'CUDA GPU'
    : 'CPU Mode'

  const scopeText =
    completedCount === 0
      ? 'No indexed PDFs yet'
      : scopedCount > 0
        ? `Scoped: ${scopedCount} of ${completedCount} docs`
        : `Searching all ${completedCount} docs`

  return (
    <header className="sticky top-0 z-30 flex h-14 shrink-0 items-center justify-between border-b border-slate-200/80 bg-white/90 px-3 backdrop-blur-md sm:px-5">
      <div className="flex min-w-0 items-center gap-2.5 sm:gap-3">
        <button
          type="button"
          onClick={onToggleSidebar}
          aria-label="Toggle library sidebar"
          className="inline-flex h-9 w-9 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-600 transition-colors hover:bg-slate-50 hover:text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 lg:hidden"
        >
          <Menu className="h-4 w-4" />
        </button>

        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-600 to-violet-600 text-white shadow-sm shadow-indigo-600/20">
            <BookOpenCheck className="h-4 w-4" />
          </div>
          <div className="hidden sm:block">
            <div className="flex items-center gap-1.5">
              <span className="font-display text-sm font-bold tracking-tight text-slate-900">
                Readwise AI
              </span>
              <span className="rounded-md bg-indigo-50 px-1.5 py-0.5 text-[10px] font-semibold text-indigo-700">
                RAG
              </span>
            </div>
            <p className="text-[11px] text-slate-500">Agentic Document Research</p>
          </div>
        </div>

        <div className="hidden h-5 w-px bg-slate-200 md:block" />

        <div className="flex min-w-0 items-center gap-2">
          <span className="truncate text-sm font-medium text-slate-800 max-w-[160px] sm:max-w-[260px] md:max-w-xs">
            {activeSession ? activeSession.title : 'Research Workspace'}
          </span>
          <span className="hidden items-center gap-1 rounded-full border border-slate-200 bg-slate-50 px-2.5 py-0.5 text-[11px] font-medium text-slate-600 sm:inline-flex">
            <Sparkles className="h-3 w-3 text-indigo-500" />
            {scopeText}
          </span>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <div
          title={
            health
              ? `DB: ${isHealthy ? 'Connected' : 'Disconnected'} | Model: ${health.llm_model} | Embed: ${health.embedding_model}`
              : 'Checking backend status...'
          }
          className="hidden items-center gap-1.5 rounded-full border border-slate-200 bg-slate-50/80 px-2.5 py-1 text-xs font-medium text-slate-700 md:inline-flex"
        >
          <span
            className={`h-2 w-2 rounded-full ${
              health === null
                ? 'bg-slate-300 animate-pulse'
                : isHealthy
                  ? 'bg-emerald-500'
                  : 'bg-rose-500'
            }`}
          />
          <Cpu className="h-3.5 w-3.5 text-slate-500" />
          <span>{health ? gpuLabel : 'Connecting...'}</span>
        </div>

        <button
          type="button"
          onClick={onTogglePdf}
          className={`inline-flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs font-semibold transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
            isPdfOpen
              ? 'border-indigo-200 bg-indigo-50 text-indigo-700 hover:bg-indigo-100/80'
              : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50'
          }`}
        >
          {isPdfOpen ? (
            <>
              <PanelRightClose className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Hide Reader</span>
            </>
          ) : (
            <>
              <PanelRightOpen className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">PDF Reader</span>
            </>
          )}
        </button>
      </div>
    </header>
  )
}

