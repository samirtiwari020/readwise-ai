import { useCallback, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { AlertCircle, CheckCircle2, Info, X } from 'lucide-react'

import { ToastContext } from './toastContext'

const TOAST_STYLES = {
  success: {
    icon: CheckCircle2,
    ring: 'border-emerald-200 bg-white text-slate-900',
    badge: 'bg-emerald-50 text-emerald-600',
  },
  info: {
    icon: Info,
    ring: 'border-indigo-200 bg-white text-slate-900',
    badge: 'bg-indigo-50 text-indigo-600',
  },
  error: {
    icon: AlertCircle,
    ring: 'border-rose-200 bg-white text-slate-900',
    badge: 'bg-rose-50 text-rose-600',
  },
}

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])

  const dismiss = useCallback((id) => {
    setToasts((prev) => prev.filter((item) => item.id !== id))
  }, [])

  const pushToast = useCallback(
    ({ type = 'info', title, description, duration = 4200 }) => {
      const id = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`
      setToasts((prev) => [...prev.slice(-3), { id, type, title, description }])
      if (duration > 0) {
        setTimeout(() => dismiss(id), duration)
      }
    },
    [dismiss]
  )

  return (
    <ToastContext.Provider value={{ pushToast, dismiss }}>
      {children}
      <div
        aria-live="polite"
        className="pointer-events-none fixed right-4 top-4 z-50 flex w-full max-w-sm flex-col gap-2.5 px-4 sm:px-0"
      >
        <AnimatePresence initial={false}>
          {toasts.map((toast) => {
            const cfg = TOAST_STYLES[toast.type] || TOAST_STYLES.info
            const Icon = cfg.icon
            return (
              <motion.div
                key={toast.id}
                initial={{ opacity: 0, y: -12, scale: 0.96 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -8, scale: 0.96 }}
                transition={{ duration: 0.2, ease: 'easeOut' }}
                className={`pointer-events-auto flex items-start gap-3 rounded-xl border p-3.5 shadow-lg shadow-slate-900/5 ${cfg.ring}`}
                role="status"
              >
                <div className={`mt-0.5 rounded-lg p-1.5 ${cfg.badge}`}>
                  <Icon className="h-4 w-4" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-semibold text-slate-900">{toast.title}</p>
                  {toast.description && (
                    <p className="mt-0.5 text-xs leading-relaxed text-slate-600">
                      {toast.description}
                    </p>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => dismiss(toast.id)}
                  aria-label="Dismiss notification"
                  className="rounded-lg p-1 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
                >
                  <X className="h-4 w-4" />
                </button>
              </motion.div>
            )
          })}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  )
}
