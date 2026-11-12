import { useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeKatex from 'rehype-katex'
import remarkGfm from 'remark-gfm'

/**
 * Convert standalone [1], [2] markers outside code spans/blocks into #cite-N links
 * so ReactMarkdown renders them as accessible interactive buttons.
 */
function injectCitationLinks(markdown) {
  if (!markdown) return ''
  // Split by fenced code blocks and inline code spans so code samples remain untouched
  const segments = markdown.split(/(```[\s\S]*?```|`[^`\n]+`)/g)
  return segments
    .map((segment, index) => {
      if (index % 2 === 1) return segment
      return segment.replace(/\[(\d+)\](?!\()/g, '[$1](#cite-$1)')
    })
    .join('')
}

export function MarkdownWithCitations({ content, citations = [], onSelectCitation }) {
  const citationByIndex = useMemo(() => {
    const map = new Map()
    for (const item of citations || []) {
      map.set(Number(item.citation_index), item)
    }
    return map
  }, [citations])

  const processed = useMemo(() => injectCitationLinks(content), [content])

  return (
    <div className="prose-research">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeKatex]}
        components={{
          a: ({ href, children, ...props }) => {
            if (href && href.startsWith('#cite-')) {
              const idx = Number(href.slice(6))
              const citation = citationByIndex.get(idx)
              if (!citation) {
                return <span className="font-mono text-xs text-slate-500">[{children}]</span>
              }
              const pageLabel =
                citation.page_start === citation.page_end
                  ? `p. ${citation.page_start}`
                  : `pp. ${citation.page_start}-${citation.page_end}`

              return (
                <button
                  type="button"
                  onClick={() => onSelectCitation?.(citation)}
                  title={`${citation.document_title} (${pageLabel}) — Click to open PDF page`}
                  className="mx-0.5 inline-flex -translate-y-0.5 cursor-pointer items-center gap-0.5 rounded-md border border-indigo-200 bg-indigo-50 px-1.5 py-0.5 font-mono text-[11px] font-semibold text-indigo-700 transition-all hover:border-indigo-400 hover:bg-indigo-100 hover:shadow-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
                >
                  {idx}
                </button>
              )
            }
            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="font-medium text-indigo-600 underline decoration-indigo-300 underline-offset-2 hover:text-indigo-700"
                {...props}
              >
                {children}
              </a>
            )
          },
        }}
      >
        {processed}
      </ReactMarkdown>
    </div>
  )
}

