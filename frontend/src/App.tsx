import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import type { Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { motion, AnimatePresence } from 'framer-motion'
import axios from 'axios'
import './App.css'

interface Message {
  role: 'user' | 'assistant'
  content: string
  showGraph?: boolean
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // История чата
  useEffect(() => {
    const saved = localStorage.getItem('chatHistory')
    if (saved) setMessages(JSON.parse(saved))
  }, [])

  useEffect(() => {
    localStorage.setItem('chatHistory', JSON.stringify(messages))
  }, [messages])

  // Скролл вниз
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Авторесайз textarea
  useEffect(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      const maxHeight = 120
      const newHeight = Math.min(textarea.scrollHeight, maxHeight)
      textarea.style.height = `${newHeight}px`
    }
  }, [input])

  const sendMessage = async () => {
    if (!input.trim() || loading) return
    const userMsg: Message = { role: 'user', content: input }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)

    try {
      const res = await axios.post<{ answer: string; showGraph?: boolean; graphData?: any }>(
        'http://127.0.0.1:8000/api/chat/',
        { question: input }
      )

      const assistantMsg: Message = {
        role: 'assistant',
        content: res.data.answer,
        showGraph: res.data.showGraph || false
      }
      setMessages(prev => [...prev, assistantMsg])
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Ошибка соединения.' }])
    } finally {
      setLoading(false)
    }
  }

  const codeComponent = ({ inline, className, children, ...props }: any) => {
    const match = /language-(\w+)/.exec(className || '')
    return !inline && match ? (
      <SyntaxHighlighter
        style={vscDarkPlus}
        language={match[1]}
        PreTag="div"
        {...props}
      >
        {String(children).replace(/\n$/, '')}
      </SyntaxHighlighter>
    ) : (
      <code className={className} {...props}>
        {children}
      </code>
    )
  }

  // 🔹 Потом объявляем markdownComponents
  const markdownComponents: Components = {
    h1: ({ children }) => <h1 className="md-h1">{children}</h1>,
    h2: ({ children }) => <h2 className="md-h2">{children}</h2>,
    h3: ({ children }) => <h3 className="md-h3">{children}</h3>,
    p: ({ children }) => <p className="md-p">{children}</p>,
    strong: ({ children }) => <strong className="md-strong">{children}</strong>,
    em: ({ children }) => <em className="md-em">{children}</em>,
    ul: ({ children }) => <ul className="md-ul">{children}</ul>,
    ol: ({ children }) => <ol className="md-ol">{children}</ol>,
    li: ({ children }) => <li className="md-li">{children}</li>,
    blockquote: ({ children }) => (
      <blockquote className="md-blockquote">{children}</blockquote>
    ),
    a: ({ children, href }) => (
      <a className="md-link" href={href} target="_blank" rel="noreferrer">
        {children}
      </a>
    ),
    table: ({ children }) => <table className="md-table">{children}</table>,
    thead: ({ children }) => <thead className="md-thead">{children}</thead>,
    tbody: ({ children }) => <tbody className="md-tbody">{children}</tbody>,
    tr: ({ children }) => <tr className="md-tr">{children}</tr>,
    th: ({ children }) => <th className="md-th">{children}</th>,
    td: ({ children }) => <td className="md-td">{children}</td>,
    code: codeComponent, // теперь доступна
  }

  return (
    <div className="app">
      <div className="container">

        {/* HEADER */}
        <header className="header">
          <div className="header-inner">
            <div className="logo">
              <div className="logo-icon">
                <img src="src/assets/icons/broke-icon.png" alt="" className="robot-icon" />
              </div>
              <div>
                <span>Finance Advisor</span>
                <small>AI-консультант по инвестициям</small>
              </div>
            </div>
            <div className="user">
              <span>Пользователь</span>
              <div className="avatar">
                <img src="src/assets/icons/user-icon.png" alt="" className="user-icon" />
              </div>
            </div>
          </div>
        </header>

        {/* MESSAGES */}
        <div className="messages-container">
          <div className="messages">
            {messages.length === 0 && <div className="empty">Задайте финансовый вопрос...</div>}

            <AnimatePresence>
              {messages.map((msg, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`message-bubble ${msg.role}`}
                >
                  {msg.role === 'assistant' && <div className="ai-icon"> <img src="src/assets/icons/robot-icon.png" alt="" className="robot-icon-baby" /> </div>}
                  <div className="bubble-content">
                    
                    {msg.role === 'assistant' ? (
                      <div className="bubble-content">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={markdownComponents}
                        >
                          {msg.content}
                        </ReactMarkdown>

                        {/* {msg.showGraph && (
                          <motion.button
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="graph-button-inline"
                            onClick={() => alert('График будет здесь!')}
                          >
                            <img src="src/assets/icons/graph-icon.png" alt="" className="graph-icon" />
                          </motion.button>
                        )} */}
                      </div>
                    ) : (
                      <p>{msg.content}</p>
                    )}
                  
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {loading && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="message-bubble assistant">
                <div className="ai-icon">
                  <img src="src/assets/icons/robot-icon.png" alt="" className="robot-icon-baby" />
                </div>
                <div className="bubble-content typing">
                  <span></span><span></span><span></span>
                </div>
              </motion.div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* ПОЛЕ ВВОДА */}
        <div className="input-container">
          <div className="input-inner">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  sendMessage()
                }
              }}
              placeholder="Как вам помочь?"
              rows={1}
              disabled={loading}
            />
            <button onClick={sendMessage} disabled={loading} className="send-button">
              <img src="src/assets/icons/send-icon.png" alt="" className="send-icon" />
            </button>
          </div>
        </div>

        {/* FOOTER */}
        <footer className="footer">
          AI может допускать ошибки. Проверяйте важную информацию.
        </footer>

      </div>
    </div>
  )
}

export default App