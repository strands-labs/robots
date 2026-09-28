import { useEffect, useState } from 'react'
import { apiBlob } from '../lib/endpoints'

export default function AuthedImg(
  { path, alt, className }: { path: string; alt: string; className?: string },
) {
  const [url, setUrl] = useState('')
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    let alive = true
    let made = ''
    setFailed(false)
    setUrl('')
    void apiBlob(path)
      .then(u => {
        if (!alive) { URL.revokeObjectURL(u); return }   // unmounted mid-flight: still not a leak
        made = u
        setUrl(u)
      })
      .catch(() => { if (alive) setFailed(true) })
    return () => {
      alive = false
      if (made) URL.revokeObjectURL(made)
    }
  }, [path])

  // No glyph for a picture that is merely late, and words instead of a broken icon when it fails.
  if (failed) return <span className="hint small" title={path}>{alt} — unavailable</span>
  if (!url) return <span className="thumb-loading" aria-label={`${alt} loading`} />
  return <img src={url} alt={alt} className={className} loading="lazy" />
}
