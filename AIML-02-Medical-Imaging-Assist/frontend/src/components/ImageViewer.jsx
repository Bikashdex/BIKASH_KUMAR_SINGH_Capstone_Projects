import { useEffect, useRef, useState } from 'react'

export default function ImageViewer({ study }) {
  const canvasRef = useRef(null)
  const [showHeatmap, setShowHeatmap] = useState(true)
  const [opacity, setOpacity] = useState(60)

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (!study.previewUrl) {
      ctx.fillStyle = '#666'
      ctx.font = '14px system-ui'
      ctx.fillText(
        'Browser preview not available for this file type (e.g. DICOM).',
        10,
        24,
      )
      ctx.fillText('The API still processed it — see tags below.', 10, 44)
      return
    }

    const img = new Image()
    img.onload = () => {
      const scale = Math.min(canvas.width / img.width, canvas.height / img.height, 1)
      const w = img.width * scale
      const h = img.height * scale
      const x = (canvas.width - w) / 2
      const y = (canvas.height - h) / 2
      ctx.globalAlpha = 1
      ctx.drawImage(img, x, y, w, h)

      if (showHeatmap && study.heatmapUrl) {
        const heat = new Image()
        heat.onload = () => {
          ctx.globalAlpha = opacity / 100
          ctx.drawImage(heat, x, y, w, h)
          ctx.globalAlpha = 1
        }
        heat.src = study.heatmapUrl
      }
    }
    img.src = study.previewUrl
  }, [study, showHeatmap, opacity])

  const tags = study.result.dicom_tags

  return (
    <div className="image-viewer">
      <canvas ref={canvasRef} width={640} height={520} className="viewer-canvas" />
      <div className="viewer-controls">
        <label>
          <input
            type="checkbox"
            checked={showHeatmap && !!study.heatmapUrl}
            disabled={!study.heatmapUrl}
            onChange={(e) => setShowHeatmap(e.target.checked)}
          />{' '}
          heatmap overlay
          {!study.heatmapUrl && <span className="hint"> (none from API yet)</span>}
        </label>
        <label>
          opacity{' '}
          <input
            type="range"
            min="0"
            max="100"
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
          />{' '}
          {opacity}%
        </label>
      </div>
      {tags && (
        <pre className="tags">{JSON.stringify(tags, null, 2)}</pre>
      )}
    </div>
  )
}
