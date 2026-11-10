import { useEffect, useRef, useState } from 'react'
import { fetchHealth, fetchModelInfo, postDecision, postPredict } from './api.js'
import StudyList from './components/StudyList.jsx'
import ImageViewer from './components/ImageViewer.jsx'
import DecisionBar from './components/DecisionBar.jsx'

const DISCLAIMER = 'EDUCATION / RESEARCH PROTOTYPE — NOT FOR CLINICAL USE'

let nextId = 1

export default function App() {
  const [health, setHealth] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [apiError, setApiError] = useState(null)
  const [studies, setStudies] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState(null)
  const fileInput = useRef(null)

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch((e) => setApiError(e.message))
    fetchModelInfo()
      .then(setModelInfo)
      .catch(() => {})
  }, [])

  const selected = studies.find((s) => s.id === selectedId) || null

  async function handleFile(file) {
    if (!file) return
    setUploading(true)
    setUploadError(null)
    try {
      const result = await postPredict(file)
      const study = {
        id: nextId++,
        fileName: file.name,
        previewUrl: file.type.startsWith('image/') ? URL.createObjectURL(file) : null,
        heatmapUrl: result.overlay_url || null,
        decision: null,
        reason: '',
        result,
      }
      setStudies((prev) => [...prev, study])
      setSelectedId(study.id)
    } catch (e) {
      setUploadError(e.message)
    } finally {
      setUploading(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  async function handleDecide(decision, reason) {
    const study = studies.find((s) => s.id === selectedId)
    if (!study) return
    const id = study.id
    setStudies((prev) => prev.map((s) => (s.id === id ? { ...s, decision, reason, auditState: 'saving' } : s)))
    try {
      const saved = await postDecision({
        study_uid: study.result.study_uid,
        decision,
        reason: reason || null,
        probability: study.result.probability,
        threshold_used: study.result.threshold_used,
        model_version: study.result.model_version,
      })
      setStudies((prev) =>
        prev.map((s) =>
          s.id === id ? { ...s, auditState: 'saved', auditId: saved.recorded.id, auditError: null } : s,
        ),
      )
    } catch (e) {
      setStudies((prev) => prev.map((s) => (s.id === id ? { ...s, auditState: 'error', auditError: e.message } : s)))
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>AIML-02 · Medical Imaging Assist</h1>
        <div className="status">
          {apiError && <span className="badge badge-bad">API offline ({apiError})</span>}
          {!apiError && health && (
            <span className={health.model_loaded ? 'badge badge-ok' : 'badge badge-warn'}>
              API ok · model {health.model_loaded ? `loaded on ${health.device}` : 'NOT loaded'}
            </span>
          )}
          {!apiError && !health && <span className="badge">connecting…</span>}
          {/* "vunversioned" is not a version string: the API reports the literal "unversioned" until
              MODEL_VERSION is set. */}
          {modelInfo && (
            <span className="badge">
              {modelInfo.version === 'unversioned' ? 'unversioned' : `v${modelInfo.version}`}
            </span>
          )}
        </div>
      </header>

      <div className="disclaimer-banner">{DISCLAIMER}</div>

      <div className="upload-row">
        <input
          ref={fileInput}
          type="file"
          accept=".dcm,image/png,image/jpeg"
          onChange={(e) => handleFile(e.target.files[0])}
          disabled={uploading}
        />
        <span className="hint">
          chest X-ray (DICOM / PNG / JPG) → POST /predict
        </span>
        {uploading && <span className="badge">running inference…</span>}
        {uploadError && <span className="badge badge-bad">{uploadError}</span>}
      </div>

      <div className="workspace">
        <aside className="sidebar">
          <StudyList studies={studies} selectedId={selectedId} onSelect={setSelectedId} />
        </aside>
        <main className="viewer-pane">
          {selected ? (
            <>
              <div className="viewer-banner">{DISCLAIMER}</div>
              <ImageViewer study={selected} />
              <DecisionBar study={selected} onDecide={handleDecide} />
            </>
          ) : (
            <p className="empty">Upload a study to see it here.</p>
          )}
        </main>
      </div>
    </div>
  )
}
