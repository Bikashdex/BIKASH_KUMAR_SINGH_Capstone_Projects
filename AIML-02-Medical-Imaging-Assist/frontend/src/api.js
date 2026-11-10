export async function fetchHealth() {
  const r = await fetch('/health')
  if (!r.ok) throw new Error(`/health returned ${r.status}`)
  return r.json()
}

export async function fetchModelInfo() {
  const r = await fetch('/model-info')
  if (!r.ok) throw new Error(`/model-info returned ${r.status}`)
  return r.json()
}

export async function postDecision(entry) {
  const r = await fetch('/audit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(entry),
  })
  if (!r.ok) throw new Error(`audit failed (${r.status}): ${await r.text()}`)
  return r.json()
}

export async function postPredict(file, studyUid) {
  const body = new FormData()
  body.append('file', file)
  if (studyUid) body.append('study_uid', studyUid)
  const r = await fetch('/predict', { method: 'POST', body })
  if (!r.ok) {
    const detail = await r.text()
    throw new Error(`predict failed (${r.status}): ${detail}`)
  }
  return r.json()
}
