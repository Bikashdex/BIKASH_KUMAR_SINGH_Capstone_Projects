const DECISION_STYLE = {
  approved: 'badge-ok',
  rejected: 'badge-bad',
  pending: 'badge-warn',
}

export default function StudyList({ studies, selectedId, onSelect }) {
  if (studies.length === 0) {
    return (
      <div className="study-list">
        <h2>Studies</h2>
        <p className="empty">No studies yet this session.</p>
      </div>
    )
  }
  return (
    <div className="study-list">
      <h2>Studies ({studies.length})</h2>
      <ul>
        {studies.map((s) => (
          <li key={s.id}>
            <button
              className={s.id === selectedId ? 'study selected' : 'study'}
              onClick={() => onSelect(s.id)}
            >
              <span className="study-name">{s.fileName}</span>
              <span className="study-uid">{s.result.study_uid}</span>
              <span>
                <span className={s.result.prediction === 'opacity' ? 'badge badge-bad' : 'badge badge-ok'}>
                  {s.result.prediction} · {(s.result.probability * 100).toFixed(1)}%
                </span>{' '}
                <span className={`badge ${DECISION_STYLE[s.decision || 'pending']}`}>
                  {s.decision || 'pending'}
                </span>
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
