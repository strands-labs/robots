/** which extra field a training provider needs, and what to say about it. */
export interface ExtraField {
  /** the spec key, sent verbatim in the submit body */
  key: 'embodiment'
  label: string
  placeholder: string
  /** shown under the input: what it is for, in the words the trainer uses */
  say: string
  /** true when validate() refuses without it — the form can then say so before submitting */
  required: boolean
}

const EMBODIMENT: ExtraField = {
  key: 'embodiment',
  label: 'embodiment',
  placeholder: 'new_embodiment',
  say: 'GR00T needs an embodiment tag (--embodiment_tag): which robot body the data came from',
  required: true,
}

/** The extra fields the chosen provider needs. Unknown providers get none. */
export function extraFields(provider: string): ExtraField[] {
  return provider === 'groot' ? [EMBODIMENT] : []
}

/** What is still missing for this provider, in one operator sentence — or '' when nothing is. */
export function missingForProvider(provider: string, values: Record<string, string>): string {
  const missing = extraFields(provider).filter(f => f.required && !(values[f.key] || '').trim())
  if (!missing.length) return ''
  return missing.map(f => f.say).join(' · ')
}
