const form = document.querySelector('#pipelineForm');
const runButton = document.querySelector('#runButton');
const stopButton = document.querySelector('#stopButton');
const resetButton = document.querySelector('#resetButton');
const downloadButton = document.querySelector('#downloadButton');
const formMessage = document.querySelector('#formMessage');
const connectionText = document.querySelector('#connectionText');
const pipelineTrack = document.querySelector('#pipelineTrack');
const progressBar = document.querySelector('#progressBar');
const progressFill = document.querySelector('#progressFill');
const progressLabel = document.querySelector('#progressLabel');
const progressValue = document.querySelector('#progressValue');
const progressSteps = document.querySelector('#progressSteps');
const progressDetail = document.querySelector('#progressDetail');
const etaLabel = document.querySelector('#etaLabel');
const fileList = document.querySelector('#fileList');
const fileCount = document.querySelector('#fileCount');
const preview = document.querySelector('#preview');
const logOutput = document.querySelector('#logOutput');
const logState = document.querySelector('#logState');
const validationPanel = document.querySelector('#validationPanel');
const validationMessage = document.querySelector('#validationMessage');
const validationDetails = document.querySelector('#validationDetails');
const statusWarning = document.querySelector('#statusWarning');
const hardwareDefaultsButton = document.querySelector('#hardwareDefaultsButton');
const hardwareDefaultsMessage = document.querySelector('#hardwareDefaultsMessage');
const modelJob = document.querySelector('#modelJob');
const helpModal = document.querySelector('#helpModal');
const helpModalTitle = document.querySelector('#helpModalTitle');
const helpModalText = document.querySelector('#helpModalText');

const stepIcons = { features: '1', validate: '✓', models: '2', reports: '▧', inference: '3' };
let uploadJobId = '';
let featureFileName = '';
let trainingUploadCount = 0;
let inferenceUploadCount = 0;
let pendingUploads = 0;
let latestState = null;
let pollTimer = null;
let runStartedAt = null;
let previousRunStatus = null;

function value(id) { return document.querySelector(`#${id}`).value.trim(); }
function escapeHtml(text) { return String(text).replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char])); }

function setWorkflow(workflow) {
  document.querySelectorAll('.workflow-option').forEach((option) => option.classList.toggle('active', option.dataset.workflow === workflow));
  document.querySelector('#trainingImagesField').classList.toggle('hidden', !['features_labelled', 'full'].includes(workflow));
  document.querySelector('#inferenceImagesField').classList.toggle('hidden', !['features_unlabelled', 'inference', 'full'].includes(workflow));
  document.querySelector('#featureFileField').classList.toggle('hidden', !['train', 'inference', 'full'].includes(workflow));
  document.querySelector('#modelJobField').classList.toggle('hidden', workflow !== 'inference');
  document.querySelector('#modelFields').classList.toggle('hidden', !['train', 'full'].includes(workflow));
  document.querySelector('#replicationField').classList.toggle('hidden', !['train', 'full'].includes(workflow));
  document.querySelector('#voxelFields').classList.toggle('hidden', workflow === 'train');
}

document.querySelectorAll('input[name="workflow"]').forEach((input) => input.addEventListener('change', (event) => setWorkflow(event.target.value)));

function renderModelJobs(jobs) {
  const selected = modelJob.value;
  if (!jobs?.length) {
    modelJob.innerHTML = '<option value="">No trained model jobs available</option>';
    return;
  }
  modelJob.innerHTML = `<option value="">Select a training job</option>${jobs.map((job) => `<option value="${escapeHtml(job.job_id)}">${escapeHtml(job.job_id.slice(0, 8))} · ${escapeHtml(job.workflow)} · ${escapeHtml(job.status)}</option>`).join('')}`;
  if ([...modelJob.options].some((option) => option.value === selected)) modelJob.value = selected;
}

async function loadDefaults() {
  const response = await fetch('/api/defaults', { cache: 'no-store' });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || 'Could not load defaults');
  const config = payload.config;
  document.querySelector('#versionLabel').textContent = `v${payload.version}`;
  document.querySelector('#workers').value = config.workers;
  document.querySelector('#topFeatures').value = config.top_features;
  document.querySelector('#correlationThreshold').value = config.correlation_threshold;
  document.querySelector('#voxelSizeX').value = config.voxel_size_x;
  document.querySelector('#voxelSizeY').value = config.voxel_size_y;
  document.querySelector('#voxelSizeZ').value = config.voxel_size_z;
  document.querySelector('#replicationUnit').value = config.replication_unit;
  document.querySelector('#learner').value = config.learner;
  renderModelJobs(payload.model_jobs);
}

async function uploadFiles(input, kind, statusElement) {
  const files = [...input.files];
  if (!files.length) return;
  pendingUploads += 1;
  try {
    for (let index = 0; index < files.length; index += 1) {
      statusElement.textContent = `Uploading ${index + 1} of ${files.length}: ${files[index].name}`;
      const headers = {
        'Content-Type': files[index].type || 'application/octet-stream',
        'X-Upload-Name': files[index].name,
        'X-Upload-Kind': kind,
      };
      if (uploadJobId) headers['X-Job-ID'] = uploadJobId;
      const response = await fetch('/api/upload', { method: 'POST', headers, body: files[index] });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'Upload failed');
      uploadJobId = payload.job_id;
    }
    if (kind === 'training') trainingUploadCount += files.length;
    else inferenceUploadCount += files.length;
    const total = kind === 'training' ? trainingUploadCount : inferenceUploadCount;
    statusElement.textContent = `${total} image${total === 1 ? '' : 's'} uploaded to job ${uploadJobId.slice(0, 8)}.`;
  } catch (error) {
    statusElement.textContent = error.message;
  } finally {
    pendingUploads -= 1;
    input.value = '';
  }
}

async function uploadFeatureFile(input) {
  const file = input.files[0];
  if (!file) return;
  pendingUploads += 1;
  const statusElement = document.querySelector('#featureFileStatus');
  statusElement.textContent = `Uploading ${file.name}…`;
  try {
    const headers = { 'Content-Type': file.type || 'application/octet-stream', 'X-Upload-Name': file.name };
    if (uploadJobId) headers['X-Job-ID'] = uploadJobId;
    const response = await fetch('/api/upload-feature', { method: 'POST', headers, body: file });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Upload failed');
    uploadJobId = payload.job_id;
    featureFileName = payload.file;
    statusElement.textContent = `${payload.file} uploaded to job ${uploadJobId.slice(0, 8)}.`;
  } catch (error) {
    statusElement.textContent = error.message;
  } finally {
    pendingUploads -= 1;
    input.value = '';
  }
}

document.querySelector('#trainingFilePicker').addEventListener('change', (event) => uploadFiles(event.target, 'training', document.querySelector('#trainingFileStatus')));
document.querySelector('#inferenceFilePicker').addEventListener('change', (event) => uploadFiles(event.target, 'inference', document.querySelector('#inferenceFileStatus')));
document.querySelector('#featureFilePicker').addEventListener('change', (event) => uploadFeatureFile(event.target));

function configFromForm() {
  return {
    workflow: document.querySelector('input[name="workflow"]:checked').value,
    job_id: uploadJobId,
    feature_file: featureFileName,
    model_job_id: modelJob.value,
    workers: Number(value('workers')),
    top_features: Number(value('topFeatures')),
    correlation_threshold: Number(value('correlationThreshold')),
    voxel_size_x: Number(value('voxelSizeX')),
    voxel_size_y: Number(value('voxelSizeY')),
    voxel_size_z: Number(value('voxelSizeZ')),
    replication_unit: value('replicationUnit'),
    all_learners: document.querySelector('input[name="learner_mode"]:checked').value === 'all',
    learner: value('learner'),
  };
}

function renderValidation(report) {
  if (!report) {
    validationPanel.classList.remove('is-invalid', 'is-valid');
    validationMessage.textContent = 'Input validation runs before processing.';
    validationDetails.innerHTML = '';
    return;
  }
  validationPanel.classList.toggle('is-invalid', report.ok === false);
  validationPanel.classList.toggle('is-valid', report.ok === true);
  validationMessage.textContent = report.ok ? 'All uploaded inputs passed preflight.' : 'Preflight needs attention.';
  const sections = Object.entries(report).filter(([, section]) => section && typeof section === 'object').map(([name, section]) => {
    if (section.features_read !== undefined) {
      const warnings = (section.warnings || []).map((warning) => `<li>${escapeHtml(warning)}</li>`).join('');
      return `<section class="validation-section"><strong>${escapeHtml(name.replace('_', ' '))}</strong><p>${section.rows} rows · ${section.features_read} features</p>${warnings ? `<ul>${warnings}</ul>` : ''}</section>`;
    }
    const labels = Object.entries(section.images_per_label || {}).map(([label, count]) => `<li>${escapeHtml(label)}: ${count}</li>`).join('');
    return `<section class="validation-section"><strong>${escapeHtml(name.replace('_', ' '))}</strong><p>${section.total_images || 0} image(s)</p>${labels ? `<ul>${labels}</ul>` : ''}</section>`;
  });
  validationDetails.innerHTML = sections.join('');
}

async function post(path, body = {}) {
  const response = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || 'Request failed');
  return payload;
}

hardwareDefaultsButton.addEventListener('click', async () => {
  try {
    const response = await fetch('/api/hardware-defaults', { cache: 'no-store' });
    const payload = await response.json();
    document.querySelector('#workers').value = payload.workers;
    hardwareDefaultsMessage.textContent = `Using ${payload.workers} of ${payload.cpu_count} CPUs visible to this container.`;
  } catch (error) {
    hardwareDefaultsMessage.textContent = error.message;
  }
});

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(seconds) {
  const rounded = Math.max(1, Math.round(seconds));
  if (rounded < 60) return `${rounded}s`;
  return `${Math.floor(rounded / 60)}m ${rounded % 60}s`;
}

function renderEta(status, progress) {
  if (!['starting', 'running'].includes(status)) {
    runStartedAt = null;
    etaLabel.textContent = status === 'completed' ? 'Complete' : 'ETA —';
    return;
  }
  if (!runStartedAt || !['starting', 'running'].includes(previousRunStatus)) runStartedAt = performance.now();
  const percent = Number(progress?.percent) || 0;
  if (percent < 1) { etaLabel.textContent = 'ETA calculating…'; return; }
  const elapsed = (performance.now() - runStartedAt) / 1000;
  etaLabel.textContent = `ETA ${formatDuration(elapsed * ((100 - percent) / percent))}`;
}

function renderProgress(progress, steps, status) {
  const percent = Math.max(0, Math.min(100, Number(progress?.percent) || 0));
  progressFill.style.width = `${percent}%`;
  progressFill.classList.toggle('is-running', ['starting', 'running'].includes(status));
  progressBar.setAttribute('aria-valuenow', percent);
  progressLabel.textContent = progress?.label || 'Ready';
  progressValue.textContent = `${percent}%`;
  progressSteps.textContent = progress?.total ? `${progress.completed} of ${progress.total} stages complete` : 'No stages started';
  const current = (steps || []).find((step) => step.status === 'running');
  progressDetail.textContent = current?.detail || progress?.detail || 'The active stage will appear here.';
}

function renderSteps(steps) {
  if (!steps?.length) {
    pipelineTrack.innerHTML = '<div class="empty-state"><span class="empty-mark">◌</span><p>Your pipeline is ready.</p><small>Upload input files and start a job.</small></div>';
    return;
  }
  pipelineTrack.innerHTML = steps.map((step) => {
    const icon = step.status === 'completed' ? '✓' : step.status === 'failed' ? '!' : step.status === 'cancelled' ? '−' : (stepIcons[step.id] || '·');
    return `<article class="pipeline-step is-${step.status}"><div class="step-mark">${icon}</div><h3>${escapeHtml(step.label)}</h3><p>${escapeHtml(step.detail || '')}</p></article>`;
  }).join('');
}

function renderFiles(artifacts, jobId) {
  const files = artifacts || [];
  fileCount.textContent = `${files.length}${files.length >= 450 ? '+' : ''} file${files.length === 1 ? '' : 's'}`;
  if (!files.length) {
    fileList.innerHTML = '<div class="empty-state compact"><span class="empty-mark">⌁</span><p>No output yet.</p></div>';
    return;
  }
  fileList.innerHTML = files.map((file) => `<button class="file-row" data-path="${escapeHtml(file.path)}" type="button"><span class="file-icon">${file.kind === 'image' ? '▧' : file.kind === 'text' ? '≡' : '·'}</span><span class="file-name">${escapeHtml(file.name)}<small class="file-meta">${escapeHtml(file.path)} · ${formatBytes(file.size)}</small></span></button>`).join('');
  fileList.querySelectorAll('.file-row').forEach((row) => row.addEventListener('click', () => inspectFile(row, jobId)));
}

async function inspectFile(row, jobId) {
  const path = row.dataset.path;
  const file = (latestState.artifacts || []).find((item) => item.path === path);
  if (!file) return;
  const encoded = `job=${encodeURIComponent(jobId)}&path=${encodeURIComponent(path)}`;
  if (file.kind === 'image') preview.innerHTML = `<div class="preview-title">${escapeHtml(path)}</div><img src="/api/artifact?${encoded}" alt="${escapeHtml(file.name)}">`;
  else if (file.kind === 'html') preview.innerHTML = `<div class="preview-title">${escapeHtml(path)}</div><iframe src="/api/artifact?${encoded}"></iframe>`;
  else if (file.kind === 'text') {
    const response = await fetch(`/api/preview?${encoded}`);
    const payload = await response.json();
    preview.innerHTML = `<div class="preview-title">${escapeHtml(payload.name)}</div><pre>${escapeHtml(payload.content)}</pre>`;
  } else preview.innerHTML = `<div class="preview-placeholder"><p>${escapeHtml(file.name)}</p><small>Use Download results ZIP to retrieve this file.</small></div>`;
}

function updateDocumentTitle(status, progress) {
  document.title = ['starting', 'running'].includes(status) ? `${progress?.percent || 0}% · MicroICS` : status === 'completed' ? 'Analysis complete · MicroICS' : 'MicroICS · Ready';
}

function renderState(nextState) {
  latestState = nextState;
  updateDocumentTitle(nextState.status, nextState.progress);
  renderEta(nextState.status, nextState.progress);
  renderModelJobs(nextState.model_jobs);
  connectionText.textContent = nextState.status === 'running' ? 'Processing' : nextState.status === 'starting' ? 'Starting' : nextState.status === 'completed' ? 'Complete' : nextState.status === 'failed' ? 'Needs attention' : nextState.status === 'cancelled' ? 'Stopped' : 'Ready';
  renderSteps(nextState.steps);
  renderProgress(nextState.progress, nextState.steps, nextState.status);
  renderValidation(nextState.validation);
  renderFiles(nextState.artifacts, nextState.job_id);
  logOutput.textContent = (nextState.logs || []).join('\n') || 'No job started.';
  logOutput.scrollTop = logOutput.scrollHeight;
  logState.textContent = nextState.status[0].toUpperCase() + nextState.status.slice(1);
  const running = ['starting', 'running'].includes(nextState.status);
  runButton.classList.toggle('hidden', running);
  stopButton.classList.toggle('hidden', !running);
  resetButton.classList.toggle('hidden', nextState.status === 'idle');
  statusWarning.classList.toggle('hidden', !['failed', 'cancelled'].includes(nextState.status));
  downloadButton.classList.toggle('hidden', !nextState.download_url);
  if (nextState.download_url) downloadButton.href = nextState.download_url;
  if (nextState.status === 'completed') formMessage.textContent = `Job ${nextState.job_id.slice(0, 8)} completed. Inspect the files on the right or download the results ZIP.`;
  if (nextState.status === 'failed') formMessage.textContent = nextState.error || 'The job failed. Review its log.';
  previousRunStatus = nextState.status;
}

async function poll(refresh = false) {
  try {
    const response = await fetch(`/api/state${refresh ? '?refresh=1' : ''}`, { cache: 'no-store' });
    renderState(await response.json());
  } catch (error) {
    connectionText.textContent = 'Offline';
  }
  clearTimeout(pollTimer);
  pollTimer = setTimeout(() => poll(false), 900);
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  formMessage.textContent = '';
  try {
    if (pendingUploads) throw new Error('Please wait for uploads to finish.');
    if (!uploadJobId) throw new Error('Upload the required input files first.');
    const config = configFromForm();
    const preflight = await post('/api/preflight', config);
    renderValidation(preflight.report);
    if (!preflight.report.ok) throw new Error('Preflight validation failed.');
    await post('/api/run', config);
    await poll();
  } catch (error) {
    formMessage.textContent = error.message;
  }
});

stopButton.addEventListener('click', async () => { try { await post('/api/stop'); } catch (error) { formMessage.textContent = error.message; } });
resetButton.addEventListener('click', async () => {
  try {
    renderState(await post('/api/reset'));
    uploadJobId = '';
    featureFileName = '';
    trainingUploadCount = 0;
    inferenceUploadCount = 0;
    document.querySelector('#trainingFileStatus').textContent = 'No training images uploaded.';
    document.querySelector('#inferenceFileStatus').textContent = 'No inference images uploaded.';
    document.querySelector('#featureFileStatus').textContent = 'No feature table uploaded.';
    formMessage.textContent = 'Ready for a new isolated job.';
  } catch (error) { formMessage.textContent = error.message; }
});
document.querySelector('#refreshButton').addEventListener('click', () => poll(true));

function syncLearnerMode() {
  const mode = document.querySelector('input[name="learner_mode"]:checked').value;
  document.querySelector('#learner').disabled = mode === 'all';
  document.querySelectorAll('.learner-option').forEach((option) => option.classList.toggle('active', option.dataset.learnerMode === mode));
}
document.querySelectorAll('input[name="learner_mode"]').forEach((input) => input.addEventListener('change', syncLearnerMode));

function closeHelpModal() { helpModal.classList.add('hidden'); }
document.querySelectorAll('[data-help-title]').forEach((button) => button.addEventListener('click', () => {
  helpModalTitle.textContent = button.dataset.helpTitle;
  helpModalText.textContent = button.dataset.helpText;
  helpModal.classList.remove('hidden');
}));
document.querySelector('#closeHelpModal').addEventListener('click', closeHelpModal);
document.querySelector('#dismissHelpModal').addEventListener('click', closeHelpModal);
document.querySelector('[data-close-help-modal]').addEventListener('click', closeHelpModal);

async function boot() {
  try {
    await loadDefaults();
    syncLearnerMode();
    setWorkflow('features_labelled');
    await poll();
  } catch (error) {
    connectionText.textContent = 'Offline';
    formMessage.textContent = error.message;
  }
}

boot();
