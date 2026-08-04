const form = document.querySelector('#pipelineForm');
const runButton = document.querySelector('#runButton');
const stopButton = document.querySelector('#stopButton');
const resetButton = document.querySelector('#resetButton');
const formMessage = document.querySelector('#formMessage');
const pipelineTrack = document.querySelector('#pipelineTrack');
const fileList = document.querySelector('#fileList');
const fileCount = document.querySelector('#fileCount');
const preview = document.querySelector('#preview');
const logOutput = document.querySelector('#logOutput');
const logState = document.querySelector('#logState');
const cleanupConfirm = document.querySelector('#cleanupConfirm');
const connectionText = document.querySelector('#connectionText');
const progressBar = document.querySelector('#progressBar');
const progressFill = document.querySelector('#progressFill');
const progressLabel = document.querySelector('#progressLabel');
const progressValue = document.querySelector('#progressValue');
const progressSteps = document.querySelector('#progressSteps');
const progressDetail = document.querySelector('#progressDetail');
const etaLabel = document.querySelector('#etaLabel');
const hardwareDefaultsButton = document.querySelector('#hardwareDefaultsButton');
const hardwareDefaultsMessage = document.querySelector('#hardwareDefaultsMessage');
const folderPicker = document.querySelector('#folderPicker');
const folderPathInput = document.querySelector('#folderPathInput');
const folderGoButton = document.querySelector('#folderGoButton');
const folderPickerMessage = document.querySelector('#folderPickerMessage');
const folderList = document.querySelector('#folderList');
const selectFolderButton = document.querySelector('#selectFolderButton');
const helpModal = document.querySelector('#helpModal');
const helpModalTitle = document.querySelector('#helpModalTitle');
const helpModalText = document.querySelector('#helpModalText');
const closeHelpModalButton = document.querySelector('#closeHelpModal');
const dismissHelpModalButton = document.querySelector('#dismissHelpModal');
const cleanupModal = document.querySelector('#cleanupModal');
const cleanupFolderPath = document.querySelector('#cleanupFolderPath');
const cleanupFolderSummary = document.querySelector('#cleanupFolderSummary');
const closeCleanupModalButton = document.querySelector('#closeCleanupModal');
const cancelCleanupButton = document.querySelector('#cancelCleanupButton');
const confirmCleanupButton = document.querySelector('#confirmCleanupButton');
const validationPanel = document.querySelector('#validationPanel');
const validationMessage = document.querySelector('#validationMessage');
const validationDetails = document.querySelector('#validationDetails');
const versionLabel = document.querySelector('#versionLabel');
const demoButton = document.querySelector('#demoButton');
const openOutputButton = document.querySelector('#openOutputButton');
const statusWarning = document.querySelector('#statusWarning');
let latestState = null;
let pollTimer = null;
let syncedConfigKey = null;
let hardwareDefaults = null;
let pendingUploads = 0;
let folderPickerTarget = null;
let selectedFolderPath = null;
let helpReturnFocus = null;
let cleanupReturnFocus = null;
let runStartedAt = null;
let previousRunStatus = null;
let demoPaths = null;

const stepIcons = { prepare: '◆', features: '1', validate: '✓', models: '2', inference: '3' };

function value(id) { return document.querySelector(`#${id}`).value.trim(); }

function updateDocumentTitle(status, progress) {
  if (status === 'running') {
    const percent = Math.max(0, Math.min(100, Number(progress?.percent) || 0));
    const activity = (progress?.label || 'Analysis ongoing').replace(/^Working:\s*/, '');
    document.title = `${percent}% · ${activity} · MicroICS`;
  } else if (status === 'complete') {
    document.title = 'Analysis complete · MicroICS';
  } else if (status === 'failed') {
    document.title = 'Action needed · MicroICS';
  } else if (status === 'cancelled') {
    document.title = 'Analysis stopped · MicroICS';
  } else {
    document.title = 'MicroICS · Ready';
  }
}

function setWorkflow(workflow) {
  document.querySelectorAll('.workflow-option').forEach((option) => option.classList.toggle('active', option.dataset.workflow === workflow));
  const labelledWorkflow = !['inference', 'features_unlabelled'].includes(workflow);
  document.querySelector('#trainingImagesField').classList.toggle('hidden', !labelledWorkflow);
  document.querySelector('#inferenceFields').classList.toggle('hidden', !['full', 'inference', 'features_unlabelled'].includes(workflow));
  document.querySelector('#resultsDir').closest('.field-group').querySelector('small').textContent = ['full', 'features_labelled', 'features_unlabelled'].includes(workflow) ? 'Feature generation replaces existing contents after confirmation.' : workflow === 'train' ? 'Read datafile.tsv and write models in this folder.' : 'Models are read from this folder/models.';
  document.querySelector('#featureFileField').classList.toggle('hidden', ['features_labelled', 'features_unlabelled'].includes(workflow));
  document.querySelector('#modelFields').classList.toggle('hidden', !['full', 'train'].includes(workflow));
  document.querySelector('#voxelFields').classList.toggle('hidden', workflow === 'train');
  cleanupConfirm.classList.toggle('hidden', !['full', 'features_labelled', 'features_unlabelled'].includes(workflow));
}

function syncForm(config) {
  const workflowInput = document.querySelector(`input[name="workflow"][value="${config.workflow}"]`);
  if (!workflowInput) return;
  workflowInput.checked = true;
  setWorkflow(config.workflow);
  document.querySelector('#trainingImages').value = config.training_images || '';
  document.querySelector('#resultsDir').value = config.results_dir || '';
  document.querySelector('#inferenceImages').value = config.inference_images || '';
  document.querySelector('#inferenceOutput').value = config.inference_output || '';
  document.querySelector('#workers').value = config.workers ?? 4;
  document.querySelector('#topFeatures').value = config.top_features ?? 10;
  document.querySelector('#correlationThreshold').value = config.correlation_threshold ?? 0.8;
  document.querySelector('#voxelSizeX').value = config.voxel_size_x ?? 0.13;
  document.querySelector('#voxelSizeY').value = config.voxel_size_y ?? 0.13;
  document.querySelector('#voxelSizeZ').value = config.voxel_size_z ?? 0.5;
  document.querySelector('#replicationUnit').value = config.replication_unit || 'date';
  document.querySelector('#featureFile').value = config.feature_file || '';
  document.querySelector(`input[name="learner_mode"][value="${config.all_learners ? 'all' : 'single'}"]`).checked = true;
  document.querySelector('#learner').value = config.learner || 'rf';
  syncLearnerMode();
  hardwareDefaults = config.cpu_limit && config.memory_limit ? { cpu_limit: config.cpu_limit, memory_limit: config.memory_limit } : null;
}

document.querySelectorAll('input[name="workflow"]').forEach((input) => input.addEventListener('change', (event) => setWorkflow(event.target.value)));

async function loadDefaults() {
  try {
    const response = await fetch('/api/defaults');
    const payload = await response.json();
    const config = payload.config;
    demoPaths = payload.demo || null;
    versionLabel.textContent = `v${payload.version || '—'}`;
    document.querySelector('#trainingImages').value = config.training_images;
    document.querySelector('#resultsDir').value = config.results_dir;
    document.querySelector('#inferenceImages').value = config.inference_images;
    document.querySelector('#inferenceOutput').value = config.inference_output;
    document.querySelector('#workers').value = config.workers;
    document.querySelector('#topFeatures').value = config.top_features;
    document.querySelector('#correlationThreshold').value = config.correlation_threshold || 0.8;
    document.querySelector('#voxelSizeX').value = config.voxel_size_x ?? 0.13;
    document.querySelector('#voxelSizeY').value = config.voxel_size_y ?? 0.13;
    document.querySelector('#voxelSizeZ').value = config.voxel_size_z ?? 0.5;
    document.querySelector('#learner').value = config.learner || 'rf';
    const workflowInput = document.querySelector(`input[name="workflow"][value="${config.workflow}"]`);
    if (workflowInput) {
      workflowInput.checked = true;
      setWorkflow(config.workflow);
    }
  } catch (error) {
    formMessage.textContent = 'Could not connect to the local GUI server.';
  }
}

function closeFolderPicker() {
  folderPicker.classList.add('hidden');
  folderPickerTarget = null;
  selectedFolderPath = null;
}

async function browseFolder(path) {
  folderPickerMessage.textContent = 'Loading folders…';
  folderList.innerHTML = '';
  try {
    const response = await fetch(`/api/browse?path=${encodeURIComponent(path || '')}`, { cache: 'no-store' });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Could not read this folder');
    selectedFolderPath = payload.path;
    folderPathInput.value = payload.path;
    folderPickerMessage.textContent = payload.directories.length ? 'Choose a folder to open, or use the current folder.' : 'This folder has no subfolders. You can use it as the input folder.';
    const entries = [];
    if (payload.parent) entries.push(`<button class="folder-entry parent" data-folder-path="${escapeHtml(payload.parent)}" type="button"><span>↑</span><span>Parent folder</span></button>`);
    entries.push(...payload.directories.map((directory) => `<button class="folder-entry" data-folder-path="${escapeHtml(directory.path)}" type="button"><span>□</span><span>${escapeHtml(directory.name)}</span></button>`));
    folderList.innerHTML = entries.join('') || '<div class="folder-empty">No subfolders found.</div>';
    folderList.querySelectorAll('[data-folder-path]').forEach((entry) => entry.addEventListener('click', () => browseFolder(entry.dataset.folderPath)));
  } catch (error) {
    folderPickerMessage.textContent = error.message;
    folderPathInput.value = path || '';
  }
}

function openFolderPicker(targetId) {
  folderPickerTarget = document.querySelector(`#${targetId}`);
  selectedFolderPath = null;
  folderPicker.classList.remove('hidden');
  browseFolder(folderPickerTarget.value.trim());
}

document.querySelectorAll('[data-folder-target]').forEach((button) => button.addEventListener('click', () => openFolderPicker(button.dataset.folderTarget)));
document.querySelector('#closeFolderPicker').addEventListener('click', closeFolderPicker);
document.querySelector('#cancelFolderButton').addEventListener('click', closeFolderPicker);
document.querySelector('[data-close-folder-picker]').addEventListener('click', closeFolderPicker);
folderGoButton.addEventListener('click', () => browseFolder(folderPathInput.value.trim()));
folderPathInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') browseFolder(folderPathInput.value.trim());
});
selectFolderButton.addEventListener('click', () => {
  if (!folderPickerTarget || !selectedFolderPath) return;
  folderPickerTarget.value = selectedFolderPath;
  folderPickerTarget.dispatchEvent(new Event('change', { bubbles: true }));
  closeFolderPicker();
});

function closeHelpModal() {
  helpModal.classList.add('hidden');
  if (helpReturnFocus) helpReturnFocus.focus();
  helpReturnFocus = null;
}

function openHelpModal(button) {
  helpReturnFocus = button;
  helpModalTitle.textContent = button.dataset.helpTitle;
  helpModalText.textContent = button.dataset.helpText;
  helpModal.classList.remove('hidden');
  dismissHelpModalButton.focus();
}

document.querySelectorAll('[data-help-title]').forEach((button) => button.addEventListener('click', () => openHelpModal(button)));
closeHelpModalButton.addEventListener('click', closeHelpModal);
dismissHelpModalButton.addEventListener('click', closeHelpModal);
document.querySelector('[data-close-help-modal]').addEventListener('click', closeHelpModal);

function closeCleanupModal() {
  cleanupModal.classList.add('hidden');
  if (cleanupReturnFocus) cleanupReturnFocus.focus();
  cleanupReturnFocus = null;
}

function openCleanupModal(status) {
  const count = status.item_count || 0;
  const itemLabel = count === 1 ? '1 existing item' : `${count} existing items`;
  cleanupFolderPath.textContent = status.path;
  cleanupFolderSummary.textContent = `${itemLabel} will be replaced before training starts.`;
  cleanupReturnFocus = runButton;
  cleanupModal.classList.remove('hidden');
  confirmCleanupButton.focus();
}

async function inspectResultsFolder(path) {
  const response = await fetch(`/api/folder-status?path=${encodeURIComponent(path)}`, { cache: 'no-store' });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || 'Could not inspect the results folder');
  return payload;
}

document.querySelector('[data-close-cleanup-modal]').addEventListener('click', closeCleanupModal);
closeCleanupModalButton.addEventListener('click', closeCleanupModal);
cancelCleanupButton.addEventListener('click', closeCleanupModal);
document.addEventListener('keydown', (event) => {
  if (event.key !== 'Escape') return;
  if (!cleanupModal.classList.contains('hidden')) closeCleanupModal();
  else if (!helpModal.classList.contains('hidden')) closeHelpModal();
  else if (!folderPicker.classList.contains('hidden')) closeFolderPicker();
});

async function uploadSelectedFiles(input, pathInput, statusElement) {
  const files = [...input.files];
  if (!files.length) return;
  pendingUploads += 1;
  statusElement.textContent = `Uploading 0 of ${files.length} files…`;
  let group = '';
  try {
    for (let index = 0; index < files.length; index += 1) {
      const file = files[index];
      statusElement.textContent = `Uploading ${index + 1} of ${files.length}: ${file.name}`;
      const response = await fetch('/api/upload', {
        method: 'POST',
        headers: { 'Content-Type': file.type || 'application/octet-stream', 'X-Upload-Name': file.name, 'X-Upload-Group': group },
        body: file,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'Upload failed');
      group = payload.group;
      pathInput.value = payload.directory;
    }
    statusElement.textContent = `${files.length} file${files.length === 1 ? '' : 's'} ready in the local upload folder.`;
  } catch (error) {
    statusElement.textContent = `Could not upload files: ${error.message}`;
  } finally {
    pendingUploads -= 1;
    input.value = '';
  }
}

async function uploadFeatureFile(input, pathInput, statusElement) {
  const file = input.files[0];
  if (!file) return;
  pendingUploads += 1;
  statusElement.textContent = `Selecting ${file.name}…`;
  try {
    const response = await fetch('/api/upload-feature', {
      method: 'POST',
      headers: { 'Content-Type': file.type || 'application/octet-stream', 'X-Upload-Name': file.name },
      body: file,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'File selection failed');
    pathInput.value = payload.path;
    statusElement.textContent = `${file.name} selected.`;
  } catch (error) {
    statusElement.textContent = `Could not select file: ${error.message}`;
  } finally {
    pendingUploads -= 1;
    input.value = '';
  }
}

function configFromForm() {
  return {
    workflow: document.querySelector('input[name="workflow"]:checked').value,
    training_images: value('trainingImages'),
    results_dir: value('resultsDir'),
    inference_images: value('inferenceImages'),
    inference_output: value('inferenceOutput'),
    workers: Number(value('workers')),
    top_features: Number(value('topFeatures')),
    correlation_threshold: Number(value('correlationThreshold')),
    voxel_size_x: Number(value('voxelSizeX')),
    voxel_size_y: Number(value('voxelSizeY')),
    voxel_size_z: Number(value('voxelSizeZ')),
    replication_unit: document.querySelector('#replicationUnit').value,
    feature_file: value('featureFile'),
    all_learners: document.querySelector('input[name="learner_mode"]:checked').value === 'all',
    learner: document.querySelector('#learner').value,
    confirm_cleanup: document.querySelector('input[name="confirm_cleanup"]').checked,
    cpu_limit: hardwareDefaults?.cpu_limit ?? null,
    memory_limit: hardwareDefaults?.memory_limit ?? null,
  };
}

function renderValidation(report) {
  if (!report) {
    validationPanel.classList.remove('is-invalid', 'is-valid');
    validationMessage.textContent = 'Input validation runs before processing.';
    validationDetails.innerHTML = '';
    return;
  }
  const images = report.images || {};
  const features = report.features || {};
  validationPanel.classList.toggle('is-invalid', report.ok === false);
  validationPanel.classList.toggle('is-valid', report.ok === true);
  validationMessage.textContent = images.message || (report.ok ? 'Preflight passed.' : 'Preflight needs attention.');
  const labelRows = Object.entries(images.images_per_label || {}).map(([label, count]) => `<tr><th scope="row">${escapeHtml(label)}</th><td>${count}</td></tr>`).join('');
  const dateRows = Object.entries(images.images_per_label_per_date || {}).flatMap(([date, counts]) => Object.entries(counts).map(([label, count]) => `<tr><th scope="row">${escapeHtml(date)}</th><td>${escapeHtml(label)}</td><td>${count}</td></tr>`)).join('');
  const filenames = (images.image_filenames || []).map((filename) => `<li>${escapeHtml(filename)}</li>`).join('');
  const filenameList = !images.labelled && filenames ? `<details class="validation-secondary"><summary>Filenames found (${images.total_images})</summary><ul class="validation-filenames">${filenames}</ul></details>` : '';
  const featureWarnings = (features.warnings || []).map((warning) => `<li>${escapeHtml(warning)}</li>`).join('');
  const warningList = featureWarnings ? `<p><strong>Handled automatically:</strong></p><ul class="validation-warnings">${featureWarnings}</ul>` : '';
  const featureSummary = features.features_read === undefined ? '' : `<p><strong>Features:</strong> ${features.features_read} read (${features.microics_features} MicroICS, ${features.external_features} external); ${features.empty_cells || 0} NaN/empty cells.</p><p><strong>Unparsed:</strong> ${features.unparsed_feature_names?.length ? escapeHtml(features.unparsed_feature_names.join(', ')) : 'none'}</p>${warningList}`;
  const primaryTable = labelRows ? `<section class="validation-section"><strong>Images per label</strong><table class="validation-table"><thead><tr><th>Label (class)</th><th>Images</th></tr></thead><tbody>${labelRows}</tbody></table></section>` : '';
  const dateTable = dateRows ? `<details class="validation-secondary"><summary>Images per label per date</summary><table class="validation-table"><thead><tr><th>Date</th><th>Label</th><th>Images</th></tr></thead><tbody>${dateRows}</tbody></table></details>` : '';
  const imageSummary = images.labelled === false ? `<p><strong>Images:</strong> ${images.total_images || 0} files found.</p>` : `<p><strong>Images:</strong> ${images.valid_images || 0}/${images.total_images || 0} filenames valid.</p>`;
  validationDetails.innerHTML = `${imageSummary}${filenameList}${primaryTable}${dateTable}${featureSummary}`;
}

async function preflight(config) {
  const payload = await post('/api/preflight', config);
  renderValidation(payload.report);
  if (!payload.report.ok) throw new Error('Preflight validation failed. Review the report before running.');
  return payload;
}

document.querySelector('#trainingFilePicker').addEventListener('change', (event) => uploadSelectedFiles(event.target, document.querySelector('#trainingImages'), document.querySelector('#trainingFileStatus')));
document.querySelector('#inferenceFilePicker').addEventListener('change', (event) => uploadSelectedFiles(event.target, document.querySelector('#inferenceImages'), document.querySelector('#inferenceFileStatus')));
document.querySelector('#featureFilePicker').addEventListener('change', (event) => uploadFeatureFile(event.target, document.querySelector('#featureFile'), document.querySelector('#featureFileStatus')));

hardwareDefaultsButton.addEventListener('click', async () => {
  hardwareDefaultsButton.disabled = true;
  hardwareDefaultsMessage.textContent = 'Detecting available CPU and RAM…';
  try {
    const response = await fetch('/api/hardware-defaults', { cache: 'no-store' });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Could not detect hardware');
    hardwareDefaults = payload;
    document.querySelector('#workers').value = payload.workers;
    const memoryText = payload.memory_limit_mib ? `${payload.memory_limit_mib} MiB RAM` : 'a conservative RAM limit';
    hardwareDefaultsMessage.textContent = `Using ${payload.workers} workers, ${payload.cpu_limit} CPU limit, and ${memoryText}.`;
  } catch (error) {
    hardwareDefaultsMessage.textContent = error.message;
  } finally {
    hardwareDefaultsButton.disabled = false;
  }
});

async function post(path, body = {}) {
  let response;
  try {
    response = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  } catch (error) {
    throw new Error('Could not reach the local GUI server. Reopen the Windows launcher and check microics-gui.log.');
  }
  let payload;
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error('The local GUI server returned an incomplete response. Check microics-gui.log for the underlying error.');
  }
  if (!response.ok) throw new Error(payload.error || 'Request failed');
  return payload;
}

async function startConfiguredPipeline(config) {
  await preflight(config);
  await post('/api/run', config);
  await poll();
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(seconds) {
  const rounded = Math.max(1, Math.round(seconds));
  const minutes = Math.floor(rounded / 60);
  const remainingSeconds = rounded % 60;
  if (!minutes) return `${remainingSeconds}s`;
  if (minutes < 60) return `${minutes}m ${remainingSeconds}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

function renderEta(status, progress) {
  if (status !== 'running') {
    runStartedAt = null;
    etaLabel.textContent = status === 'complete' ? 'Complete' : 'ETA —';
    etaLabel.classList.toggle('is-ready', status === 'complete');
    return;
  }
  if (previousRunStatus !== 'running' || !runStartedAt) runStartedAt = performance.now();
  const percent = Math.max(0, Math.min(100, Number(progress?.percent) || 0));
  if (percent < 1) {
    etaLabel.textContent = 'ETA calculating…';
    etaLabel.classList.remove('is-ready');
    return;
  }
  const elapsedSeconds = (performance.now() - runStartedAt) / 1000;
  const remainingSeconds = elapsedSeconds * ((100 - percent) / percent);
  etaLabel.textContent = `ETA ${formatDuration(remainingSeconds)}`;
  etaLabel.classList.remove('is-ready');
}

function renderProgress(progress, steps, status) {
  const current = (steps || []).find((step) => step.status === 'running');
  const total = progress?.total || steps?.length || 0;
  const completed = progress?.completed || 0;
  const percent = Math.max(0, Math.min(100, Number(progress?.percent) || 0));
  progressFill.style.width = `${percent}%`;
  progressFill.classList.toggle('is-running', status === 'running');
  progressBar.classList.toggle('is-running', status === 'running');
  progressBar.setAttribute('aria-valuenow', percent);
  progressLabel.textContent = progress?.label || (status === 'complete' ? 'Complete' : 'Ready to run');
  progressValue.textContent = `${percent}%`;
  progressSteps.textContent = total ? `${completed} of ${total} stages complete` : 'No stages started';
  progressDetail.textContent = current ? current.detail || progress?.detail || 'Working…' : progress?.detail || (status === 'complete' ? 'All requested work finished.' : 'The active stage will appear here.');
}

function renderSteps(steps) {
  if (!steps || !steps.length) {
    pipelineTrack.innerHTML = '<div class="empty-state"><span class="empty-mark">◌</span><p>Your pipeline is ready.</p><small>Start a run to see each Docker step light up here.</small></div>';
    return;
  }
  pipelineTrack.innerHTML = steps.map((step) => {
    const statusClass = `is-${step.status}`;
    const icon = step.status === 'complete' ? '✓' : step.status === 'failed' ? '!' : step.status === 'cancelled' ? '−' : (stepIcons[step.id] || '·');
    return `<article class="pipeline-step ${statusClass}"><div class="step-mark">${icon}</div><h3>${step.label}</h3><p>${step.detail || ''}</p></article>`;
  }).join('');
}

function escapeHtml(text) {
  return String(text).replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
}

function renderFiles(artifacts) {
  const files = artifacts || [];
  fileCount.textContent = `${files.length}${files.length >= 450 ? '+' : ''} file${files.length === 1 ? '' : 's'}`;
  if (!files.length) {
    fileList.innerHTML = '<div class="empty-state compact"><span class="empty-mark">⌁</span><p>No output yet.</p><small>Generated files will appear here as the pipeline completes.</small></div>';
    return;
  }
  fileList.innerHTML = files.map((file) => {
    const icon = file.kind === 'image' ? '▧' : file.kind === 'html' ? '↗' : file.kind === 'text' ? '≡' : '·';
    return `<div class="file-row" data-root="${escapeHtml(file.root)}" data-path="${escapeHtml(file.path)}"><span class="file-icon">${icon}</span><span class="file-name" title="${escapeHtml(file.path)}">${escapeHtml(file.name)}<small class="file-meta">${escapeHtml(file.path)} · ${formatBytes(file.size)}</small></span><span class="file-root">${escapeHtml(file.root)}</span></div>`;
  }).join('');
  fileList.querySelectorAll('.file-row').forEach((row) => row.addEventListener('click', () => inspectFile(row)));
}

async function inspectFile(row) {
  fileList.querySelectorAll('.file-row').forEach((item) => item.classList.remove('selected'));
  row.classList.add('selected');
  const root = row.dataset.root;
  const path = row.dataset.path;
  const file = (latestState.artifacts || []).find((item) => item.root === root && item.path === path);
  if (!file) return;
  const encoded = `root=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}`;
  if (file.kind === 'image') {
    preview.innerHTML = `<div class="preview-title">${escapeHtml(path)}</div><img src="/api/artifact?${encoded}" alt="${escapeHtml(file.name)}">`;
  } else if (file.kind === 'html') {
    preview.innerHTML = `<div class="preview-title">${escapeHtml(path)}</div><iframe src="/api/artifact?${encoded}" title="${escapeHtml(file.name)}"></iframe>`;
  } else if (file.kind === 'text') {
    preview.innerHTML = '<div class="preview-placeholder"><p>Loading preview…</p></div>';
    try {
      const response = await fetch(`/api/preview?${encoded}`);
      const payload = await response.json();
      preview.innerHTML = `<div class="preview-title">${escapeHtml(payload.name)}</div><pre>${escapeHtml(payload.content)}</pre>`;
    } catch (error) {
      preview.innerHTML = `<div class="preview-placeholder"><p>Preview unavailable.</p><small>${escapeHtml(error.message)}</small></div>`;
    }
  } else {
    preview.innerHTML = `<div class="preview-placeholder"><span>·</span><p>${escapeHtml(file.name)}</p><small>This file is available in the results folder.</small></div>`;
  }
}

function renderState(nextState) {
  latestState = nextState;
  updateDocumentTitle(nextState.status, nextState.progress);
  renderEta(nextState.status, nextState.progress);
  const configKey = nextState.config ? JSON.stringify(nextState.config) : null;
  if (configKey && configKey !== syncedConfigKey) {
    syncForm(nextState.config);
    syncedConfigKey = configKey;
  }
  connectionText.textContent = nextState.status === 'running' ? 'Processing' : nextState.status === 'complete' ? 'Complete' : nextState.status === 'failed' ? 'Needs attention' : nextState.status === 'cancelled' ? 'Stopped' : 'Ready';
  renderSteps(nextState.steps);
  renderProgress(nextState.progress, nextState.steps, nextState.status);
  renderValidation(nextState.validation);
  statusWarning.classList.toggle('hidden', !['failed', 'cancelled'].includes(nextState.status));
  statusWarning.title = nextState.error || 'The run did not complete. Open the run log for details.';
  renderFiles(nextState.artifacts);
  logOutput.textContent = (nextState.logs || []).join('\n') || 'No run started.';
  logOutput.scrollTop = logOutput.scrollHeight;
  logState.textContent = nextState.status[0].toUpperCase() + nextState.status.slice(1);
  const running = nextState.status === 'running';
  runButton.disabled = running;
  hardwareDefaultsButton.disabled = running;
  runButton.classList.toggle('hidden', running);
  stopButton.classList.toggle('hidden', !running);
  resetButton.classList.toggle('hidden', nextState.status === 'idle');
  if (nextState.status === 'idle' && previousRunStatus && previousRunStatus !== 'idle') {
    preview.innerHTML = '<div class="preview-placeholder"><span>＋</span><p>Select a file to inspect it.</p><small>Tables, plots, and HTML visualizations open here.</small></div>';
  }
  formMessage.classList.toggle('success', !running && nextState.status === 'complete');
  if (!running && nextState.status === 'complete') formMessage.textContent = 'Pipeline complete. Select an output file to inspect it.';
  if (nextState.status === 'failed' && nextState.error) formMessage.textContent = nextState.error;
  previousRunStatus = nextState.status;
}

async function poll(refresh = false) {
  try {
    const response = await fetch(`/api/state${refresh ? '?refresh=1' : ''}`, { cache: 'no-store' });
    renderState(await response.json());
  } catch (error) {
    connectionText.textContent = 'Offline';
    document.title = 'MicroICS · Offline';
    if (latestState?.status === 'running') formMessage.textContent = 'Lost connection to the local GUI server. Reopen the Windows launcher and check microics-gui.log.';
  }
  clearTimeout(pollTimer);
  pollTimer = setTimeout(() => poll(false), 900);
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  formMessage.textContent = '';
  formMessage.classList.remove('success');
  try {
    if (pendingUploads) throw new Error('Please wait for selected files to finish uploading.');
    const config = configFromForm();
    const skipsGeneration = config.workflow === 'full' && Boolean(config.feature_file);
    const needsCleanupCheck = ['full', 'features_labelled', 'features_unlabelled'].includes(config.workflow) && !skipsGeneration && config.results_dir && !config.confirm_cleanup;
    if (needsCleanupCheck) {
      const status = await inspectResultsFolder(config.results_dir);
      if (status.item_count > 0) {
        openCleanupModal(status);
        return;
      }
    }
    await startConfiguredPipeline(config);
  } catch (error) {
    formMessage.textContent = error.message;
  }
});

demoButton.addEventListener('click', () => {
  const images = demoPaths?.images || document.querySelector('#trainingImages').value;
  const results = demoPaths?.results || document.querySelector('#resultsDir').value;
  const inferenceOutput = demoPaths?.inference_output || `${results}/inference`;
  document.querySelector('#trainingImages').value = images;
  document.querySelector('#inferenceImages').value = images;
  document.querySelector('#resultsDir').value = results;
  document.querySelector('#inferenceOutput').value = inferenceOutput;
  document.querySelector('input[name="workflow"][value="train"]').checked = true;
  setWorkflow('train');
  document.querySelector('input[name="confirm_cleanup"]').checked = true;
  formMessage.textContent = 'Sample configuration loaded. Press Run pipeline to verify the installation.';
});

openOutputButton.addEventListener('click', async () => {
  try {
    const folder = document.querySelector('input[name="workflow"]:checked').value === 'inference' ? value('inferenceOutput') : value('resultsDir');
    await post('/api/open-folder', { path: folder });
  } catch (error) {
    formMessage.textContent = error.message;
  }
});

confirmCleanupButton.addEventListener('click', async () => {
  confirmCleanupButton.disabled = true;
  try {
    document.querySelector('input[name="confirm_cleanup"]').checked = true;
    closeCleanupModal();
    await startConfiguredPipeline(configFromForm());
  } catch (error) {
    formMessage.textContent = error.message;
  } finally {
    confirmCleanupButton.disabled = false;
  }
});

stopButton.addEventListener('click', async () => {
  stopButton.disabled = true;
  try { await post('/api/stop'); } catch (error) { formMessage.textContent = error.message; }
  stopButton.disabled = false;
});

resetButton.addEventListener('click', async () => {
  resetButton.disabled = true;
  try {
    const resetState = await post('/api/reset');
    renderState(resetState);
    formMessage.classList.remove('success');
    formMessage.textContent = 'Run cleared. MicroICS is ready to start again.';
  } catch (error) {
    formMessage.textContent = error.message;
  } finally {
    resetButton.disabled = false;
  }
});

document.querySelector('#refreshButton').addEventListener('click', () => poll(true));
function syncLearnerMode() {
  const mode = document.querySelector('input[name="learner_mode"]:checked').value;
  document.querySelector('#learner').disabled = mode === 'all';
  document.querySelectorAll('.learner-option').forEach((option) => option.classList.toggle('active', option.dataset.learnerMode === mode));
}
document.querySelectorAll('input[name="learner_mode"]').forEach((input) => input.addEventListener('change', syncLearnerMode));
syncLearnerMode();
setWorkflow('features_labelled');

async function boot() {
  await loadDefaults();
  await poll();
}

boot();
