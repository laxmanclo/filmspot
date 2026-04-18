const state = {
  movies: [],
  selectedMovie: null,
  selectedScenes: [],
  ingestPollTimer: null,
};

const homeView = document.querySelector('#home-view');
const playerView = document.querySelector('#player-view');
const moviesGrid = document.querySelector('#movies-grid');
const homeEmpty = document.querySelector('#home-empty');
const movieCardTemplate = document.querySelector('#movie-card-template');

const addClipBtn = document.querySelector('#add-clip-btn');
const addClipModal = document.querySelector('#add-clip-modal');
const ingestForm = document.querySelector('#ingest-form');
const cancelIngestBtn = document.querySelector('#cancel-ingest');
const ingestProgressWrap = document.querySelector('#ingest-progress-wrap');
const ingestProgress = document.querySelector('#ingest-progress');
const ingestStatus = document.querySelector('#ingest-status');

const backBtn = document.querySelector('#back-btn');
const playerTitle = document.querySelector('#player-title');
const playerSubtitle = document.querySelector('#player-subtitle');
const moviePlayer = document.querySelector('#movie-player');
const chatToggle = document.querySelector('#chat-toggle');
const chatPanel = document.querySelector('#chat-panel');
const chatLog = document.querySelector('#chat-log');
const chatEntryTemplate = document.querySelector('#chat-entry-template');
const queryForm = document.querySelector('#query-form');
const queryText = document.querySelector('#query-text');
const queryImage = document.querySelector('#query-image');

const timelineTrack = document.querySelector('#timeline-track');
const timelineList = document.querySelector('#timeline-list');

function fmtTime(sec) {
  const total = Math.max(0, Math.floor(Number(sec) || 0));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function showView(view) {
  homeView.classList.toggle('active', view === 'home');
  playerView.classList.toggle('active', view === 'player');
}

async function fetchJson(url, options = {}) {
  const res = await fetch(url, options);
  const isJson = (res.headers.get('content-type') || '').includes('application/json');
  const body = isJson ? await res.json() : null;
  if (!res.ok) {
    const detail = body?.detail || `Request failed (${res.status})`;
    throw new Error(detail);
  }
  return body;
}

function renderMovies() {
  moviesGrid.innerHTML = '';
  homeEmpty.classList.toggle('hidden', state.movies.length > 0);

  state.movies.forEach((movie) => {
    const node = movieCardTemplate.content.cloneNode(true);
    const card = node.querySelector('.movie-card');
    node.querySelector('.movie-name').textContent = movie.movie_id;
    node.querySelector('.movie-meta').textContent = `${fmtTime(movie.duration_sec || 0)} • ${movie.node_count || 0} nodes`;
    node.querySelector('.movie-history').textContent = `${movie.history_count || 0} previous queries`;

    card.addEventListener('click', () => openMovie(movie));
    moviesGrid.appendChild(node);
  });
}

async function loadMovies() {
  const data = await fetchJson('/movies');
  state.movies = data.movies || [];
  renderMovies();
}

function clearTimeline() {
  timelineTrack.innerHTML = '';
  timelineList.innerHTML = '';
}

function drawTimelineMarkers(scenes) {
  clearTimeline();
  const duration = Number(moviePlayer.duration || state.selectedMovie?.duration_sec || 0);
  if (!duration || !Array.isArray(scenes)) {
    return;
  }

  scenes.forEach((scene, idx) => {
    const start = Math.max(0, Number(scene.start_t || 0));
    const end = Math.max(start, Number(scene.end_t || start));
    const left = Math.min(99.5, (start / duration) * 100);
    const width = Math.max(0.8, ((end - start) / duration) * 100);

    const marker = document.createElement('button');
    marker.className = 'timeline-marker';
    marker.style.left = `${left}%`;
    marker.style.width = `${Math.min(100 - left, width)}%`;
    marker.title = `Scene ${idx + 1} (${fmtTime(start)} - ${fmtTime(end)})`;
    marker.addEventListener('click', () => {
      moviePlayer.currentTime = start;
      moviePlayer.play().catch(() => {});
    });
    timelineTrack.appendChild(marker);

    const chip = document.createElement('button');
    chip.className = 'time-chip';
    chip.textContent = `${fmtTime(start)} → ${fmtTime(end)}`;
    chip.addEventListener('click', () => {
      moviePlayer.currentTime = start;
      moviePlayer.play().catch(() => {});
    });
    timelineList.appendChild(chip);
  });
}

function renderChatEntry(entry) {
  const node = chatEntryTemplate.content.cloneNode(true);
  const queryTextEl = node.querySelector('.chat-query');
  const scenesList = node.querySelector('.chat-scenes');

  const q = entry.query_text || (entry.query_image ? '[Image query]' : '[Unknown query]');
  queryTextEl.textContent = q;

  (entry.scenes || []).forEach((scene) => {
    const li = document.createElement('li');
    li.textContent = `${fmtTime(scene.start_t)} → ${fmtTime(scene.end_t)}${scene.conflict ? ' ⚠ conflict' : ''}`;
    li.title = scene.caption || '';
    li.addEventListener('click', () => {
      moviePlayer.currentTime = Number(scene.start_t || 0);
      moviePlayer.play().catch(() => {});
    });
    scenesList.appendChild(li);
  });

  chatLog.appendChild(node);
  chatLog.scrollTop = chatLog.scrollHeight;
}

async function loadHistory(movieId) {
  chatLog.innerHTML = '';
  const data = await fetchJson(`/movies/${encodeURIComponent(movieId)}/history?limit=200`);
  (data.entries || []).forEach(renderChatEntry);
}

async function openMovie(movie) {
  state.selectedMovie = movie;
  state.selectedScenes = [];
  playerTitle.textContent = movie.movie_id;
  playerSubtitle.textContent = `${fmtTime(movie.duration_sec || 0)} • ${movie.node_count || 0} indexed nodes`;
  moviePlayer.src = `/movies/${encodeURIComponent(movie.movie_id)}/stream`;
  moviePlayer.load();
  clearTimeline();
  await loadHistory(movie.movie_id);
  showView('player');
}

function startIngestPolling(jobId) {
  if (state.ingestPollTimer) {
    clearInterval(state.ingestPollTimer);
  }

  state.ingestPollTimer = setInterval(async () => {
    try {
      const payload = await fetchJson(`/ingest/jobs/${jobId}`);
      ingestProgressWrap.classList.remove('hidden');
      ingestProgress.value = Number(payload.progress || 0);
      ingestStatus.textContent = payload.error ? `${payload.message}: ${payload.error}` : payload.message;
      ingestStatus.classList.toggle('error', payload.status === 'failed');

      if (payload.status === 'completed' || payload.status === 'failed') {
        clearInterval(state.ingestPollTimer);
        state.ingestPollTimer = null;
        if (payload.status === 'completed') {
          await loadMovies();
          setTimeout(() => addClipModal.close(), 600);
        }
      }
    } catch (err) {
      clearInterval(state.ingestPollTimer);
      state.ingestPollTimer = null;
      ingestStatus.textContent = err.message;
      ingestStatus.classList.add('error');
    }
  }, 1500);
}

addClipBtn.addEventListener('click', () => {
  ingestForm.reset();
  ingestProgress.value = 0;
  ingestStatus.textContent = 'Queued…';
  ingestStatus.classList.remove('error');
  ingestProgressWrap.classList.add('hidden');
  addClipModal.showModal();
});

cancelIngestBtn.addEventListener('click', () => addClipModal.close());

ingestForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const movieName = document.querySelector('#movie-name').value.trim();
  const movieFile = document.querySelector('#movie-file').files[0];
  const fps = document.querySelector('#fps').value;
  const semanticK = document.querySelector('#semantic-k').value;

  if (!movieName || !movieFile) {
    ingestStatus.textContent = 'Clip name and movie file are required.';
    ingestStatus.classList.add('error');
    ingestProgressWrap.classList.remove('hidden');
    return;
  }

  const fd = new FormData();
  fd.append('movie_name', movieName);
  fd.append('clip', movieFile);
  fd.append('fps', fps);
  fd.append('semantic_k', semanticK);

  ingestProgressWrap.classList.remove('hidden');
  ingestProgress.value = 5;
  ingestStatus.textContent = 'Uploading clip…';
  ingestStatus.classList.remove('error');

  try {
    const payload = await fetchJson('/ingest/upload', {
      method: 'POST',
      body: fd,
    });
    startIngestPolling(payload.job_id);
  } catch (err) {
    ingestStatus.textContent = err.message;
    ingestStatus.classList.add('error');
  }
});

backBtn.addEventListener('click', () => {
  moviePlayer.pause();
  showView('home');
});

chatToggle.addEventListener('click', () => {
  chatPanel.classList.toggle('hidden');
});

queryForm.addEventListener('submit', async (event) => {
  event.preventDefault();

  if (!state.selectedMovie) {
    return;
  }

  const text = queryText.value.trim();
  const imageFile = queryImage.files[0] || null;

  if (!text && !imageFile) {
    return;
  }

  const fd = new FormData();
  fd.append('movie_id', state.selectedMovie.movie_id);
  if (text) {
    fd.append('text', text);
  }
  if (imageFile) {
    fd.append('image', imageFile);
  }

  try {
    const result = await fetchJson('/query', { method: 'POST', body: fd });
    state.selectedScenes = result.scenes || [];
    drawTimelineMarkers(state.selectedScenes);

    renderChatEntry({
      query_text: text || null,
      query_image: imageFile ? imageFile.name : null,
      scenes: state.selectedScenes,
    });

    queryText.value = '';
    queryImage.value = '';
  } catch (err) {
    renderChatEntry({
      query_text: `[Error] ${err.message}`,
      query_image: null,
      scenes: [],
    });
  }
});

moviePlayer.addEventListener('loadedmetadata', () => {
  if (state.selectedScenes.length > 0) {
    drawTimelineMarkers(state.selectedScenes);
  }
});

loadMovies().catch((err) => {
  homeEmpty.classList.remove('hidden');
  homeEmpty.textContent = err.message;
});
