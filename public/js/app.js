(function () {
  'use strict';

  let isAdmin = false;

  const $ = (id) => document.getElementById(id);

  const loginBtn = $('login-btn');
  const logoutBtn = $('logout-btn');
  const adminStatus = $('admin-status');
  const loginModal = $('login-modal');
  const loginClose = $('login-close');
  const loginForm = $('login-form');
  const loginMsg = $('login-msg');

  const introPlayerWrap = $('intro-player-wrap');
  const introPlayer = $('intro-player');
  const introEmpty = $('intro-empty');
  const introAdminPanel = $('intro-admin-panel');
  const introForm = $('intro-form');
  const introFormMsg = $('intro-form-msg');
  const introDeleteBtn = $('intro-delete-btn');

  const libraryAdminPanel = $('library-admin-panel');
  const videoForm = $('video-form');
  const videoFormMsg = $('video-form-msg');
  const videoGrid = $('video-grid');
  const videoEmpty = $('video-empty');

  const reviewForm = $('review-form');
  const reviewList = $('review-list');

  function setText(el, text) {
    el.textContent = text;
  }

  async function api(path, options) {
    const res = await fetch(path, {
      credentials: 'same-origin',
      headers: options && options.body && !(options.body instanceof FormData)
        ? { 'Content-Type': 'application/json' }
        : undefined,
      ...options,
    });
    let data = null;
    try { data = await res.json(); } catch (e) { /* no body */ }
    if (!res.ok) {
      throw new Error((data && data.error) || 'Something went wrong.');
    }
    return data;
  }

  function updateAdminUI() {
    loginBtn.classList.toggle('hidden', isAdmin);
    logoutBtn.classList.toggle('hidden', !isAdmin);
    adminStatus.classList.toggle('hidden', !isAdmin);
    if (isAdmin) setText(adminStatus, 'Logged in as Luke');
    introAdminPanel.classList.toggle('hidden', !isAdmin);
    libraryAdminPanel.classList.toggle('hidden', !isAdmin);
    renderVideos.lastData && renderVideos(renderVideos.lastData);
    renderReviews.lastData && renderReviews(renderReviews.lastData);
  }

  async function refreshSession() {
    const data = await api('/api/session');
    isAdmin = !!data.loggedIn;
    updateAdminUI();
  }

  // ---------- Login modal ----------
  loginBtn.addEventListener('click', () => {
    loginMsg.textContent = '';
    loginForm.reset();
    loginModal.classList.remove('hidden');
    $('login-username').focus();
  });

  loginClose.addEventListener('click', () => loginModal.classList.add('hidden'));
  loginModal.addEventListener('click', (e) => {
    if (e.target === loginModal) loginModal.classList.add('hidden');
  });

  loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    loginMsg.textContent = '';
    loginMsg.className = 'form-msg';
    const username = $('login-username').value;
    const password = $('login-password').value;
    try {
      await api('/api/login', { method: 'POST', body: JSON.stringify({ username, password }) });
      isAdmin = true;
      updateAdminUI();
      loginModal.classList.add('hidden');
    } catch (err) {
      loginMsg.textContent = err.message;
      loginMsg.className = 'form-msg error';
    }
  });

  logoutBtn.addEventListener('click', async () => {
    await api('/api/logout', { method: 'POST' });
    isAdmin = false;
    updateAdminUI();
  });

  // ---------- Intro video ----------
  async function loadIntro() {
    const intro = await api('/api/intro');
    if (intro && intro.filename) {
      introPlayer.src = `/uploads/intro/${intro.filename}`;
      introPlayer.classList.remove('hidden');
      introEmpty.classList.add('hidden');
      introDeleteBtn.classList.toggle('hidden', !isAdmin);
    } else {
      introPlayer.classList.add('hidden');
      introPlayer.removeAttribute('src');
      introEmpty.classList.remove('hidden');
      introDeleteBtn.classList.add('hidden');
    }
  }

  introForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    introFormMsg.textContent = '';
    introFormMsg.className = 'form-msg';
    const fileInput = $('intro-file');
    if (!fileInput.files[0]) return;
    const formData = new FormData();
    formData.append('video', fileInput.files[0]);
    formData.append('title', $('intro-title').value);
    try {
      await api('/api/intro', { method: 'POST', body: formData });
      introForm.reset();
      introFormMsg.textContent = 'Intro video uploaded!';
      introFormMsg.className = 'form-msg success';
      await loadIntro();
    } catch (err) {
      introFormMsg.textContent = err.message;
      introFormMsg.className = 'form-msg error';
    }
  });

  introDeleteBtn.addEventListener('click', async () => {
    if (!confirm('Remove the introductory video?')) return;
    await api('/api/intro', { method: 'DELETE' });
    await loadIntro();
  });

  // ---------- Video library ----------
  function renderVideos(videos) {
    renderVideos.lastData = videos;
    videoGrid.innerHTML = '';
    videoEmpty.classList.toggle('hidden', videos.length > 0);

    videos.forEach((v) => {
      const card = document.createElement('div');
      card.className = 'video-card';

      const video = document.createElement('video');
      video.controls = true;
      video.src = `/uploads/${v.filename}`;
      card.appendChild(video);

      const body = document.createElement('div');
      body.className = 'video-card-body';

      const category = document.createElement('span');
      category.className = 'video-category';
      category.textContent = v.category;
      body.appendChild(category);

      const title = document.createElement('h3');
      title.textContent = v.title;
      body.appendChild(title);

      if (v.description) {
        const desc = document.createElement('p');
        desc.className = 'video-description';
        desc.textContent = v.description;
        body.appendChild(desc);
      }

      if (isAdmin) {
        const delBtn = document.createElement('button');
        delBtn.className = 'btn btn-danger';
        delBtn.textContent = 'Delete Video';
        delBtn.addEventListener('click', async () => {
          if (!confirm(`Delete "${v.title}"?`)) return;
          await api(`/api/videos/${v.id}`, { method: 'DELETE' });
          await loadVideos();
        });
        body.appendChild(delBtn);
      }

      card.appendChild(body);
      videoGrid.appendChild(card);
    });
  }

  async function loadVideos() {
    const videos = await api('/api/videos');
    renderVideos(videos);
  }

  videoForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    videoFormMsg.textContent = '';
    videoFormMsg.className = 'form-msg';
    const fileInput = $('video-file');
    if (!fileInput.files[0]) return;
    const formData = new FormData();
    formData.append('video', fileInput.files[0]);
    formData.append('title', $('video-title').value);
    formData.append('category', $('video-category').value);
    formData.append('description', $('video-description').value);
    try {
      await api('/api/videos', { method: 'POST', body: formData });
      videoForm.reset();
      videoFormMsg.textContent = 'Video uploaded!';
      videoFormMsg.className = 'form-msg success';
      await loadVideos();
    } catch (err) {
      videoFormMsg.textContent = err.message;
      videoFormMsg.className = 'form-msg error';
    }
  });

  // ---------- Reviews ----------
  function renderReviews(reviews) {
    renderReviews.lastData = reviews;
    reviewList.innerHTML = '';
    reviews.forEach((r) => {
      const item = document.createElement('div');
      item.className = 'review-item';

      if (isAdmin) {
        const delBtn = document.createElement('button');
        delBtn.className = 'review-delete';
        delBtn.textContent = 'Remove';
        delBtn.addEventListener('click', async () => {
          await api(`/api/reviews/${r.id}`, { method: 'DELETE' });
          await loadReviews();
        });
        item.appendChild(delBtn);
      }

      const name = document.createElement('div');
      name.className = 'review-name';
      name.textContent = r.name;
      item.appendChild(name);

      const comment = document.createElement('p');
      comment.className = 'review-comment';
      comment.textContent = r.comment;
      item.appendChild(comment);

      reviewList.appendChild(item);
    });
  }

  async function loadReviews() {
    const reviews = await api('/api/reviews');
    renderReviews(reviews);
  }

  reviewForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = $('review-name').value;
    const comment = $('review-comment').value;
    if (!comment.trim()) return;
    try {
      await api('/api/reviews', { method: 'POST', body: JSON.stringify({ name, comment }) });
      reviewForm.reset();
      await loadReviews();
    } catch (err) {
      alert(err.message);
    }
  });

  // ---------- Init ----------
  (async function init() {
    await refreshSession();
    await Promise.all([loadIntro(), loadVideos(), loadReviews()]);
  })();
})();
