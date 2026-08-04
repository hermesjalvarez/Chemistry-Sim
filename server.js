const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const express = require('express');
const session = require('express-session');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');

const PORT = process.env.PORT || 3000;
const ADMIN_USERNAME = 'Luke';
const ADMIN_PASSWORD = '3113';

const DATA_DIR = path.join(__dirname, 'data');
const UPLOADS_DIR = path.join(__dirname, 'uploads');
const INTRO_DIR = path.join(UPLOADS_DIR, 'intro');
const VIDEOS_FILE = path.join(DATA_DIR, 'videos.json');
const INTRO_FILE = path.join(DATA_DIR, 'intro.json');
const REVIEWS_FILE = path.join(DATA_DIR, 'reviews.json');

for (const dir of [DATA_DIR, UPLOADS_DIR, INTRO_DIR]) {
  fs.mkdirSync(dir, { recursive: true });
}

function readJSON(file, fallback) {
  try {
    const raw = fs.readFileSync(file, 'utf8').trim();
    if (!raw) return fallback;
    return JSON.parse(raw);
  } catch (err) {
    if (err.code === 'ENOENT') return fallback;
    throw err;
  }
}

function writeJSON(file, data) {
  fs.writeFileSync(file, JSON.stringify(data, null, 2));
}

const ALLOWED_VIDEO_TYPES = new Set([
  'video/mp4',
  'video/webm',
  'video/ogg',
  'video/quicktime',
  'video/x-matroska',
]);

function videoFileFilter(req, file, cb) {
  if (ALLOWED_VIDEO_TYPES.has(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error('Only video files (mp4, webm, ogg, mov, mkv) are allowed.'));
  }
}

const MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024; // 2GB

const videoStorage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOADS_DIR),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    cb(null, `${uuidv4()}${ext}`);
  },
});
const uploadVideo = multer({
  storage: videoStorage,
  fileFilter: videoFileFilter,
  limits: { fileSize: MAX_UPLOAD_BYTES },
});

const introStorage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, INTRO_DIR),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    cb(null, `${uuidv4()}${ext}`);
  },
});
const uploadIntro = multer({
  storage: introStorage,
  fileFilter: videoFileFilter,
  limits: { fileSize: MAX_UPLOAD_BYTES },
});

const app = express();
app.set('trust proxy', 1);
app.use(express.json());

app.use(
  session({
    name: 'sh.sid',
    secret: crypto.randomBytes(32).toString('hex'),
    resave: false,
    saveUninitialized: false,
    cookie: {
      httpOnly: true,
      sameSite: 'lax',
      secure: process.env.NODE_ENV === 'production',
      maxAge: 1000 * 60 * 60 * 12, // 12 hours
    },
  })
);

function requireAdmin(req, res, next) {
  if (req.session && req.session.isAdmin) return next();
  return res.status(401).json({ error: 'Admin login required.' });
}

function escapeText(value, maxLen) {
  return String(value || '').trim().slice(0, maxLen);
}

// ---------- Auth ----------
app.post('/api/login', (req, res) => {
  const { username, password } = req.body || {};
  if (username === ADMIN_USERNAME && password === ADMIN_PASSWORD) {
    req.session.isAdmin = true;
    req.session.username = ADMIN_USERNAME;
    return res.json({ loggedIn: true, username: ADMIN_USERNAME });
  }
  return res.status(401).json({ error: 'Invalid username or password.' });
});

app.post('/api/logout', (req, res) => {
  req.session.destroy(() => res.json({ loggedIn: false }));
});

app.get('/api/session', (req, res) => {
  if (req.session && req.session.isAdmin) {
    return res.json({ loggedIn: true, username: req.session.username });
  }
  return res.json({ loggedIn: false });
});

// ---------- Videos (library, shown at the bottom of the page) ----------
app.get('/api/videos', (req, res) => {
  const videos = readJSON(VIDEOS_FILE, []);
  res.json(videos.slice().sort((a, b) => b.uploadedAt - a.uploadedAt));
});

app.post('/api/videos', requireAdmin, uploadVideo.single('video'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'A video file is required.' });
  const title = escapeText(req.body.title, 150) || 'Untitled video';
  const description = escapeText(req.body.description, 1000);
  const category = escapeText(req.body.category, 50) || 'General';

  const videos = readJSON(VIDEOS_FILE, []);
  const video = {
    id: uuidv4(),
    title,
    description,
    category,
    filename: req.file.filename,
    uploadedAt: Date.now(),
  };
  videos.push(video);
  writeJSON(VIDEOS_FILE, videos);
  res.status(201).json(video);
});

app.delete('/api/videos/:id', requireAdmin, (req, res) => {
  const videos = readJSON(VIDEOS_FILE, []);
  const video = videos.find((v) => v.id === req.params.id);
  if (!video) return res.status(404).json({ error: 'Video not found.' });

  const remaining = videos.filter((v) => v.id !== req.params.id);
  writeJSON(VIDEOS_FILE, remaining);

  const filePath = path.join(UPLOADS_DIR, video.filename);
  fs.unlink(filePath, () => {});

  res.json({ ok: true });
});

// ---------- Intro video (shown above the library, below the description) ----------
app.get('/api/intro', (req, res) => {
  const intro = readJSON(INTRO_FILE, null);
  res.json(intro);
});

app.post('/api/intro', requireAdmin, uploadIntro.single('video'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'A video file is required.' });
  const title = escapeText(req.body.title, 150) || 'Welcome to Science Hound';

  const previous = readJSON(INTRO_FILE, null);
  const intro = { title, filename: req.file.filename, uploadedAt: Date.now() };
  writeJSON(INTRO_FILE, intro);

  if (previous && previous.filename) {
    fs.unlink(path.join(INTRO_DIR, previous.filename), () => {});
  }

  res.status(201).json(intro);
});

app.delete('/api/intro', requireAdmin, (req, res) => {
  const previous = readJSON(INTRO_FILE, null);
  writeJSON(INTRO_FILE, null);
  if (previous && previous.filename) {
    fs.unlink(path.join(INTRO_DIR, previous.filename), () => {});
  }
  res.json({ ok: true });
});

// ---------- Reviews (public to post, admin can moderate/delete) ----------
app.get('/api/reviews', (req, res) => {
  const reviews = readJSON(REVIEWS_FILE, []);
  res.json(reviews.slice().sort((a, b) => b.postedAt - a.postedAt));
});

app.post('/api/reviews', (req, res) => {
  const name = escapeText(req.body && req.body.name, 80) || 'Anonymous';
  const comment = escapeText(req.body && req.body.comment, 500);
  if (!comment) return res.status(400).json({ error: 'A comment is required.' });

  const reviews = readJSON(REVIEWS_FILE, []);
  const review = { id: uuidv4(), name, comment, postedAt: Date.now() };
  reviews.push(review);
  writeJSON(REVIEWS_FILE, reviews);
  res.status(201).json(review);
});

app.delete('/api/reviews/:id', requireAdmin, (req, res) => {
  const reviews = readJSON(REVIEWS_FILE, []);
  const remaining = reviews.filter((r) => r.id !== req.params.id);
  writeJSON(REVIEWS_FILE, remaining);
  res.json({ ok: true });
});

// ---------- Static files ----------
app.use('/uploads/intro', express.static(INTRO_DIR));
app.use('/uploads', express.static(UPLOADS_DIR));
app.use(express.static(path.join(__dirname, 'public')));

// ---------- Error handling (e.g. multer file-type / size errors) ----------
app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError || err) {
    return res.status(400).json({ error: err.message || 'Upload failed.' });
  }
  next();
});

app.listen(PORT, () => {
  console.log(`Science Hound is running at http://localhost:${PORT}`);
});
