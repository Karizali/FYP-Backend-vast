const B2 = require('backblaze-b2');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');
const logger = require('../utils/logger');

// ─── Configure Backblaze B2 ───────────────────────────────────────────────────
const b2 = new B2({
  applicationKeyId: process.env.B2_KEY_ID,
  applicationKey: process.env.B2_APP_KEY,
});

let _isAuthorized = false;
let _downloadUrl = '';

async function ensureAuthorized() {
  if (_isAuthorized) return;
  try {
    const { data } = await b2.authorize();
    _downloadUrl = data.downloadUrl;
    _isAuthorized = true;
  } catch (err) {
    logger.error(`Failed to authorize B2: ${err.message}`);
    throw err;
  }
}

async function getBucket() {
  await ensureAuthorized();
  return process.env.B2_BUCKET_NAME;
}

// ─── Retry helper with exponential backoff ────────────────────────────────────
async function retryWithBackoff(fn, retries = 4, baseDelayMs = 1000) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      const status = err?.response?.status;
      const isRetryable =
        status === 503 ||
        status === 500 ||
        status === 429 ||
        err?.code === 'ECONNRESET' ||
        err?.code === 'ETIMEDOUT';

      if (!isRetryable || attempt === retries) throw err;

      const delay = baseDelayMs * Math.pow(2, attempt); // 1s, 2s, 4s, 8s
      logger.warn(
        `B2 upload attempt ${attempt + 1} failed (${status || err?.code}), retrying in ${delay}ms...`
      );
      await new Promise(res => setTimeout(res, delay));

      // Re-authorize on retry — B2 may have rotated tokens
      _isAuthorized = false;
      await ensureAuthorized();
    }
  }
}

// ─── Upload concurrency limiter ───────────────────────────────────────────────
// Prevents hammering B2 with hundreds of simultaneous uploads
const MAX_CONCURRENT_UPLOADS = 5;
let _activeUploads = 0;
const _uploadQueue = [];

function acquireUploadSlot() {
  return new Promise(resolve => {
    if (_activeUploads < MAX_CONCURRENT_UPLOADS) {
      _activeUploads++;
      resolve();
    } else {
      _uploadQueue.push(resolve);
    }
  });
}

function releaseUploadSlot() {
  _activeUploads--;
  if (_uploadQueue.length > 0) {
    const next = _uploadQueue.shift();
    _activeUploads++;
    next();
  }
}

// ─── Allowed MIME types ────────────────────────────────────────────────────────
const ALLOWED_IMAGE_TYPES = ['image/jpg', 'image/jpeg', 'image/png', 'image/webp'];
const ALLOWED_VIDEO_TYPES = ['video/mp4', 'video/quicktime', 'video/x-msvideo'];
const ALLOWED_TYPES = [...ALLOWED_IMAGE_TYPES, ...ALLOWED_VIDEO_TYPES];

// ─── File filter ──────────────────────────────────────────────────────────────
const fileFilter = (req, file, cb) => {
  if (ALLOWED_TYPES.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(
      new Error(
        `Invalid file type: ${file.mimetype}. ` +
        `Allowed: JPEG, PNG, WEBP images or MP4, MOV, AVI videos.`
      ),
      false
    );
  }
};

// ─── Custom B2 Storage Engine for Multer ──────────────────────────────────────
class B2Storage {
  constructor(options = {}) {}

  async _handleFile(req, file, cb) {
    // Check if client already disconnected before we even start
    if (req.socket?.destroyed) {
      return cb(new Error('Client disconnected before upload started'));
    }

    await acquireUploadSlot();

    try {
      await ensureAuthorized();

      const isVideo = file.mimetype.startsWith('video/');
      const userId = req.user?.id || 'unknown';
      const jobId = req.jobId;
      const publicId = uuidv4();

      let folder, contentType;
      if (isVideo) {
        folder = `uploads/videos/${userId}/${jobId}`;
        contentType = 'video/mp4';
      } else {
        folder = `uploads/images/${userId}/${jobId}`;
        contentType = 'image/' + file.mimetype.split('/')[1];
      }

      const fileName = `${folder}/${publicId}`;

      // Buffer the stream — must happen outside retry loop (stream is single-read)
      // For large video files consider disk-buffering instead, but for images this is fine
      const chunks = [];
      await new Promise((resolve, reject) => {
        file.stream.on('data', chunk => chunks.push(chunk));
        file.stream.on('end', resolve);
        file.stream.on('error', reject);
      });
      const fileData = Buffer.concat(chunks);

      // Retry getUploadUrl + uploadFile together — fresh URL fetched on each attempt
      const fileInfo = await retryWithBackoff(async () => {
        const { data: uploadData } = await b2.getUploadUrl({ bucketId: process.env.B2_BUCKET_ID });
        const { data } = await b2.uploadFile({
          uploadUrl: uploadData.uploadUrl,
          uploadAuthToken: uploadData.authorizationToken,
          fileName,
          data: fileData,
          contentType,
          info: {
            originalName: file.originalname,
            userId,
            jobId,
          },
        });
        return data;
      });

      const downloadUrl = `${_downloadUrl}/file/${process.env.B2_BUCKET_NAME}/${fileName}`;

      cb(null, {
        fieldname: file.fieldname,
        originalname: file.originalname,
        encoding: file.encoding,
        mimetype: file.mimetype,
        size: fileData.length,
        bucket: process.env.B2_BUCKET_NAME,
        filename: fileInfo.fileId,
        path: downloadUrl,
        folder,
      });
    } catch (err) {
      cb(err);
    } finally {
      releaseUploadSlot();
    }
  }

  _removeFile(req, file, cb) {
    cb(null);
  }
}

const storage = new B2Storage();

// ─── Multer instance ──────────────────────────────────────────────────────────
const upload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 1000 * 1024 * 1024, // 1000 MB per file
    files: 300,
  },
});

// ─── Helper: generate a download URL ───────────────────────────────────────────
async function getSignedUrl(fileName, expiresInSeconds = 86400) {
  try {
    await ensureAuthorized();
    return `${_downloadUrl}/file/${process.env.B2_BUCKET_NAME}/${fileName}`;
  } catch (error) {
    logger.error(`Failed to generate download URL for ${fileName}: ${error.message}`);
    throw error;
  }
}

// ─── Helper: upload a buffer directly (for GLB output from GPU worker) ────────
async function uploadBuffer(buffer, options = {}) {
  try {
    await ensureAuthorized();

    const fileName = options.folder
      ? `${options.folder}/${options.publicId || uuidv4()}`
      : `uploads/${options.publicId || uuidv4()}`;

    const fileInfo = await retryWithBackoff(async () => {
      const { data: uploadData } = await b2.getUploadUrl({ bucketId: process.env.B2_BUCKET_ID });
      const { data } = await b2.uploadFile({
        uploadUrl: uploadData.uploadUrl,
        uploadAuthToken: uploadData.authorizationToken,
        fileName,
        data: buffer,
        contentType: 'application/octet-stream',
        info: options.tags ? { tags: options.tags } : {},
      });
      return data;
    });

    const downloadUrl = `${_downloadUrl}/file/${process.env.B2_BUCKET_NAME}/${fileName}`;

    return {
      fileId: fileInfo.fileId,
      fileName,
      downloadUrl,
    };
  } catch (error) {
    logger.error(`B2 buffer upload failed: ${error.message}`);
    throw error;
  }
}

// ─── Helper: delete a single file ─────────────────────────────────────────────
async function deleteFile(b2FileId) {
  try {
    await ensureAuthorized();
    await b2.deleteFile(b2FileId);
    logger.info(`B2: deleted file ${b2FileId}`);
    return { success: true };
  } catch (error) {
    logger.error(`B2 delete failed for ${b2FileId}: ${error.message}`);
    return { success: false };
  }
}

// ─── Helper: delete all files for a job ──────────────────────────────────────
async function deleteJobFiles(jobId) {
  try {
    await ensureAuthorized();

    let deleted = 0;
    const prefixes = [
      `uploads/images/*/` + jobId,
      `uploads/videos/*/` + jobId,
    ];

    for (const prefix of prefixes) {
      try {
        const files = await b2.listFileNames(process.env.B2_BUCKET_NAME, { prefix });
        for (const file of files) {
          await b2.deleteFile(file.fileId);
          deleted++;
        }
      } catch (err) {
        logger.debug(`No files found for prefix ${prefix}`);
      }
    }

    logger.info(`B2: cleaned up ${deleted} files for job ${jobId}`);
  } catch (error) {
    logger.error(`B2 bulk delete failed for job ${jobId}: ${error.message}`);
  }
}

module.exports = {
  b2,
  ensureAuthorized,
  upload,
  getSignedUrl,
  uploadBuffer,
  deleteFile,
  deleteJobFiles,
};