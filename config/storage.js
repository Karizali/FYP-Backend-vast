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
let _downloadUrl = '';   // add this

async function ensureAuthorized() {
  if (_isAuthorized) return;
  try {
    const { data } = await b2.authorize();
    _downloadUrl = data.downloadUrl;   // save it here
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

// ─── Allowed MIME types ────────────────────────────────────────────────────────
const ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
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
  constructor(options = {}) {
    this.bucket = null;
  }

  async _handleFile(req, file, cb) {
    try {
      await ensureAuthorized();

      const isVideo = file.mimetype.startsWith('video/');
      const userId = req.user?.id || 'unknown';
      const jobId = req.jobId;
      const publicId = uuidv4();

      // Determine folder and content type
      let folder, contentType;
      if (isVideo) {
        folder = `uploads/videos/${userId}/${jobId}`;
        contentType = 'video/mp4';
      } else {
        folder = `uploads/images/${userId}/${jobId}`;
        contentType = 'image/' + file.mimetype.split('/')[1];
      }

      const fileName = `${folder}/${publicId}`;

      // Read file buffer from multer stream
      const chunks = [];

      await new Promise((resolve, reject) => {
        file.stream.on('data', chunk => chunks.push(chunk));
        file.stream.on('end', () => resolve());
        file.stream.on('error', reject);
      });

      const fileData = Buffer.concat(chunks);

      // Upload to B2
      const { data: uploadData } = await b2.getUploadUrl({ bucketId: process.env.B2_BUCKET_ID });
      const { data: fileInfo } = await b2.uploadFile({
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

      const downloadUrl = `${_downloadUrl}/file/${process.env.B2_BUCKET_NAME}/${fileName}`;

      // Match Multer/Cloudinary response format
      cb(null, {
        fieldname: file.fieldname,
        originalname: file.originalname,
        encoding: file.encoding,
        mimetype: file.mimetype,
        size: fileData.length,
        bucket: process.env.B2_BUCKET_NAME,
        filename: fileInfo.fileId,  // Store B2 file ID
        path: downloadUrl,      // Download URL
        folder: folder,
      });
    } catch (err) {
      cb(err);
    }
  }

  _removeFile(req, file, cb) {
    // B2 file deletion is handled by the deleteJobFiles function
    // This is called by multer on error; we don't need to do anything here
    cb(null);
  }
}

const storage = new B2Storage();

// ─── Multer instance ──────────────────────────────────────────────────────────
const upload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 1000 * 1024 * 1024, // 1000 MB per file (mobile videos can be large)
    files: 50,
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

    const fileName = options.folder ?
      `${options.folder}/${options.publicId || uuidv4()}`
      : `uploads/${options.publicId || uuidv4()}`;

    const { data: uploadData } = await b2.getUploadUrl({ bucketId: process.env.B2_BUCKET_ID });
    const { data: fileInfo } = await b2.uploadFile({
      uploadUrl: uploadData.uploadUrl,
      uploadAuthToken: uploadData.authorizationToken,
      fileName,
      data: buffer,
      contentType: 'application/octet-stream',
      info: options.tags ? { tags: options.tags } : {},
    });

    const downloadUrl = `${_downloadUrl}/file/${process.env.B2_BUCKET_NAME}/${fileName}`;

    return {
      fileId: fileInfo.fileId,
      fileName,
      downloadUrl: downloadUrl,
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

    // Delete all files matching the job ID pattern
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