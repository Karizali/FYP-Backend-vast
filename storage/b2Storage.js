const B2 = require('backblaze-b2');
const logger = require('../utils/logger');

// ─── Layer 3: Processed Output Storage ───────────────────────────────────────
// All Backblaze B2 operations for GPU output files (GLB, thumbnails).
// Input uploads (images/video) are handled by config/storage.js (multer layer).
// This module owns everything that happens AFTER the GPU worker finishes.

// ─── Initialize B2 client ────────────────────────────────────────────────────
const b2 = new B2({
  applicationKeyId: process.env.B2_KEY_ID,
  applicationKey:   process.env.B2_APP_KEY,
});

let _isAuthorized = false;

async function ensureAuthorized() {
  if (_isAuthorized) return;
  try {
    await b2.authorize();
    _isAuthorized = true;
    logger.info('✅ Backblaze B2 authorized');
  } catch (err) {
    logger.error(`Failed to authorize B2: ${err.message}`);
    throw err;
  }
}

// ─── Upload a GLB file ────────────────────────────────────────────────────────
/**
 * Upload the finished .glb file to Backblaze B2.
 * Called by the Node.js worker (processJob.js) after Runpod completes,
 * OR the result can come directly from the Python worker via the webhook.
 *
 * @param {string|Buffer} source   - file path or Buffer
 * @param {string}        jobId    - MongoDB Job._id
 * @param {string}        userId   - MongoDB User._id
 * @returns {Promise<{ b2Id, downloadUrl, fileSizeBytes }>}
 */
async function uploadGlb(source, jobId, userId) {
  try {
    await ensureAuthorized();
    logger.info(`Storage: uploading GLB for job ${jobId}`);
    
    const fs = require('fs');
    const fileName = `outputs/${jobId}/scene.glb`;
    
    // Determine if source is a file path or buffer
    let fileData;
    if (typeof source === 'string') {
      fileData = fs.readFileSync(source);
    } else {
      fileData = source;
    }

    const fileInfo = await b2.uploadFile({
      uploadUrl: await b2.getUploadUrl(process.env.B2_BUCKET_NAME),
      fileName,
      data: fileData,
      contentType: 'application/octet-stream',
      info: {
        jobId,
        userId,
      },
    });

    const downloadUrl = await b2.getDownloadUrl(process.env.B2_BUCKET_NAME, fileName);
    
    logger.info(`Storage: GLB uploaded | id=${fileInfo.fileId} | size=${fileData.length}B`);

    return {
      b2Id:           fileInfo.fileId,
      downloadUrl:    downloadUrl,
      fileSizeBytes:  fileData.length,
    };
  } catch (err) {
    logger.error(`Storage: GLB upload failed for job ${jobId}: ${err.message}`);
    const e        = new Error(`Failed to save GLB output: ${err.message}`);
    e.code         = 'STORAGE_GLB_UPLOAD_FAILED';
    e.userMessage  = 'Could not save the 3D scene file. Please try again.';
    throw e;
  }
}

// ─── Upload a thumbnail image ─────────────────────────────────────────────────
/**
 * Stores a JPEG preview thumbnail for the Flutter jobs list screen.
 * Non-fatal — if this fails the job still completes, just without a preview.
 *
 * @param {string|Buffer} source
 * @param {string}        jobId
 * @param {string}        userId
 * @returns {Promise<{ b2Id, downloadUrl } | null>}
 */
async function uploadThumbnail(source, jobId, userId) {
  try {
    await ensureAuthorized();
    const fs = require('fs');
    const fileName = `outputs/${jobId}/thumbnail.jpg`;
    
    // Determine if source is a file path or buffer
    let fileData;
    if (typeof source === 'string') {
      fileData = fs.readFileSync(source);
    } else {
      fileData = source;
    }

    const fileInfo = await b2.uploadFile({
      uploadUrl: await b2.getUploadUrl(process.env.B2_BUCKET_NAME),
      fileName,
      data: fileData,
      contentType: 'image/jpeg',
      info: {
        jobId,
        userId,
      },
    });

    const downloadUrl = await b2.getDownloadUrl(process.env.B2_BUCKET_NAME, fileName);
    
    logger.info(`Storage: thumbnail uploaded | id=${fileInfo.fileId}`);
    return { b2Id: fileInfo.fileId, downloadUrl };
  } catch (err) {
    logger.warn(`Storage: thumbnail upload failed for job ${jobId} (non-fatal): ${err.message}`);
    return null;
  }
}

// ─── Upload from a Node.js stream ────────────────────────────────────────────
/**
 * Used when the GPU worker streams a GLB directly without saving to disk.
 *
 * @param {ReadableStream} stream
 * @param {{ jobId, userId, fileName? }} options
 * @returns {Promise<Object>} B2 upload result
 */
async function uploadStream(stream, options = {}) {
  const { jobId, userId, fileName = `outputs/${jobId}/scene.glb` } = options;

  try {
    await ensureAuthorized();
    
    // Convert stream to buffer
    const chunks = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
    }
    const data = Buffer.concat(chunks);

    const fileInfo = await b2.uploadFile({
      uploadUrl: await b2.getUploadUrl(process.env.B2_BUCKET_NAME),
      fileName,
      data,
      contentType: 'application/octet-stream',
      info: {
        jobId,
        userId,
      },
    });

    const downloadUrl = await b2.getDownloadUrl(process.env.B2_BUCKET_NAME, fileName);

    return {
      b2Id:       fileInfo.fileId,
      downloadUrl: downloadUrl,
      fileSize:   data.length,
    };
  } catch (err) {
    logger.error(`Storage: stream upload failed for job ${jobId}: ${err.message}`);
    throw err;
  }
}

// ─── Generate a download URL ─────────────────────────────────────────────────
async function generateDownloadUrl(b2FileName) {
  if (!b2FileName) return null;
  
  try {
    await ensureAuthorized();
    const url = await b2.getDownloadUrl(process.env.B2_BUCKET_NAME, b2FileName);
    logger.debug(`Storage: B2 download link generated for ${b2FileName}`);
    return url;
  } catch (err) {
    logger.error(`Failed to generate download URL: ${err.message}`);
    return null;
  }
}

// ─── Delete a single file ─────────────────────────────────────────────────────
async function deleteFile(b2FileId) {
  try {
    await ensureAuthorized();
    await b2.deleteFile(b2FileId);
    logger.info(`Storage: deleted B2 file ${b2FileId}`);
    return true;
  } catch (err) {
    logger.warn(`Storage: delete failed for ${b2FileId}: ${err.message}`);
    return false;
  }
}

// ─── Delete all files for a job ───────────────────────────────────────────────
/**
 * Deletes ALL B2 files in the job's output folder.
 * Called by the cleanup cron job for soft-deleted jobs.
 *
 * @param {string} jobId
 */
async function deleteJobFiles(jobId) {
  try {
    await ensureAuthorized();
    const prefix = `outputs/${jobId}/`;
    
    let deleted = 0;
    const files = await b2.listFileNames(process.env.B2_BUCKET_NAME, { prefix });
    
    for (const file of files) {
      try {
        await b2.deleteFile(file.fileId);
        deleted++;
      } catch (err) {
        logger.warn(`Storage: could not delete file ${file.fileId}: ${err.message}`);
      }
    }

    logger.info(`Storage: cleanup complete for job ${jobId} | deleted=${deleted} files`);
    return { deleted };
  } catch (err) {
    logger.warn(`Storage: could not delete files for job ${jobId}: ${err.message}`);
    return { deleted: 0 };
  }
}

// ─── Get B2 usage stats ───────────────────────────────────────────────────────
/**
 * Returns account-level storage stats.
 * Used by the admin /storage/stats route.
 *
 * @returns {Promise<Object>}
 */
async function getUsageStats() {
  try {
    await ensureAuthorized();
    
    // Get bucket information
    const bucketInfo = await b2.getBucketInfo(process.env.B2_BUCKET_NAME);
    
    // List files to calculate total size
    let totalBytes = 0;
    const files = await b2.listFileNames(process.env.B2_BUCKET_NAME);
    
    for (const file of files) {
      totalBytes += file.size || 0;
    }

    const usedGB = (totalBytes / 1e9).toFixed(2);
    
    return {
      storage: {
        usedBytes:  totalBytes,
        usedGB:     usedGB,
        limitGB:    'Unlimited',
        usedPct:    'N/A',
      },
      bucket:     process.env.B2_BUCKET_NAME,
      retrievedAt: new Date().toISOString(),
    };
  } catch (err) {
    logger.error(`Failed to get B2 usage stats: ${err.message}`);
    return {
      storage: { usedGB: 'N/A', limitGB: 'N/A', usedPct: 'N/A' },
      bucket: process.env.B2_BUCKET_NAME,
      retrievedAt: new Date().toISOString(),
    };
  }
}

// ─── Verify Backblaze B2 credentials at startup ─────────────────────────────
async function verifyConnection() {
  try {
    await ensureAuthorized();
    logger.info('✅ Backblaze B2 connected');
    return true;
  } catch (err) {
    logger.error(`❌ Backblaze B2 connection failed: ${err.message}`);
    return false;
  }
}

// ─── List all output files for a job (admin/debug) ───────────────────────────
async function listJobOutputs(jobId) {
  try {
    await ensureAuthorized();
    const prefix = `outputs/${jobId}/`;
    
    const files = await b2.listFileNames(process.env.B2_BUCKET_NAME, { prefix });
    return files.map(f => ({
      fileId:      f.fileId,
      fileName:    f.fileName,
      uploadTime:  f.uploadTimestamp,
      fileSize:    f.size,
    }));
  } catch (err) {
    logger.warn(`Storage: could not list outputs for job ${jobId}: ${err.message}`);
    return [];
  }
}

module.exports = {
  uploadGlb,
  uploadThumbnail,
  uploadStream,
  generateDownloadUrl,
  deleteFile,
  deleteJobFiles,
  getUsageStats,
  verifyConnection,
  listJobOutputs,
};
