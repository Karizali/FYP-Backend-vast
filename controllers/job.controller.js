const Job = require('../models/Job');
const { generateDownloadUrl, deleteJobFiles } = require('../storage/cloudinaryStorage');
const notificationService = require('../services/notifications');
const logger = require('../utils/logger');

// ─── Verify request is from our trusted GPU worker ────────────────────────────
function verifyWorkerSecret(req) {
  // return req.headers['x-worker-secret'] === process.env.WORKER_SECRET;
  return true;   // skip auth for testing
}

// ─── GET /api/jobs ────────────────────────────────────────────────────────────
async function listJobs(req, res, next) {
  try {
    const page  = Math.max(1, parseInt(req.query.page)  || 1);
    const limit = Math.min(50, parseInt(req.query.limit) || 10);
    const skip  = (page - 1) * limit;

    const filter = { userId: req.user._id, deletedAt: null };
    if (req.query.status) filter.status = req.query.status;

    const [jobs, total] = await Promise.all([
      Job.find(filter)
        .sort({ createdAt: -1 })
        .skip(skip)
        .limit(limit)
        .select('-inputFiles.cloudinaryId -inputFiles.secureUrl -inputFiles.folder -output.glbCloudinaryId -output.thumbnailCloudinaryId'),
      Job.countDocuments(filter),
    ]);

    res.json({
      success: true,
      data: jobs.map((j) => j.toSummary()),
      pagination: {
        page, limit, total,
        pages:   Math.ceil(total / limit),
        hasNext: page * limit < total,
      },
    });
  } catch (error) { next(error); }
}

// ─── GET /api/jobs/:id ────────────────────────────────────────────────────────
async function getJob(req, res, next) {
  try {
    const job = await Job.findOne({ _id: req.params.id, userId: req.user._id, deletedAt: null });
    if (!job) return res.status(404).json({ success: false, message: 'Job not found.' });
    res.json({ success: true, data: job.toSummary() });
  } catch (error) { next(error); }
}

// ─── GET /api/jobs/:id/result ─────────────────────────────────────────────────
async function getJobResult(req, res, next) {
  try {
    const job = await Job.findOne({ _id: req.params.id, userId: req.user._id, deletedAt: null });
    if (!job) return res.status(404).json({ success: false, message: 'Job not found.' });

    if (job.status !== 'done') {
      return res.status(400).json({
        success: false,
        message: `Job is not complete yet. Current status: "${job.status}".`,
        status:  job.status, progressPct: job.progressPct,
      });
    }

    if (!job.output?.glbCloudinaryId) {
      return res.status(500).json({ success: false, message: 'Output file missing.' });
    }

    const [glbUrl, thumbnailUrl] = await Promise.all([
      generateDownloadUrl(job.output.glbCloudinaryId, 'raw', 86400),
      job.output.thumbnailCloudinaryId
        ? generateDownloadUrl(job.output.thumbnailCloudinaryId, 'image', 7200)
        : null,
    ]);

    res.json({
      success: true,
      data: {
        jobId:           job._id,
        title:           job.title,
        glbUrl,
        thumbnailUrl,
        glbExpiresAt:    new Date(Date.now() + 86400 * 1000).toISOString(),
        fileSizeBytes:   job.output.fileSizeBytes,
        durationSeconds: job.durationSeconds,
        completedAt:     job.timeline.completedAt,
      },
    });
  } catch (error) { next(error); }
}

// ─── DELETE /api/jobs/:id ─────────────────────────────────────────────────────
async function deleteJob(req, res, next) {
  try {
    const job = await Job.findOne({ _id: req.params.id, userId: req.user._id, deletedAt: null });
    if (!job) return res.status(404).json({ success: false, message: 'Job not found.' });

    if (job.isProcessing) {
      return res.status(400).json({
        success: false,
        message: `Cannot delete a job with status "${job.status}". Wait for it to finish.`,
      });
    }

    job.deletedAt = new Date();
    await job.save();

    deleteJobFiles(job._id.toString()).catch(err =>
      logger.warn(`Cloudinary cleanup failed for job ${job._id}: ${err.message}`)
    );

    res.json({ success: true, message: 'Job deleted.' });
  } catch (error) { next(error); }
}

// ─── POST /api/jobs/worker-dequeue ───────────────────────────────────────────
// Called by the Vast.ai worker every POLL_INTERVAL seconds.
// Returns the next queued job (with input file URLs) or 204 if queue is empty.
async function workerDequeue(req, res, next) {
  try {
    if (!verifyWorkerSecret(req)) {
      return res.status(401).json({ success: false, message: 'Unauthorized.' });
    }

    // Atomically find a queued job and lock it to prevent double-processing
    const job = await Job.findOneAndUpdate(
      { status: 'queued', deletedAt: null },
      { $set: { status: 'preprocessing', progressPct: 10, 'timeline.preprocessedAt': new Date() } },
      { sort: { createdAt: 1 }, new: true }   // oldest first
    );

    if (!job) return res.status(204).send();   // queue empty

    logger.info(`Worker dequeued job ${job._id}`);

    res.json({
      success: true,
      job: {
        jobId:      job._id.toString(),
        userId:     job.userId.toString(),
        inputFiles: job.inputFiles,
        inputType:  job.inputType,
        settings:   job.settings,
      },
    });
  } catch (error) { next(error); }
}

// ─── PATCH /api/jobs/:id/worker-update ───────────────────────────────────────
// Called by the Vast.ai worker to report progress and final result.
// Body: { status, progressPct?, output?, error? }
async function workerUpdate(req, res, next) {
  try {
    if (!verifyWorkerSecret(req)) {
      return res.status(401).json({ success: false, message: 'Unauthorized.' });
    }

    const { status, progressPct, output, error } = req.body;
    const job = await Job.findById(req.params.id);

    if (!job) return res.status(404).json({ success: false, message: 'Job not found.' });

    // Skip if already terminal
    if (job.status === 'done' || job.status === 'failed') {
      return res.json({ success: true, message: 'Already in terminal state.' });
    }

    logger.info(`Worker update: job ${job._id} → ${status} (${progressPct ?? '?'}%)`);

    if (status === 'failed') {
      await job.fail(
        error?.message || 'GPU worker error',
        error?.code    || 'WORKER_ERROR',
        error?.stage   || job.status,
      );
      await notificationService.notifyJobComplete(job.userId, job._id, 'failed').catch(() => {});
      return res.json({ success: true });
    }

    if (status === 'done') {
      if (!output?.glbCloudinaryId) {
        await job.fail('Worker completed but GLB output is missing.', 'MISSING_OUTPUT');
        return res.json({ success: true });
      }
      await job.transition('done', { progressPct: 100, output });
      await notificationService.notifyJobComplete(job.userId, job._id, 'done').catch(() => {});
      return res.json({ success: true });
    }

    // Intermediate progress update
    const extra = { progressPct: progressPct ?? job.progressPct };
    try {
      if (job.status !== status) await job.transition(status, extra);
      else await Job.findByIdAndUpdate(job._id, { progressPct });
    } catch (transitionErr) {
      logger.warn(`Worker update transition error: ${transitionErr.message}`);
    }

    res.json({ success: true });
  } catch (error) { next(error); }
}

module.exports = { listJobs, getJob, getJobResult, deleteJob, workerDequeue, workerUpdate };
