const { v4: uuidv4 } = require('uuid');
const { upload } = require('../config/storage');
const Job    = require('../models/Job');
const logger = require('../utils/logger');

// ─── Middleware: assign a folderKey BEFORE multer runs ────────────────────────
// Used only for the Cloudinary folder path — NOT used as the MongoDB _id.
function assignJobId(req, res, next) {
  req.jobId = uuidv4();
  next();
}

// ─── Middleware: check plan limits ────────────────────────────────────────────
function checkPlanLimit(req, res, next) {
  if (!req.user.canCreateJob()) {
    return res.status(403).json({
      success:       false,
      message:       `Monthly job limit reached for your "${req.user.plan}" plan.`,
      jobsRemaining: req.user.jobsRemaining,
      plan:          req.user.plan,
    });
  }
  next();
}

// ─── Middleware: validate quality setting against plan ─────────────────────────
function checkQualitySetting(req, res, next) {
  if (req.body.quality === 'high' && req.user.plan === 'free') {
    return res.status(403).json({
      success: false,
      message: 'High quality is only available on the Pro plan. Use "fast" or "balanced".',
    });
  }
  next();
}

// ─── Main upload handler ───────────────────────────────────────────────────────
const handleUpload = [
  checkPlanLimit,
  checkQualitySetting,
  assignJobId,

  upload.array('files', 50),

  async (req, res, next) => {
    try {
      if (!req.files || req.files.length === 0) {
        return res.status(400).json({
          success: false,
          message: 'No files uploaded. Include files under the "files" field.',
        });
      }

      const mimeTypes = req.files.map((f) => f.mimetype);
      const hasVideo  = mimeTypes.some((m) => m.startsWith('video/'));
      const inputType = hasVideo ? 'video' : 'images';

      if (hasVideo && req.files.length > 1) {
        return res.status(400).json({
          success: false,
          message: 'Only one video file is allowed per job.',
        });
      }

      if (!hasVideo && req.files.length < 5) {
        return res.status(400).json({
          success: false,
          message: `Too few images (${req.files.length}). Upload at least 5; 20–50 recommended.`,
        });
      }

      const inputFiles = req.files.map((file) => ({
        originalName: file.originalname,
        cloudinaryId: file.filename,
        secureUrl:    file.path,
        resourceType: hasVideo ? 'video' : 'image',
        mimeType:     file.mimetype,
        sizeBytes:    file.size,
        folder:       file.folder || null,
      }));

      const settings = {
        enhanceImages: req.body.enhanceImages !== 'false',
        quality: ['fast', 'balanced', 'high'].includes(req.body.quality)
          ? req.body.quality
          : 'balanced',
      };

      // ── Create job in MongoDB with status "queued" ────────────────────────
      // The Vast.ai worker polls GET /api/jobs/worker-dequeue and picks this up
      const job = await Job.create({
        userId:    req.user._id,
        title:     req.body.title?.trim() || `Scene – ${new Date().toLocaleDateString()}`,
        inputType,
        inputFiles,
        settings,
      });

      await req.user.updateOne({ $inc: { jobsThisMonth: 1 } });

      const jobId = job._id.toString();
      logger.info(`Job ${jobId} created | user=${req.user._id} | files=${req.files.length} | type=${inputType}`);

      res.status(201).json({
        success:          true,
        message:          'Files uploaded. Your 3D scene is being processed.',
        jobId,
        status:           job.status,
        filesUploaded:    req.files.length,
        inputType,
        estimatedMinutes: job.estimatedMinutes,
        jobsRemaining:    req.user.jobsRemaining - 1,
      });
    } catch (error) {
      next(error);
    }
  },
];

module.exports = { handleUpload };