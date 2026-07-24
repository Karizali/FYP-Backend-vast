// routes/splat.js


const express  = require('express');
const { authenticate } = require('../middleware/auth');
const Job = require('../models/Job');
const { Readable } = require('stream'); 
const router = express.Router();

// All storage routes require authentication
// In production, add an isAdmin middleware here
// router.use(authenticate);

router.get('/:jobId', async (req, res) => {
  try {
    console.log('Splat proxy request for jobId:', req.params.jobId);
    const jobId = req.params.jobId.replace(/_scene\.ply$/i, '').replace(/\.ply$/i, '');

    const job = await Job.findById(jobId);

    if (!job) {
      return res.status(404).json({ error: 'Job not found' });
    }

    // if (job.userId.toString() !== req.user.id.toString()) {
    //   return res.status(403).json({ error: 'Forbidden' });
    // }

    const fileUrl = `https://f005.backblazeb2.com/file/3d-guassian/outputs/${job._id}_scene.ply`;
    console.log('Fetching from B2:', fileUrl);

    const b2Response = await fetch(fileUrl);

    if (!b2Response.ok) {
      console.error('B2 fetch failed:', b2Response.status);
      return res.status(502).json({ error: 'Failed to fetch from storage' });
    }

    const contentLength = b2Response.headers.get('content-length');
    if (contentLength) res.setHeader('Content-Length', contentLength);
    res.setHeader('Content-Type', 'application/octet-stream');

    Readable.fromWeb(b2Response.body).pipe(res);

  } catch (err) {
    console.error('Splat proxy error:', err);
    if (!res.headersSent) res.status(500).json({ error: 'Internal server error' });
  }
});

module.exports = router;