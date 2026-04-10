const express = require('express');
const { param, query } = require('express-validator');
const { authenticate } = require('../middleware/auth');
const {
  listJobs, getJob, getJobResult, deleteJob,
  workerDequeue, workerUpdate,
} = require('../controllers/job.controller');

const router = express.Router();

// ─── Validation helpers ───────────────────────────────────────────────────────
const jobIdParam = [
  param('id').isMongoId().withMessage('Invalid job ID format.'),
];

const listQueryRules = [
  query('page').optional().isInt({ min: 1 }),
  query('limit').optional().isInt({ min: 1, max: 50 }),
  query('status').optional().isIn(['queued','preprocessing','training','converting','done','failed']),
];

// ─── Worker routes (no JWT — authenticated by X-Worker-Secret header) ─────────
// Must be defined BEFORE router.use(authenticate) so they aren't JWT-blocked
router.post('/worker-dequeue',     workerDequeue);
router.patch('/:id/worker-update', jobIdParam, workerUpdate);

// ─── User routes (JWT required) ───────────────────────────────────────────────
router.use(authenticate);

router.get('/',           listQueryRules, listJobs);
router.get('/:id',        jobIdParam, getJob);
router.get('/:id/result', jobIdParam, getJobResult);
router.delete('/:id',     jobIdParam, deleteJob);

module.exports = router;
