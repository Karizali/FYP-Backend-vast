const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');
const { validationResult } = require('express-validator');
const User = require('../models/User');
const logger = require('../utils/logger');
const Otp = require('../models/otp');
const { sendOTPEmail, generateOTP } = require('../config/userEmailOTP.js');

// ─── Helper: sign a JWT ───────────────────────────────────────────────────────
function signToken(userId) {
  return jwt.sign(
    { id: userId },
    process.env.JWT_SECRET,
    { expiresIn: process.env.JWT_EXPIRES_IN || '7d' }
  );
}

// ─── POST /api/auth/register ──────────────────────────────────────────────────
async function register(req, res, next) {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ success: false, errors: errors.array() });
    }

    const { name, email, password } = req.body;

    const existing = await User.findOne({ email });
    if (existing) {
      return res.status(409).json({
        success: false,
        message: 'An account with this email already exists.',
      });
    }

    const user = await User.create({ name, email, password });
    const token = signToken(user._id);

    logger.info(`New user registered: ${user.email}`);

    res.status(201).json({
      success: true,
      message: 'Account created successfully.',
      token,
      user: user.toPublic(),
    });
  } catch (error) {
    next(error);
  }
}

// ─── POST /api/auth/login ─────────────────────────────────────────────────────
async function login(req, res, next) {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ success: false, errors: errors.array() });
    }

    const { email, password } = req.body;

    // Explicitly select password (it has select:false in the schema)
    const user = await User.findOne({ email }).select('+password');
    if (!user || !(await user.comparePassword(password))) {
      // Intentionally vague — don't reveal whether email exists
      return res.status(401).json({
        success: false,
        message: 'Invalid email or password.',
      });
    }

    if (!user.isActive) {
      return res.status(403).json({
        success: false,
        message: 'Your account has been deactivated. Please contact support.',
      });
    }

    // Stamp last login time
    user.lastLoginAt = new Date();
    await user.save({ validateBeforeSave: false });

    const token = signToken(user._id);
    logger.info(`User logged in: ${user.email}`);

    // Cookie options
    const cookieOptions = {
      httpOnly: true,
      secure: true,
      sameSite: 'None',
      maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
    };

    res.cookie('token', token, cookieOptions);

    res.json({
      success: true,
      message: 'Logged in successfully.',
      token,
      user: user.toPublic(),
    });
  } catch (error) {
    next(error);
  }
}

// ─── GET /api/auth/me ─────────────────────────────────────────────────────────
async function getProfile(req, res) {
  // req.user is already populated by the authenticate middleware
  res.json({
    success: true,
    user: req.user.toPublic(),
  });
}

// ─── PUT /api/auth/me ─────────────────────────────────────────────────────────
async function updateProfile(req, res, next) {
  try {
    const { name } = req.body;

    if (!name || name.trim().length === 0) {
      return res.status(400).json({ success: false, message: 'Name is required.' });
    }

    req.user.name = name.trim();
    await req.user.save();

    res.json({
      success: true,
      message: 'Profile updated.',
      user: req.user.toPublic(),
    });
  } catch (error) {
    next(error);
  }
}

// ─── PUT /api/auth/password ───────────────────────────────────────────────────
async function changePassword(req, res, next) {
  try {
    const { currentPassword, newPassword } = req.body;

    if (!currentPassword || !newPassword) {
      return res.status(400).json({
        success: false,
        message: 'currentPassword and newPassword are required.',
      });
    }
    if (newPassword.length < 8) {
      return res.status(400).json({
        success: false,
        message: 'New password must be at least 8 characters.',
      });
    }

    // Re-fetch with password field
    const user = await User.findById(req.user._id).select('+password');
    if (!(await user.comparePassword(currentPassword))) {
      return res.status(401).json({ success: false, message: 'Current password is incorrect.' });
    }

    user.password = newPassword;
    await user.save();

    res.json({ success: true, message: 'Password changed successfully.' });
  } catch (error) {
    next(error);
  }
}

// ─── PUT /api/auth/fcm-token ──────────────────────────────────────────────────
// Flutter app calls this after login to register its push notification token
async function updateFcmToken(req, res, next) {
  try {
    const { fcmToken } = req.body;

    if (!fcmToken || typeof fcmToken !== 'string') {
      return res.status(400).json({ success: false, message: 'fcmToken is required.' });
    }

    await User.findByIdAndUpdate(req.user._id, { fcmToken });
    logger.info(`FCM token updated for user ${req.user._id}`);

    res.json({ success: true, message: 'Push notification token registered.' });
  } catch (error) {
    next(error);
  }
}

// ─── POST /api/auth/forget-password ──────────────────────────────────────────
// Step 1: Validate email, generate OTP, send to user's email
async function forgetPassword(req, res, next) {
  try {
    const { email } = req.body;

    if (!email) {
      return res.status(400).json({ success: false, message: 'Email is required.' });
    }

    const normalizedEmail = email.toLowerCase();

    const user = await User.findOne({ email: normalizedEmail });
    if (!user) {
      return res.status(400).json({ success: false, message: 'No account found with this email.' });
    }

    const otp = generateOTP();
    const otpExpiry = new Date(Date.now() + 10 * 60 * 1000); // 10 minutes

    let otpHash;
    try {
      const salt = await bcrypt.genSalt();
      otpHash = await bcrypt.hash(otp, salt);
    } catch (error) {
      logger.error('OTP hash error:', error);
      return next(error);
    }

    await Otp.findOneAndUpdate(
      { email: normalizedEmail },
      { email: normalizedEmail, otp: otpHash, expiresAt: otpExpiry, createdAt: new Date() },
      { upsert: true, new: true }
    );

    const emailSent = await sendOTPEmail(normalizedEmail, otp);
    if (!emailSent) {
      return res.status(500).json({ success: false, message: 'Failed to send OTP email. Please try again.' });
    }

    logger.info(`Forget-password OTP sent to ${normalizedEmail}`);

    res.status(200).json({
      success: true,
      message: 'OTP sent to your email. It expires in 10 minutes.',
      email: normalizedEmail,
    });
  } catch (error) {
    next(error);
  }
}

// ─── POST /api/auth/forget-password/verify ───────────────────────────────────
// Step 2: Verify OTP and set new password
async function forgetPasswordVerify(req, res, next) {
  try {
    const { email, otp, newPassword } = req.body;

    if (!email || !otp || !newPassword) {
      return res.status(400).json({
        success: false,
        message: 'email, otp, and newPassword are required.',
      });
    }

    if (newPassword.length < 8) {
      return res.status(400).json({
        success: false,
        message: 'New password must be at least 8 characters.',
      });
    }

    const normalizedEmail = email.toLowerCase();

    const otpRecord = await Otp.findOne({ email: normalizedEmail });
    if (!otpRecord) {
      return res.status(400).json({ success: false, message: 'OTP not found or already expired.' });
    }

    if (new Date() > otpRecord.expiresAt) {
      await Otp.deleteOne({ email: normalizedEmail });
      return res.status(400).json({ success: false, message: 'OTP has expired. Please request a new one.' });
    }

    const isMatch = await bcrypt.compare(otp, otpRecord.otp);
    if (!isMatch) {
      return res.status(400).json({ success: false, message: 'Invalid OTP.' });
    }

    let passwordHash;
    try {
      const salt = await bcrypt.genSalt();
      passwordHash = await bcrypt.hash(newPassword, salt);
    } catch (error) {
      logger.error('Password hash error:', error);
      return next(error);
    }

    await User.updateOne(
      { email: normalizedEmail },
      { $set: { password: passwordHash } }
    );

    await Otp.deleteOne({ email: normalizedEmail });

    logger.info(`Password reset successful for ${normalizedEmail}`);

    res.status(200).json({ success: true, message: 'Password reset successfully.' });
  } catch (error) {
    next(error);
  }
}

module.exports = {
  register,
  login,
  getProfile,
  updateProfile,
  changePassword,
  updateFcmToken,
  forgetPassword,
  forgetPasswordVerify,
};