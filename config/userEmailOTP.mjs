import { Resend } from 'resend';
import crypto from "crypto";

const resend = new Resend(process.env.RESEND_API_KEY);

// Generate OTP
export const generateOTP = () => {
    return crypto.randomInt(0, 10000).toString().padStart(4, "0");
};

// Send OTP email
export const sendOTPEmail = async (email, otp) => {
    try {
        const { error } = await resend.emails.send({
            from: 'onlinejobs.lat <noreply@onlinejobs.lat>',
            to: email,
            subject: 'Your OTP for 3D Maaka Signup',
            html: `
                <h2>Welcome to 3D Maakan</h2>
                <p>Your One-Time Password (OTP) is:</p>
                <h1 style="color: #45A7DE; font-size: 32px; letter-spacing: 5px;">${otp}</h1>
                <p>This OTP is valid for 10 minutes.</p>
                <p>Do not share this OTP with anyone.</p>
            `
        });

        if (error) {
            console.error("Resend error:", error);
            return false;
        }

        console.log(`OTP sent to ${email}`);
        return true;
    } catch (error) {
        console.error("Error sending OTP email:", error);
        return false;
    }
};