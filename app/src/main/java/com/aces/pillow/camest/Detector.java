package com.aces.pillow.camest;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import org.opencv.core.Point;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Size;

/**
 * Created by Administrator on 2017/6/2 0002.
 */

public class Detector {

    private static Context context;

    private static final String    TAG                 = Detector.class.getName();

    public static final int        JAVA_DETECTOR       = 0;

    private int                     mDetectorType       = JAVA_DETECTOR;

    private File                    mCascadeFile;
    private CascadeClassifier       mJavaDetector;

    private File                    mCascadeFileEye;
    private File                    mCascadeFileShoulder;

    private CascadeClassifier       mJavaDetectorEye;
    private CascadeClassifier       mJavaDetectorShoulder;

    private float                  mRelativeFaceSize   = 0.2f;
    private int                    mAbsoluteFaceSize   = 0;
    private double                 mEyeDistance;

    private boolean               FaceDetected = false;
    private boolean               ShoulderDetected = false;

    public Detector(Context curr) {
        context = curr;
    }

    public void load() {
        try {
            // load cascade file from application resources
            InputStream is = context.getResources().openRawResource(R.raw.haarcascade_frontalface_default);
            File cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
            mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_default.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            mJavaDetector.load(mCascadeFile.getAbsolutePath());
            if (mJavaDetector.empty()) {
                Log.e(TAG, "Failed to load cascade classifier");
                mJavaDetector = null;
            } else
                Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

            // load cascade file from application resources
            InputStream isEYE = context.getResources().openRawResource(R.raw.haarcascade_eye_tree_eyeglasses);
            File cascadeDirEYE = context.getDir("cascadeEYE", Context.MODE_PRIVATE);
            mCascadeFileEye = new File(cascadeDirEYE, "haarcascade_eye_tree_eyeglasses.xml");
            FileOutputStream osEYE = new FileOutputStream(mCascadeFileEye);

            byte[] bufferEYE = new byte[4096];
            int bytesReadEYE;
            while ((bytesReadEYE = isEYE.read(bufferEYE)) != -1) {
                osEYE.write(bufferEYE, 0, bytesReadEYE);
            }
            isEYE.close();
            osEYE.close();

            mJavaDetectorEye = new CascadeClassifier(mCascadeFileEye.getAbsolutePath());
            mJavaDetectorEye.load(mCascadeFileEye.getAbsolutePath());
            if (mJavaDetectorEye.empty()) {
                Log.e(TAG, "Failed to load cascade classifier");
                mJavaDetectorEye = null;
            } else
                Log.i(TAG, "Loaded cascade classifier for eye from " + mCascadeFile.getAbsolutePath());

            // load cascade file from application resources
            InputStream isShoulder = context.getResources().openRawResource(R.raw.haarcascade_mcs_upperbody);
            File cascadeDirShoulder = context.getDir("cascadeShoulder", Context.MODE_PRIVATE);
            mCascadeFileShoulder = new File(cascadeDirShoulder, "haarcascade_mcs_upperbody.xml");
            FileOutputStream osShoulder = new FileOutputStream(mCascadeFileShoulder);

            byte[] bufferShoulder = new byte[4096];
            int bytesReadShoulder;
            while ((bytesReadShoulder = isShoulder.read(bufferShoulder)) != -1) {
                osShoulder.write(bufferShoulder, 0, bytesReadShoulder);
            }
            isShoulder.close();
            osShoulder.close();

            mJavaDetectorShoulder = new CascadeClassifier(mCascadeFileShoulder.getAbsolutePath());
            mJavaDetectorShoulder.load(mCascadeFileShoulder.getAbsolutePath());
            if (mJavaDetectorShoulder.empty()) {
                Log.e(TAG, "Failed to load cascade classifier");
                mJavaDetectorShoulder = null;
            } else
                Log.i(TAG, "Loaded cascade classifier for upper body from " + mCascadeFile.getAbsolutePath());

            cascadeDir.delete();
            cascadeDirEYE.delete();
            cascadeDirShoulder.delete();

        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
        }
    }

    public boolean isFaceDetected() {
        return FaceDetected;
    }

    public boolean isShoulderDetected() {
        return ShoulderDetected;
    }

    public void setFaceDetected(boolean b) {
        FaceDetected = b;
    }

    public void setShoulderDetected(boolean b) {ShoulderDetected = b;}

    public double getEyeDistance() { return mEyeDistance; }

    public Rect FaceDetector (Mat gray) {

        MatOfRect faces = new MatOfRect();
        MatOfRect eyes = new MatOfRect();

        if (mAbsoluteFaceSize == 0) {
            int height = gray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }

        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null)
                mJavaDetector.detectMultiScale(gray, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }
        else {
            Log.e(TAG, "Detection method is not selected!");
            return null;
        }

        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++) {

            Mat mFace = gray.submat(facesArray[i]);

            if (mJavaDetectorEye != null) {
                mJavaDetectorEye.detectMultiScale(mFace, eyes, 1.1, 2,
                        Objdetect.CASCADE_SCALE_IMAGE, new Size(0, 0), new Size());
                if (!eyes.empty()) {
                    Rect[] eyeArray = eyes.toArray();
                    if (eyeArray.length == 2) {
                        FaceDetected = true;
                        Point p0 = new Point((eyeArray[0].br().x + eyeArray[0].tl().x) / 2,
                                (eyeArray[0].br().y + eyeArray[0].tl().y) / 2);
                        Point p1 = new Point((eyeArray[1].br().x + eyeArray[1].tl().x) / 2,
                                (eyeArray[1].br().y + eyeArray[1].tl().y) / 2);
                        mEyeDistance = Math.sqrt((double) ((p1.x - p0.x) * (p1.x - p0.x) + (p1.y - p0.y) * (p1.y - p0.y)));
                        return facesArray[i];
                    }
                }
            }
        }

        return null;
    }

    public Rect ShoulderDetector (Mat gray) {

        MatOfRect shoulders = new MatOfRect();
        MatOfRect eyes = new MatOfRect();

        if (mAbsoluteFaceSize == 0) {
            int height = gray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }

        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetectorShoulder != null)
                mJavaDetectorShoulder.detectMultiScale(gray, shoulders, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }
        else {
            Log.e(TAG, "Detection method is not selected!");
            return null;
        }

        int max_idx = 0;
        if (!shoulders.empty()) {
            Rect[] shouldersArray = shoulders.toArray();
            for (int i = 1; i < shouldersArray.length; i++) {

                if (shouldersArray[i].width > shouldersArray[max_idx].width)
                    max_idx = i;
            }

            ShoulderDetected = true;
            return shouldersArray[max_idx];
        }

//        if (!shoulders.empty()) {
//            Rect[] shouldersArray = shoulders.toArray();
//            for (int i = 1; i < shouldersArray.length; i++) {
//
//                Mat mShoulder = gray.submat(shouldersArray[i]);
//                if (mJavaDetectorEye != null) {
//                    mJavaDetectorEye.detectMultiScale(mShoulder, eyes, 1.1, 2,
//                            Objdetect.CASCADE_SCALE_IMAGE, new Size(0, 0), new Size());
//                    if (!eyes.empty()) {
//                        ShoulderDetected = true;
//                        return shouldersArray[i];
//                    }
//                }
//            }
//        }

        return null;
    }
}
