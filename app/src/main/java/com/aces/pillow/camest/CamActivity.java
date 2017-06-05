package com.aces.pillow.camest;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Rect;

import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;

public class CamActivity extends Activity implements CvCameraViewListener2 {

    private static final String    TAG                 = CamActivity.class.getName();

    private static final Scalar    FACE_RECT_COLOR     = new Scalar(0, 255, 0, 127);

    private static final int      thickness = 2;

    private static final int      MAXIMUM_ALLOWED_SKIPPED_FRAMES = 15;

    private static final int      MAXIMUM_ALLOWED_UNDETECTED_FRAMES = 30;

    private int                    shoulder_frame_count = 0;
    private int                    face_frame_count = 0;

    private double                width = 0;

    private CameraBridgeViewBase   mOpenCvCameraView;

    private Mat                    mRgba;
    private Mat                    mGray;

    private Detector               mDetector;

    private Rect                   face_frame;
    private Rect                   shoulder_frame;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    mDetector.load();

                    mOpenCvCameraView.setCameraIndex(1);
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public CamActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_cam);

        mDetector = new Detector(this);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.cam_activity_surface_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();

        face_frame = new Rect();
        shoulder_frame = new Rect();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        if (face_frame_count == 0) {
            face_frame = mDetector.FaceDetector(mGray);
        }

        if (mDetector.isFaceDetected()) {
            if (face_frame_count > MAXIMUM_ALLOWED_SKIPPED_FRAMES) {
                face_frame_count = 0;
                face_frame = null;
                mDetector.setFaceDetected(false);
            } else {
                if (!mDetector.isShoulderDetected()) {
                    shoulder_frame = mDetector.ShoulderDetector(mGray);
                    if (mDetector.isShoulderDetected() &&
                            (shoulder_frame.width > face_frame.width) && (shoulder_frame.height
                            > face_frame.height)) {
                        this.width = (shoulder_frame.width - face_frame.width) / 2;
                        shoulder_frame_count = 1;
                    }
                    else {
                        mDetector.setShoulderDetected(false);
                        shoulder_frame = null;
                    }
                }
            }

        }

        if (mDetector.isFaceDetected()) {
            show_face();
            face_frame_count++;
            if (mDetector.isShoulderDetected()) {
                shoulder_frame_count++;
                if (shoulder_frame_count > MAXIMUM_ALLOWED_SKIPPED_FRAMES) {
                    mDetector.setShoulderDetected(false);
                    shoulder_frame = null;
                    shoulder_frame_count = 0;
                } else show_shoulder();
            }
        }

        return mRgba;
    }

    public double getWidth() {
        return width;
    }

    private void show_face() {
        if (face_frame != null) {
            Imgproc.rectangle(mRgba, face_frame.tl(), face_frame.br(), FACE_RECT_COLOR, thickness);
        }
    }

    private void show_shoulder() {
        if (shoulder_frame != null)
            Imgproc.rectangle(mRgba, shoulder_frame.tl(), shoulder_frame.br(), FACE_RECT_COLOR, thickness);
    }

}
